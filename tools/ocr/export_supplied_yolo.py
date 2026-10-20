"""Export the three supplied YOLOv8 checkpoints to ONNX staging and verify numeric parity.

CLI usage:
    .cache/yolo-env/Scripts/python.exe -m tools.ocr.export_supplied_yolo --models models --out artifacts/supplied-yolo
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata as importlib_metadata
import io
import json
import logging
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import PIL
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER

SEMANTICS_VERSION = "supplied-yolo-3stage-v1"
PIPELINE_ID = "supplied-yolo-3stage-v1"
PREPROCESS_VERSION = "opencv-linear-byte-letterbox-rgb-div255-pad114-obb-warp-v1"

EXPECTED_CHECKPOINTS: dict[str, dict[str, Any]] = {
    "frame": {
        "sourceFilename": "best_latest.pt",
        "onnxFilename": "frame.onnx",
        "expectedSha256": "2f1d73f4ca887b3a59cfe263163ac09f538100b84cb9837c91edd7dc3c105675",
        "expectedSizeBytes": 6427020,
        "task": "obb",
        "imgsz": 640,
        "expectedInputShape": [1, 3, 640, 640],
        "expectedOutputShape": [1, 6, 8400],
        "outputLayout": "BCN:[1, 6, 8400] where C=[cx, cy, w, h, score_khung, angle_rad]",
        "nc": 1,
        "expectedNames": {"0": "khung"},
        "confThres": 0.25,
        "iouThres": 0.50,
        "maxDet": 300,
        "nmsMethod": "probiou_greedy",
    },
    "digits": {
        "sourceFilename": "best_0608.pt",
        "onnxFilename": "digits.onnx",
        "expectedSha256": "c475d9e8753f20daae4c405e99b03dfff4868c02926965fce6060743a692a9bd",
        "expectedSizeBytes": 6225635,
        "task": "detect",
        "imgsz": 320,
        "expectedInputShape": [1, 3, 320, 320],
        "expectedOutputShape": [1, 34, 2100],
        "outputLayout": "BCN:[1, 34, 2100] where C=[cx, cy, w, h, score_0..score_29]",
        "nc": 30,
        "expectedNames": {str(i): str(i) for i in range(30)},
        "confThres": 0.25,
        "iouThres": 0.50,
        "agnosticNms": True,
        "maxDet": 300,
        "mappingRule": "int(label) % 10",
        "classToDigit": {str(i): str(i % 10) for i in range(30)},
        "sortOrder": "left_to_right_x1_then_cx_then_anchor",
    },
    "reverse": {
        "sourceFilename": "best_1608.pt",
        "onnxFilename": "reverse.onnx",
        "expectedSha256": "17413d10235061c85a6da177b414b38eaaf73b1ab70ee74317cdb018a267fe25",
        "expectedSizeBytes": 6216035,
        "task": "detect",
        "imgsz": 320,
        "expectedInputShape": [1, 3, 320, 320],
        "expectedOutputShape": [1, 10, 2100],
        "outputLayout": "BCN:[1, 10, 2100] where C=[cx, cy, w, h, score_0..score_5]",
        "nc": 6,
        "expectedNames": {
            "0": "1n",
            "1": "2n",
            "2": "3n",
            "3": "4n",
            "4": "5n",
            "5": "7n",
        },
        "confThres": 0.25,
        "iouThres": 0.50,
        "agnosticNms": True,
        "maxDet": 300,
        "triggerRule": "any_asymmetric_or_at_least_two",
        "asymmetricLabels": ["2n", "3n", "4n", "5n", "7n"],
        "symmetricLabels": ["1n"],
        "minTotalBoxesForSymmetricTrigger": 2,
        "actionOnTrigger": "rotate_crop_180_then_run_digits",
    },
}

OUT_OF_SCOPE_CHECKPOINT = {
    "filename": "best_2107.pt",
    "expectedSha256": "b4f7ffc3a91c7864efb58bcbb4dd558e4a6e09ff8c793c6685f4144347f16496",
    "reason": "Out of scope per Outcome Contract; retained untouched in models/",
}

RAW_TENSOR_ATOL = 1e-4
RAW_TENSOR_RTOL = 1e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_environment_lock(out_dir: Path) -> dict[str, Any]:
    dist_map: dict[str, str] = {}
    for dist in importlib_metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            dist_map[str(name)] = str(dist.version)
    pip_freeze = [f"{name}=={dist_map[name]}" for name in sorted(dist_map, key=str.lower)]

    lock_txt = out_dir / "requirements-lock.txt"
    lock_txt.write_text("\n".join(pip_freeze) + "\n", encoding="utf-8")

    env_info = {
        "pythonVersion": sys.version,
        "pythonExecutable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "requirementsLockFile": lock_txt.name,
        "requirementsLockSha256": sha256_file(lock_txt),
        "packageCount": len(pip_freeze),
    }
    (out_dir / "python-env-lock.json").write_text(
        json.dumps(env_info, indent=2) + "\n", encoding="utf-8"
    )
    return env_info


def inspect_checkpoint(pt_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if not pt_path.exists():
        raise FileNotFoundError(f"Missing required checkpoint: {pt_path}")
    actual_sha = sha256_file(pt_path)
    actual_size = pt_path.stat().st_size
    if actual_sha != spec["expectedSha256"]:
        raise ValueError(
            f"Checkpoint SHA-256 mismatch for {pt_path}: "
            f"expected {spec['expectedSha256']}, got {actual_sha}"
        )
    if actual_size != spec["expectedSizeBytes"]:
        raise ValueError(
            f"Checkpoint size mismatch for {pt_path}: "
            f"expected {spec['expectedSizeBytes']}, got {actual_size}"
        )

    raw_ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    train_args = raw_ckpt.get("train_args", {}) or {}
    ckpt_version = str(raw_ckpt.get("version", "unknown"))
    ckpt_date = str(raw_ckpt.get("date", "unknown"))

    yolo = YOLO(str(pt_path))
    model = yolo.model
    task = str(yolo.task)
    names = {str(k): str(v) for k, v in model.names.items()}
    stride = [int(x) for x in model.stride.tolist()]
    imgsz = int(train_args.get("imgsz", spec["imgsz"]))

    if task != spec["task"]:
        raise ValueError(f"Task mismatch for {pt_path}: expected {spec['task']}, got {task}")
    if imgsz != spec["imgsz"]:
        raise ValueError(f"Train imgsz mismatch for {pt_path}: expected {spec['imgsz']}, got {imgsz}")
    if names != spec["expectedNames"]:
        raise ValueError(f"Class names mismatch for {pt_path}: expected {spec['expectedNames']}, got {names}")

    return {
        "sourceFile": f"models/{pt_path.name}",
        "sourceSha256": actual_sha,
        "sourceSizeBytes": actual_size,
        "ckptUltralyticsVersion": ckpt_version,
        "ckptDate": ckpt_date,
        "task": task,
        "imgsz": imgsz,
        "trainImgsz": imgsz,
        "stride": stride,
        "nc": len(names),
        "names": names,
        "modelClass": model.__class__.__name__,
    }


def export_single_model(
    role: str,
    pt_path: Path,
    out_dir: Path,
    tmp_dir: Path,
    spec: dict[str, Any],
    log_buffer: io.StringIO,
) -> dict[str, Any]:
    ckpt_meta = inspect_checkpoint(pt_path, spec)
    staged_pt = tmp_dir / pt_path.name
    shutil.copy2(pt_path, staged_pt)

    imgsz = int(spec["imgsz"])
    export_kwargs = {
        "format": "onnx",
        "imgsz": imgsz,
        "batch": 1,
        "dynamic": False,
        "simplify": False,
        "opset": 17,
        "nms": False,
        "half": False,
        "int8": False,
        "device": "cpu",
    }

    log_buffer.write(f"\n=== Exporting [{role}] from {pt_path.name} ===\n")
    handler = logging.StreamHandler(log_buffer)
    handler.setLevel(logging.INFO)
    ULTRALYTICS_LOGGER.addHandler(handler)
    try:
        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
            exported_path_str = YOLO(str(staged_pt)).export(**export_kwargs)
    finally:
        ULTRALYTICS_LOGGER.removeHandler(handler)

    exported_path = Path(str(exported_path_str))
    if not exported_path.exists():
        raise RuntimeError(f"ONNX export failed to produce file for {role}: {exported_path}")

    target_onnx = out_dir / spec["onnxFilename"]
    shutil.move(str(exported_path), str(target_onnx))

    # Validate ONNX graph structure, normalize dynamic export timestamp to checkpoint date for deterministic SHA-256
    onnx_model = onnx.load(str(target_onnx))
    for prop in onnx_model.metadata_props:
        if prop.key == "date":
            prop.value = str(ckpt_meta["ckptDate"])
    onnx.save(onnx_model, str(target_onnx))
    onnx.checker.check_model(onnx_model)
    opset_imports = {imp.domain or "ai.onnx": int(imp.version) for imp in onnx_model.opset_import}
    onnx_meta = {prop.key: prop.value for prop in onnx_model.metadata_props}

    # Validate ONNX Runtime session and shapes
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(target_onnx),
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"Expected 1 input and 1 output for {role}, got {len(inputs)} and {len(outputs)}")

    input_name = inputs[0].name
    input_shape = [int(d) for d in inputs[0].shape]
    output_name = outputs[0].name
    output_shape = [int(d) for d in outputs[0].shape]

    if input_shape != spec["expectedInputShape"]:
        raise ValueError(
            f"ONNX input shape mismatch for {role}: "
            f"expected {spec['expectedInputShape']}, got {input_shape}"
        )
    if output_shape != spec["expectedOutputShape"]:
        raise ValueError(
            f"ONNX output shape mismatch for {role}: "
            f"expected {spec['expectedOutputShape']}, got {output_shape}"
        )

    # Numeric parity between PyTorch raw head and ONNX Runtime CPU
    pt_model = YOLO(str(pt_path)).model.eval().cpu()
    rng = np.random.default_rng(261016 + imgsz + len(role))
    sample_tensors = [
        rng.random(tuple(input_shape), dtype=np.float32),
        rng.random(tuple(input_shape), dtype=np.float32),
        rng.random(tuple(input_shape), dtype=np.float32),
        np.full(tuple(input_shape), 114.0 / 255.0, dtype=np.float32),
    ]

    max_abs_diff = 0.0
    max_rel_diff = 0.0
    mean_abs_diffs: list[float] = []

    for sample_idx, sample in enumerate(sample_tensors):
        with torch.no_grad():
            pt_raw = pt_model(torch.from_numpy(sample))[0].detach().cpu().numpy()
        ort_raw = session.run([output_name], {input_name: sample})[0]

        if pt_raw.shape != ort_raw.shape:
            raise ValueError(
                f"Runtime output shape mismatch on sample {sample_idx} for {role}: "
                f"{pt_raw.shape} vs {ort_raw.shape}"
            )
        if not np.isfinite(pt_raw).all() or not np.isfinite(ort_raw).all():
            raise ValueError(f"Non-finite values detected in output tensor for {role}")

        abs_diff = np.abs(pt_raw - ort_raw)
        sample_max_abs = float(np.max(abs_diff))
        sample_mean_abs = float(np.mean(abs_diff))
        denom = np.maximum(np.abs(pt_raw), 1e-3)
        sample_max_rel = float(np.max(abs_diff / denom))

        max_abs_diff = max(max_abs_diff, sample_max_abs)
        max_rel_diff = max(max_rel_diff, sample_max_rel)
        mean_abs_diffs.append(sample_mean_abs)

        if not np.allclose(pt_raw, ort_raw, atol=RAW_TENSOR_ATOL, rtol=RAW_TENSOR_RTOL):
            raise ValueError(
                f"PyTorch/ONNX raw tensor parity failed for {role} on sample {sample_idx}: "
                f"max_abs={sample_max_abs:.6e}, max_rel={sample_max_rel:.6e}"
            )

    # Verify source .pt was not mutated
    if sha256_file(pt_path) != ckpt_meta["sourceSha256"]:
        raise RuntimeError(f"Source checkpoint {pt_path} was modified during export!")

    entry: dict[str, Any] = {
        "file": target_onnx.name,
        "sha256": sha256_file(target_onnx),
        "sizeBytes": target_onnx.stat().st_size,
        **ckpt_meta,
        "exportUltralyticsVersion": ultralytics.__version__,
        "opset": opset_imports.get("ai.onnx", 17),
        "opsetImports": opset_imports,
        "exportConfig": export_kwargs,
        "inputName": input_name,
        "inputShape": input_shape,
        "inputDtype": "float32",
        "outputName": output_name,
        "outputShape": output_shape,
        "outputDtype": "float32",
        "outputLayout": spec["outputLayout"],
        "confThres": spec["confThres"],
        "iouThres": spec["iouThres"],
        "maxDet": spec["maxDet"],
        "onnxMetadataProps": onnx_meta,
        "exportParity": {
            "samples": len(sample_tensors),
            "maxAbsDelta": max_abs_diff,
            "meanAbsDelta": float(np.mean(mean_abs_diffs)),
            "maxRelDelta": max_rel_diff,
            "atol": RAW_TENSOR_ATOL,
            "rtol": RAW_TENSOR_RTOL,
            "passed": True,
        },
    }

    for extra_key in (
        "nmsMethod",
        "agnosticNms",
        "mappingRule",
        "classToDigit",
        "sortOrder",
        "triggerRule",
        "asymmetricLabels",
        "symmetricLabels",
        "minTotalBoxesForSymmetricTrigger",
        "actionOnTrigger",
    ):
        if extra_key in spec:
            entry[extra_key] = spec[extra_key]

    return entry


def export_all(models_dir: Path, out_dir: Path) -> dict[str, Any]:
    torch.set_num_threads(2)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "_export_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env_lock = collect_environment_lock(out_dir)
    log_buffer = io.StringIO()

    exported_models: dict[str, Any] = {}
    try:
        for role in ("frame", "digits", "reverse"):
            spec = EXPECTED_CHECKPOINTS[role]
            pt_path = models_dir / spec["sourceFilename"]
            exported_models[role] = export_single_model(
                role=role,
                pt_path=pt_path,
                out_dir=out_dir,
                tmp_dir=tmp_dir,
                spec=spec,
                log_buffer=log_buffer,
            )
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    stdout_log_path = out_dir / "export-stdout.log"
    stdout_log_path.write_text(log_buffer.getvalue(), encoding="utf-8")

    out_of_scope_path = models_dir / OUT_OF_SCOPE_CHECKPOINT["filename"]
    out_of_scope_info = None
    if out_of_scope_path.exists():
        out_of_scope_info = {
            **OUT_OF_SCOPE_CHECKPOINT,
            "actualSha256": sha256_file(out_of_scope_path),
            "sizeBytes": out_of_scope_path.stat().st_size,
        }

    manifest: dict[str, Any] = {
        "schemaVersion": 1,
        "pipelineId": PIPELINE_ID,
        "semanticsVersion": SEMANTICS_VERSION,
        "contractStatus": "approved",
        "created": "2026-10-16",
        "runtime": "ONNX Runtime CPU: YOLOv8n-OBB (frame) + YOLOv8n (reverse) + YOLOv8n (digits)",
        "preprocessing": {
            "version": PREPROCESS_VERSION,
            "colorOrder": "RGB",
            "scale": 1.0 / 255.0,
            "padValue": 114,
            "interpolation": "INTER_LINEAR",
            "obbWarpInterpolation": "INTER_LINEAR",
            "obbWarpBorderMode": "BORDER_REPLICATE",
        },
        "models": exported_models,
        "policy": {
            "multipleFramesPolicy": "uncertain_multiple_rows",
            "noFramePolicy": "uncertain_no_row",
            "obbCropPolicy": "perspective_warp_long_edge_horizontal_no_margin",
            "geometryBounds": {
                "minWidth": 16,
                "minHeight": 8,
                "minAspect": 1.2,
                "maxAspect": 25.0,
            },
            "acceptedLengths": [5, 6],
            "preserveLeadingZeros": True,
        },
        "tolerances": {
            "rawTensorAtol": RAW_TENSOR_ATOL,
            "rawTensorRtol": RAW_TENSOR_RTOL,
            "rawCoordPxAtol": 0.01,
            "rawScoreAtol": 0.0005,
            "hostScoreAtol": 0.0005,
            "hostBoxPxAtol": 0.05,
            "androidBoxPxAtol": 0.5,
            "rationale": (
                "YOLOv8 DFL box channels (cx,cy,w,h) multiply 16-bin softmax expectations by stride (8..32), "
                "giving <=0.01px FP32 Conv+BN fusion delta in network space; sigmoid score/angle channels "
                "agree within 5e-4; post-NMS class labels, anchor indices, orientation, status, and final "
                "digit strings must match 100% with zero tolerance."
            ),
        },
    }

    manifest_path = out_dir / "supplied-yolo-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    receipt: dict[str, Any] = {
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "manifestFile": manifest_path.name,
        "manifestSha256": sha256_file(manifest_path),
        "stdoutLogFile": stdout_log_path.name,
        "stdoutLogSha256": sha256_file(stdout_log_path),
        "environment": env_lock,
        "models": {
            role: {
                "file": info["file"],
                "sha256": info["sha256"],
                "sizeBytes": info["sizeBytes"],
                "sourceFile": info["sourceFile"],
                "sourceSha256": info["sourceSha256"],
                "inputShape": info["inputShape"],
                "outputShape": info["outputShape"],
                "opset": info["opset"],
                "exportParity": info["exportParity"],
            }
            for role, info in exported_models.items()
        },
        "outOfScopeCheckpoint": out_of_scope_info,
    }
    receipt_path = out_dir / "export-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def stage_bundle_to_android_assets(staging_dir: Path, assets_dir: Path) -> dict[str, Any]:
    """Copy verified ONNX models and manifest from staging into Android assets and verify SHA-256."""
    manifest_src = staging_dir / "supplied-yolo-manifest.json"
    if not manifest_src.exists():
        raise FileNotFoundError(f"Staging manifest not found: {manifest_src}")
    manifest = json.loads(manifest_src.read_text(encoding="utf-8"))
    if manifest.get("pipelineId") != PIPELINE_ID or manifest.get("contractStatus") != "approved":
        raise ValueError(f"Staging manifest is not approved for {PIPELINE_ID}")

    assets_dir.mkdir(parents=True, exist_ok=True)
    legacy_assets = (
        "row-detector.onnx",
        "row-recognizer.onnx",
        "row-characters.json",
        "row-pipeline-manifest.json",
        "diagnostic-fp32.tflite",
        "diagnostic-fp32-manifest.json",
    )
    removed_legacy: list[str] = []
    for legacy_name in legacy_assets:
        legacy_path = assets_dir / legacy_name
        if legacy_path.exists():
            legacy_path.unlink()
            removed_legacy.append(legacy_name)

    staged_files: dict[str, Any] = {}
    for role in ("frame", "digits", "reverse"):
        spec = manifest["models"][role]
        filename = spec["file"]
        expected_sha = spec["sha256"]
        expected_size = int(spec["sizeBytes"])
        src_onnx = staging_dir / filename
        if not src_onnx.exists():
            raise FileNotFoundError(f"Missing staged ONNX model for {role}: {src_onnx}")
        if src_onnx.stat().st_size != expected_size or sha256_file(src_onnx) != expected_sha:
            raise ValueError(f"Staged ONNX checksum/size mismatch for {role}: {src_onnx}")
        dst_onnx = assets_dir / filename
        shutil.copy2(src_onnx, dst_onnx)
        if dst_onnx.stat().st_size != expected_size or sha256_file(dst_onnx) != expected_sha:
            raise ValueError(f"Copied Android asset checksum/size mismatch for {role}: {dst_onnx}")
        staged_files[role] = {
            "file": filename,
            "sha256": expected_sha,
            "sizeBytes": expected_size,
        }

    manifest_dst = assets_dir / "supplied-yolo-manifest.json"
    shutil.copy2(manifest_src, manifest_dst)
    manifest_sha = sha256_file(manifest_src)
    if sha256_file(manifest_dst) != manifest_sha:
        raise ValueError(f"Copied Android manifest checksum mismatch: {manifest_dst}")

    return {
        "assetsDir": str(assets_dir),
        "manifestFile": manifest_dst.name,
        "manifestSha256": manifest_sha,
        "models": staged_files,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        type=Path,
        default=Path("models"),
        help="Directory containing best_latest.pt, best_0608.pt, best_1608.pt",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/supplied-yolo"),
        help="Staging output directory for ONNX models, manifest, and export receipt",
    )
    parser.add_argument(
        "--stage-android-assets",
        type=Path,
        default=None,
        help="Optional Android assets/ocr directory to copy and verify staged ONNX bundle",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="When combined with --stage-android-assets, copy existing verified staging bundle without re-exporting",
    )
    args = parser.parse_args(argv)
    if args.stage_only:
        if args.stage_android_assets is None:
            raise ValueError("--stage-only requires --stage-android-assets")
        staged = stage_bundle_to_android_assets(args.out, args.stage_android_assets)
        print(json.dumps(staged, indent=2))
        return 0

    receipt = export_all(args.models, args.out)
    if args.stage_android_assets is not None:
        receipt["stagedAndroidAssets"] = stage_bundle_to_android_assets(args.out, args.stage_android_assets)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
