"""Unit and contract parity tests for the 3-stage supplied YOLOv8 water-meter OCR pipeline.

CLI usage:
    .cache/yolo-env/Scripts/python.exe -m unittest tools.ocr.tests.test_supplied_yolo_contract -v
"""
from __future__ import annotations

import hashlib
import json
import math
import unittest
import zipfile
from pathlib import Path

import numpy as np

from tools.ocr.export_supplied_yolo import EXPECTED_CHECKPOINTS, OUT_OF_SCOPE_CHECKPOINT, sha256_file
from tools.ocr.supplied_yolo_reference import (
    DEFAULT_CLASS_TO_DIGIT,
    crop_and_straighten_frame,
    decode_axis_aligned_boxes,
    decode_digits_reading,
    decode_frame_obb,
    evaluate_reverse_trigger,
    load_rgb_oriented,
    map_digit_label,
    obb_to_corners,
    order_obb_corners,
    probiou,
    sha256_bytes,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGING_DIR = REPO_ROOT / "artifacts" / "supplied-yolo"
FIXTURES_MANIFEST_PATH = STAGING_DIR / "fixtures" / "manifest.json"
HOST_REPORT_DIR = STAGING_DIR / "host-report"


class SuppliedYoloContractTests(unittest.TestCase):
    def test_manifest_and_exported_onnx_integrity(self) -> None:
        manifest_path = STAGING_DIR / "supplied-yolo-manifest.json"
        receipt_path = STAGING_DIR / "export-receipt.json"
        lock_path = STAGING_DIR / "python-env-lock.json"
        self.assertTrue(manifest_path.exists(), f"Missing {manifest_path}")
        self.assertTrue(receipt_path.exists(), f"Missing {receipt_path}")
        self.assertTrue(lock_path.exists(), f"Missing {lock_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["semanticsVersion"], "supplied-yolo-3stage-v1")
        self.assertEqual(manifest["pipelineId"], "supplied-yolo-3stage-v1")
        self.assertEqual(manifest["contractStatus"], "approved")

        # Verify all 3 source .pt checkpoints and exported .onnx files
        for role, spec in EXPECTED_CHECKPOINTS.items():
            pt_path = REPO_ROOT / "models" / spec["sourceFilename"]
            self.assertTrue(pt_path.exists(), f"Missing source checkpoint {pt_path}")
            self.assertEqual(sha256_file(pt_path), spec["expectedSha256"])

            model_entry = manifest["models"][role]
            onnx_path = STAGING_DIR / model_entry["file"]
            self.assertTrue(onnx_path.exists(), f"Missing exported ONNX {onnx_path}")
            self.assertEqual(sha256_file(onnx_path), model_entry["sha256"])
            self.assertEqual(model_entry["inputShape"], spec["expectedInputShape"])
            self.assertEqual(model_entry["outputShape"], spec["expectedOutputShape"])
            self.assertEqual(model_entry["task"], spec["task"])
            self.assertEqual(model_entry["imgsz"], spec["imgsz"])
            self.assertEqual(model_entry["names"], spec["expectedNames"])
            self.assertEqual(model_entry["onnxMetadataProps"].get("date"), model_entry["ckptDate"])
            self.assertTrue(model_entry["exportParity"]["passed"])

        # Verify out-of-scope best_2107.pt is untouched
        oos_path = REPO_ROOT / "models" / OUT_OF_SCOPE_CHECKPOINT["filename"]
        if oos_path.exists():
            self.assertEqual(sha256_file(oos_path), OUT_OF_SCOPE_CHECKPOINT["expectedSha256"])

        # Verify staged Android assets (when present) match staging manifest and ONNX checksums
        android_ocr_dir = REPO_ROOT / "app" / "android" / "app" / "src" / "main" / "assets" / "ocr"
        android_manifest = android_ocr_dir / "supplied-yolo-manifest.json"
        if android_manifest.exists():
            self.assertEqual(sha256_file(android_manifest), sha256_file(manifest_path))
            actual_files = {p.name for p in android_ocr_dir.iterdir() if p.is_file()}
            expected_files = {
                "frame.onnx",
                "digits.onnx",
                "reverse.onnx",
                "supplied-yolo-manifest.json",
                "THIRD_PARTY_LICENSES.txt",
            }
            self.assertEqual(actual_files, expected_files)
            for role in ("frame", "digits", "reverse"):
                model_entry = manifest["models"][role]
                asset_onnx = android_ocr_dir / model_entry["file"]
                self.assertTrue(asset_onnx.exists(), f"Missing staged Android ONNX {asset_onnx}")
                self.assertEqual(asset_onnx.stat().st_size, model_entry["sizeBytes"])
                self.assertEqual(sha256_file(asset_onnx), model_entry["sha256"])

    def test_30_class_mapping_and_invalid_class_guards(self) -> None:
        for cls_idx in range(30):
            label = str(cls_idx)
            expected_digit = str(cls_idx % 10)
            self.assertEqual(map_digit_label(label), expected_digit)

        # Transitional wheel class 13 -> '3'
        self.assertEqual(map_digit_label("13"), "3")
        self.assertEqual(map_digit_label("20"), "0")
        self.assertEqual(map_digit_label("29"), "9")

        for invalid in ("30", "-1", "1n", "khung", "", "3.0", "abc"):
            with self.assertRaises(ValueError):
                map_digit_label(invalid)

        tampered = {**DEFAULT_CLASS_TO_DIGIT, "13": "4"}
        with self.assertRaises(ValueError):
            map_digit_label("13", class_to_digit=tampered)

    def test_reverse_check_trigger_policy(self) -> None:
        self.assertEqual(evaluate_reverse_trigger([]), (False, "no_reverse_boxes"))
        self.assertEqual(
            evaluate_reverse_trigger([{"class": "1n", "score": 0.83}]),
            (False, "single_symmetric_1n_ignored"),
        )
        self.assertEqual(
            evaluate_reverse_trigger([{"class": "1n", "score": 0.8}, {"class": "1n", "score": 0.7}]),
            (True, "multiple_reverse_boxes"),
        )
        for asym in ("2n", "3n", "4n", "5n", "7n"):
            self.assertEqual(
                evaluate_reverse_trigger([{"class": asym, "score": 0.45}]),
                (True, "asymmetric_reverse_label"),
            )

        for unsupported in ("0n", "6n", "8n", "9n", "1", "khung"):
            with self.assertRaises(ValueError):
                evaluate_reverse_trigger([{"class": unsupported, "score": 0.9}])

    def test_obb_corner_ordering_and_crop_policies(self) -> None:
        dummy_rgb = np.full((480, 640, 3), 128, dtype=np.uint8)

        # 0 frames -> uncertain / no_row
        res_empty = crop_and_straighten_frame(dummy_rgb, [])
        self.assertIsNone(res_empty["cropRgb"])
        self.assertEqual(res_empty["reason"], "no_row")
        self.assertIsNone(res_empty["rowBox"])

        # >= 2 frames -> uncertain / multiple_rows
        two_frames = [
            {"polygon": [[100.0, 100.0], [220.0, 100.0], [220.0, 130.0], [100.0, 130.0]]},
            {"polygon": [[300.0, 200.0], [420.0, 200.0], [420.0, 230.0], [300.0, 230.0]]},
        ]
        res_multi = crop_and_straighten_frame(dummy_rgb, two_frames)
        self.assertIsNone(res_multi["cropRgb"])
        self.assertEqual(res_multi["reason"], "multiple_rows")
        self.assertIsNone(res_multi["rowBox"])

        # Degenerate small geometry -> unusable_row_geometry
        tiny_frame = [{"polygon": [[50.0, 50.0], [60.0, 50.0], [60.0, 55.0], [50.0, 55.0]]}]
        res_tiny = crop_and_straighten_frame(dummy_rgb, tiny_frame)
        self.assertIsNone(res_tiny["cropRgb"])
        self.assertEqual(res_tiny["reason"], "unusable_row_geometry")

        # Test corner ordering under cyclic shifts, mirror flips, and tilt angles
        corners = obb_to_corners(300.0, 200.0, 140.0, 32.0, math.radians(-22.0))
        for shift in range(4):
            shifted = np.roll(corners, shift=shift, axis=0)
            for pts in (shifted, shifted[::-1]):
                ordered = order_obb_corners(pts)
                tl, tr, br, bl = ordered
                w_len = float(np.linalg.norm(tr - tl))
                h_len = float(np.linalg.norm(bl - tl))
                cross = float((tr[0] - tl[0]) * (bl[1] - tl[1]) - (tr[1] - tl[1]) * (bl[0] - tl[0]))
                self.assertGreaterEqual(w_len, h_len)
                self.assertGreater(cross, 0.0)
                self.assertGreater(float(tr[0] - tl[0]), 0.0)

        # Identical OBB has ProbIoU near 1.0, disjoint OBB has ProbIoU near 0.0
        self.assertGreater(probiou((100, 100, 120, 30, 0.2), (100, 100, 120, 30, 0.2)), 0.99)
        self.assertLess(probiou((100, 100, 50, 20, 0.0), (400, 400, 50, 20, 0.0)), 0.05)

    def test_digits_sorting_leading_zeros_and_length_policy(self) -> None:
        # Unordered detections with leading '0' and transitional class '13' -> '085913'
        unordered = [
            {"anchor": 15, "classId": 13, "class": "13", "score": 0.80, "box": [105.0, 2.0, 120.0, 25.0], "cx": 112.5},
            {"anchor": 2, "classId": 0, "class": "0", "score": 0.95, "box": [3.0, 2.0, 18.0, 25.0], "cx": 10.5},
            {"anchor": 9, "classId": 9, "class": "9", "score": 0.90, "box": [63.0, 2.0, 78.0, 25.0], "cx": 70.5},
            {"anchor": 5, "classId": 8, "class": "8", "score": 0.92, "box": [23.0, 2.0, 38.0, 25.0], "cx": 30.5},
            {"anchor": 12, "classId": 1, "class": "1", "score": 0.88, "box": [83.0, 2.0, 98.0, 25.0], "cx": 90.5},
            {"anchor": 7, "classId": 5, "class": "5", "score": 0.91, "box": [43.0, 2.0, 58.0, 25.0], "cx": 50.5},
        ]
        decoded = decode_digits_reading(unordered)
        self.assertEqual(decoded["status"], "ok")
        self.assertEqual(decoded["rawDigits"], "085913")
        self.assertIsNone(decoded["reason"])
        self.assertTrue(decoded["rawDigits"].startswith("0"))
        self.assertEqual(len(decoded["rawDigits"]), 6)

        # 5 digits is also accepted
        decoded_5 = decode_digits_reading(unordered[1:])
        self.assertEqual(decoded_5["status"], "ok")
        self.assertEqual(decoded_5["rawDigits"], "08591")

        # Empty -> no_characters
        decoded_empty = decode_digits_reading([])
        self.assertEqual(decoded_empty["status"], "uncertain")
        self.assertIsNone(decoded_empty["rawDigits"])
        self.assertEqual(decoded_empty["reason"], "no_characters")

        # 4 digits -> unexpected_length
        decoded_4 = decode_digits_reading(unordered[:4])
        self.assertEqual(decoded_4["status"], "uncertain")
        self.assertIsNone(decoded_4["rawDigits"])
        self.assertEqual(decoded_4["reason"], "unexpected_length")
        self.assertEqual(len(decoded_4["rawDiagnostic"]), 4)

    def test_class_agnostic_nms_suppresses_overlapping_transitional_wheel_classes(self) -> None:
        raw = np.zeros((1, 34, 2100), dtype=np.float32)
        # Anchor 10: class "3" with score 0.72 at (100, 50, 20, 30)
        raw[0, 0:4, 10] = [100.0, 50.0, 20.0, 30.0]
        raw[0, 4 + 3, 10] = 0.72
        # Anchor 11: class "13" with score 0.86 at almost identical position (101, 50, 20, 30)
        raw[0, 0:4, 11] = [101.0, 50.0, 20.0, 30.0]
        raw[0, 4 + 13, 11] = 0.86

        meta = {"gain": 1.0, "left": 0, "top": 0, "inputWidth": 320, "inputHeight": 320}
        names = {str(i): str(i) for i in range(30)}

        dets = decode_axis_aligned_boxes(raw, meta, names=names, conf_thres=0.25, iou_thres=0.50, agnostic_nms=True)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["class"], "13")
        self.assertEqual(dets[0]["anchor"], 11)

        # Also verify non-finite tensor guard
        raw_nan = raw.copy()
        raw_nan[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            decode_axis_aligned_boxes(raw_nan, meta, names=names)

        raw_obb_nan = np.full((1, 6, 8400), np.nan, dtype=np.float32)
        with self.assertRaises(ValueError):
            decode_frame_obb(raw_obb_nan, meta)

    def test_corrupt_and_missing_image_guards(self) -> None:
        corrupt_path = STAGING_DIR / "fixtures" / "images" / "corrupt_unreadable.bin"
        with self.assertRaises(ValueError) as ctx:
            load_rgb_oriented(corrupt_path)
        self.assertIn("E_INFERENCE", str(ctx.exception))

        missing_path = STAGING_DIR / "fixtures" / "images" / "does_not_exist.png"
        with self.assertRaises(ValueError) as ctx_missing:
            load_rgb_oriented(missing_path)
        self.assertIn("E_IMAGE_URI", str(ctx_missing.exception))

    def test_host_reference_pytorch_vs_onnx_parity_report(self) -> None:
        report_path = HOST_REPORT_DIR / "parity-report.json"
        self.assertTrue(report_path.exists(), f"Missing parity report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        fixtures_manifest = json.loads(FIXTURES_MANIFEST_PATH.read_text(encoding="utf-8"))

        self.assertTrue(report["allPass"], "Host parity suite did not pass all checks")
        self.assertEqual(report["fixtureCount"], len(fixtures_manifest["fixtures"]))
        self.assertTrue(report["exifOrientationParity"]["allMatchCanonical"])
        self.assertEqual(report["exifOrientationParity"]["count"], 8)

        by_id = {f["id"]: f for f in report["fixtures"]}
        for expected_fx in fixtures_manifest["fixtures"]:
            fx_id = expected_fx["id"]
            self.assertIn(fx_id, by_id)
            actual_fx = by_id[fx_id]
            self.assertTrue(actual_fx["passed"], f"Fixture {fx_id} failed parity")

            detail_path = HOST_REPORT_DIR / f"{fx_id}.result.json"
            self.assertTrue(detail_path.exists(), f"Missing detail report for {fx_id}")
            detail = json.loads(detail_path.read_text(encoding="utf-8"))

            if expected_fx["expected"].get("promiseRejected"):
                self.assertTrue(detail["promiseRejected"])
                self.assertIn("E_INFERENCE", detail["onnxError"])
                self.assertIn("E_INFERENCE", detail["pytorchError"])
            else:
                rgb, _ = load_rgb_oriented(REPO_ROOT / expected_fx["path"])
                expected_pixel_sha = sha256_bytes(rgb.tobytes())
                self.assertEqual(detail["onnxResult"]["imagePixelSha256"], expected_pixel_sha)
                self.assertEqual(detail["pytorchResult"]["imagePixelSha256"], expected_pixel_sha)
                self.assertEqual(detail["onnxResult"]["status"], expected_fx["expected"]["status"])
                self.assertEqual(detail["onnxResult"]["rawDigits"], expected_fx["expected"]["rawDigits"])
                self.assertEqual(detail["onnxResult"]["reason"], expected_fx["expected"]["reason"])
                self.assertEqual(detail["onnxResult"]["status"], detail["pytorchResult"]["status"])
                self.assertEqual(detail["onnxResult"]["rawDigits"], detail["pytorchResult"]["rawDigits"])
                self.assertEqual(detail["onnxResult"]["isReversed"], detail["pytorchResult"]["isReversed"])
                self.assertEqual(detail["onnxResult"]["reason"], detail["pytorchResult"]["reason"])

    def test_android_gate_assets_staging(self) -> None:
        gates_dir = REPO_ROOT / ".cache" / "android-gates-assets"
        gates_manifest_path = gates_dir / "gates-manifest.json"
        if not gates_manifest_path.exists():
            return
        gates_manifest = json.loads(gates_manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(gates_manifest["pipelineId"], "supplied-yolo-3stage-v1")
        self.assertEqual(gates_manifest["semanticsVersion"], "supplied-yolo-3stage-v1")
        files_by_name = {f["name"]: f for f in gates_manifest["files"]}
        for required in (
            "parity-report.json",
            "fixtures-manifest.json",
            "sample.jfif",
            "lifecycle.png",
            "corrupt.img",
            "oversize.png",
            "oversize-dimension.png",
            "meter-upright-01.png",
            "meter-upright-01.jfif",
            "meter-upright-01.result.json",
            "meter-upright-01.frame_input.f32",
            "meter-upright-01.frame_output.f32",
            "meter-upright-01.reverse_input.f32",
            "meter-upright-01.reverse_output.f32",
            "meter-upright-01.digits_input.f32",
            "meter-upright-01.digits_output.f32",
        ):
            self.assertIn(required, files_by_name, f"Missing required gate asset: {required}")
            path = gates_dir / required
            self.assertTrue(path.exists())
            self.assertEqual(path.stat().st_size, files_by_name[required]["sizeBytes"])
            self.assertEqual(sha256_file(path), files_by_name[required]["sha256"])
        for orientation in range(1, 9):
            self.assertIn(f"exif-{orientation}.png", files_by_name)
            self.assertIn(f"exif-{orientation}.frame_input.f32", files_by_name)

    def test_built_apk_zip_audit(self) -> None:
        manifest = json.loads((STAGING_DIR / "supplied-yolo-manifest.json").read_text(encoding="utf-8"))
        manifest_sha = sha256_file(STAGING_DIR / "supplied-yolo-manifest.json")
        apk_root = REPO_ROOT / "app" / "android" / "app" / "build" / "outputs" / "apk"
        release_apk = apk_root / "release" / "app-release.apk"
        debug_apk = apk_root / "debug" / "app-debug.apk"

        forbidden_substrings = (
            "row-detector.onnx",
            "row-recognizer.onnx",
            "row-characters.json",
            "row-pipeline-manifest.json",
            "diagnostic-fp32.tflite",
            "diagnostic-fp32-manifest.json",
            ".f32",
        )
        expected_ocr_entries = {
            "assets/ocr/frame.onnx",
            "assets/ocr/digits.onnx",
            "assets/ocr/reverse.onnx",
            "assets/ocr/supplied-yolo-manifest.json",
            "assets/ocr/THIRD_PARTY_LICENSES.txt",
        }

        for variant, apk_path in (("release", release_apk), ("debug", debug_apk)):
            if not apk_path.exists():
                continue
            with zipfile.ZipFile(apk_path, "r") as zf:
                names = zf.namelist()
                ocr_entries = {n for n in names if n.startswith("assets/ocr/") and not n.endswith("/")}
                self.assertEqual(ocr_entries, expected_ocr_entries, f"Unexpected assets/ocr entries in {variant} APK")
                for entry in names:
                    for forbidden in forbidden_substrings:
                        self.assertNotIn(forbidden, entry, f"Forbidden entry {entry} in {variant} APK")

                apk_manifest_bytes = zf.read("assets/ocr/supplied-yolo-manifest.json")
                self.assertEqual(hashlib.sha256(apk_manifest_bytes).hexdigest(), manifest_sha)

                for role in ("frame", "digits", "reverse"):
                    spec = manifest["models"][role]
                    entry_name = f"assets/ocr/{spec['file']}"
                    data = zf.read(entry_name)
                    self.assertEqual(len(data), spec["sizeBytes"], f"{variant} APK {entry_name} size mismatch")
                    self.assertEqual(
                        hashlib.sha256(data).hexdigest(),
                        spec["sha256"],
                        f"{variant} APK {entry_name} SHA-256 mismatch",
                    )

                if variant == "release":
                    self.assertIn("assets/index.android.bundle", names, "Release APK must bundle JS offline")


if __name__ == "__main__":
    unittest.main()
