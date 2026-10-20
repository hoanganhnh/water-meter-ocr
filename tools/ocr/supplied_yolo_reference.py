"""Host reference pipeline for the 3-stage supplied YOLOv8 water-meter OCR contract.

Implements raw-tensor preprocessing, OBB decoding, ProbIoU NMS, perspective crop straightening,
reverse orientation check, and 30-class digit decoding in pure math + OpenCV without relying
on Ultralytics post-processing wrappers. Runs both PyTorch and ONNX Runtime backends to isolate
export vs decoder behavior and produce fixture parity reports.

CLI usage:
    .cache/yolo-env/Scripts/python.exe -m tools.ocr.supplied_yolo_reference \
        --bundle artifacts/supplied-yolo \
        --fixtures artifacts/supplied-yolo/fixtures/manifest.json \
        --out artifacts/supplied-yolo/host-report
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

MAX_SOURCE_BYTES = 25 * 1024 * 1024
MAX_SOURCE_PIXELS = 12_000_000
MAX_SOURCE_DIMENSION = 16_384

DEFAULT_GEOMETRY_BOUNDS = {
    "minWidth": 16,
    "minHeight": 8,
    "minAspect": 1.2,
    "maxAspect": 25.0,
}

SUPPORTED_REVERSE_ASYMMETRIC = {"2n", "3n", "4n", "5n", "7n"}
SUPPORTED_REVERSE_SYMMETRIC = {"1n"}
SUPPORTED_REVERSE_ALL = SUPPORTED_REVERSE_ASYMMETRIC | SUPPORTED_REVERSE_SYMMETRIC
DEFAULT_CLASS_TO_DIGIT = {str(i): str(i % 10) for i in range(30)}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def orient_rgb(rgb: np.ndarray, orientation: int) -> np.ndarray:
    """Apply EXIF orientation (1..8) matching Android MeterOcrModule.orient."""
    ops = {
        2: lambda img: np.fliplr(img),
        3: lambda img: np.rot90(img, 2),
        4: lambda img: np.flipud(img),
        5: lambda img: np.fliplr(np.rot90(img, 3)),
        6: lambda img: np.rot90(img, 3),
        7: lambda img: np.fliplr(np.rot90(img, 1)),
        8: lambda img: np.rot90(img, 1),
    }
    op = ops.get(orientation)
    return np.ascontiguousarray(op(rgb) if op is not None else rgb)


def read_exif_orientation(raw_bytes: bytes) -> int:
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            exif = img.getexif()
            if exif:
                return int(exif.get(274, 1))
    except Exception:
        pass
    return 1


def load_rgb_oriented(path: Path) -> tuple[np.ndarray, int]:
    """Load an image file into an EXIF-oriented RGB uint8 array with Android contract guards."""
    if not path.exists():
        raise ValueError(f"E_IMAGE_URI: Image file does not exist: {path}")
    size_bytes = path.stat().st_size
    if size_bytes > MAX_SOURCE_BYTES:
        raise ValueError(f"E_INFERENCE: Cropped image exceeds the 25 MB file limit ({size_bytes} bytes)")
    raw_bytes = path.read_bytes()
    if not raw_bytes:
        raise ValueError("E_INFERENCE: Image file is corrupt or unsupported (empty file)")

    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if bgr is None or bgr.ndim != 3 or bgr.shape[0] <= 0 or bgr.shape[1] <= 0:
        raise ValueError("E_INFERENCE: Image file is corrupt or unsupported")

    height, width = int(bgr.shape[0]), int(bgr.shape[1])
    if width > MAX_SOURCE_DIMENSION or height > MAX_SOURCE_DIMENSION:
        raise ValueError("E_INFERENCE: Image dimension exceeds the 16384 pixel limit")
    if width * height > MAX_SOURCE_PIXELS:
        raise ValueError("E_INFERENCE: Cropped image exceeds the 12 MP pixel limit")

    orientation = read_exif_orientation(raw_bytes)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    oriented = orient_rgb(rgb, orientation)
    return oriented, orientation


def letterbox_rgb(
    rgb: np.ndarray,
    imgsz: int,
    pad_value: int = 114,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resize and pad RGB uint8 image into NCHW float32 [1, 3, imgsz, imgsz] in [0, 1]."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HWC 3-channel RGB image, got shape {rgb.shape}")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: {width}x{height}")

    gain = min(float(imgsz) / float(width), float(imgsz) / float(height))
    rw = int(round(width * gain))
    rh = int(round(height * gain))
    pad_w = (imgsz - rw) / 2.0
    pad_h = (imgsz - rh) / 2.0
    left = int(round(pad_w - 0.1))
    top = int(round(pad_h - 0.1))
    right = imgsz - rw - left
    bottom = imgsz - rh - top

    if (width, height) != (rw, rh):
        resized = cv2.resize(rgb, (rw, rh), interpolation=cv2.INTER_LINEAR)
    else:
        resized = rgb

    canvas = np.full((imgsz, imgsz, 3), pad_value, dtype=np.uint8)
    canvas[top : top + rh, left : left + rw] = resized
    tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1))[None].astype(np.float32) / np.float32(255.0)

    meta = {
        "inputWidth": width,
        "inputHeight": height,
        "imgsz": imgsz,
        "gain": gain,
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "resizedWidth": rw,
        "resizedHeight": rh,
    }
    return tensor, meta


def probiou(
    obb1: tuple[float, float, float, float, float] | list[float] | np.ndarray,
    obb2: tuple[float, float, float, float, float] | list[float] | np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Compute probabilistic IoU between two oriented bounding boxes (cx, cy, w, h, angle_rad)."""
    x1, y1, w1, h1, r1 = (float(v) for v in obb1)
    x2, y2, w2, h2, r2 = (float(v) for v in obb2)

    aa1, bb1 = (w1 * w1) / 12.0, (h1 * h1) / 12.0
    c1, s1 = math.cos(r1), math.sin(r1)
    a1 = aa1 * c1 * c1 + bb1 * s1 * s1
    b1 = aa1 * s1 * s1 + bb1 * c1 * c1
    cov1 = (aa1 - bb1) * c1 * s1

    aa2, bb2 = (w2 * w2) / 12.0, (h2 * h2) / 12.0
    c2, s2 = math.cos(r2), math.sin(r2)
    a2 = aa2 * c2 * c2 + bb2 * s2 * s2
    b2 = aa2 * s2 * s2 + bb2 * c2 * c2
    cov2 = (aa2 - bb2) * c2 * s2

    denom = (a1 + a2) * (b1 + b2) - (cov1 + cov2) ** 2 + eps
    t1 = (((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / denom) * 0.25
    t2 = (((cov1 + cov2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5
    det1 = max(0.0, a1 * b1 - cov1 * cov1)
    det2 = max(0.0, a2 * b2 - cov2 * cov2)
    t3 = 0.5 * math.log(denom / (4.0 * math.sqrt(det1 * det2) + eps) + eps)

    bd = min(max(t1 + t2 + t3, eps), 100.0)
    hd = math.sqrt(1.0 - math.exp(-bd) + eps)
    return float(1.0 - hd)


def obb_to_corners(cx: float, cy: float, w: float, h: float, angle_rad: float) -> np.ndarray:
    """Convert OBB (cx, cy, w, h, angle_rad) to 4 corners (4, 2) matching Ultralytics xywhr2xyxyxyxy."""
    cos_a = math.cos(float(angle_rad))
    sin_a = math.sin(float(angle_rad))
    vx1 = (float(w) / 2.0) * cos_a
    vy1 = (float(w) / 2.0) * sin_a
    vx2 = -(float(h) / 2.0) * sin_a
    vy2 = (float(h) / 2.0) * cos_a
    return np.array(
        [
            [cx + vx1 + vx2, cy + vy1 + vy2],
            [cx + vx1 - vx2, cy + vy1 - vy2],
            [cx - vx1 - vx2, cy - vy1 - vy2],
            [cx - vx1 + vx2, cy - vy1 + vy2],
        ],
        dtype=np.float32,
    )


def decode_frame_obb(
    raw_output: np.ndarray,
    meta: dict[str, Any],
    names: dict[str, str] | None = None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.50,
    max_det: int = 300,
) -> list[dict[str, Any]]:
    """Decode [1, 6, 8400] raw OBB output, run ProbIoU NMS, and map polygons to image coordinates."""
    if raw_output.shape != (1, 6, 8400) or not np.isfinite(raw_output).all():
        raise ValueError(f"E_INFERENCE: Invalid frame OBB output tensor shape {raw_output.shape}")
    label_map = names or {"0": "khung"}
    if label_map.get("0") != "khung":
        raise ValueError(f"Unsupported frame class dictionary: {label_map}")

    preds = raw_output[0]
    scores = preds[4]
    candidate_indices = np.flatnonzero(scores >= conf_thres)

    candidates: list[tuple[int, float, tuple[float, float, float, float, float]]] = []
    for idx in candidate_indices:
        anchor = int(idx)
        cx, cy, w, h, score, angle = (float(preds[c, anchor]) for c in range(6))
        if w <= 0.0 or h <= 0.0:
            continue
        candidates.append((anchor, score, (cx, cy, w, h, angle)))

    candidates.sort(key=lambda item: (-item[1], item[0]))

    kept: list[tuple[int, float, tuple[float, float, float, float, float]]] = []
    for cand in candidates:
        if all(probiou(cand[2], prev[2]) < iou_thres for prev in kept):
            kept.append(cand)
            if len(kept) >= max_det:
                break

    gain = float(meta["gain"])
    left = float(meta["left"])
    top = float(meta["top"])

    results: list[dict[str, Any]] = []
    for anchor, score, obb in kept:
        cx, cy, w, h, angle = obb
        corners_net = obb_to_corners(cx, cy, w, h, angle)
        corners_img = np.empty_like(corners_net, dtype=np.float32)
        corners_img[:, 0] = (corners_net[:, 0] - left) / gain
        corners_img[:, 1] = (corners_net[:, 1] - top) / gain
        results.append(
            {
                "anchor": anchor,
                "classId": 0,
                "class": label_map["0"],
                "score": score,
                "netObb": [cx, cy, w, h, angle],
                "polygon": [[float(pt[0]), float(pt[1])] for pt in corners_img],
            }
        )
    return results


def order_obb_corners(pts: np.ndarray | list[list[float]]) -> np.ndarray:
    """Order 4 OBB vertices clockwise [tl, tr, br, bl] with long edge horizontal and non-mirrored chirality."""
    arr = np.asarray(pts, dtype=np.float32)
    if arr.shape != (4, 2) or not np.isfinite(arr).all():
        raise ValueError(f"Expected (4, 2) finite polygon vertices, got shape {arr.shape}")

    d01 = float(np.linalg.norm(arr[1] - arr[0]))
    d12 = float(np.linalg.norm(arr[2] - arr[1]))
    if d01 >= d12:
        c1 = arr[[0, 1, 2, 3]]
        c2 = arr[[2, 3, 0, 1]]
    else:
        c1 = arr[[1, 2, 3, 0]]
        c2 = arr[[3, 0, 1, 2]]

    def ensure_positive_chirality(c: np.ndarray) -> np.ndarray:
        tl, tr, br, bl = c
        cross = float((tr[0] - tl[0]) * (bl[1] - tl[1]) - (tr[1] - tl[1]) * (bl[0] - tl[0]))
        if cross < 0.0:
            return np.array([bl, br, tr, tl], dtype=np.float32)
        return c

    c1 = ensure_positive_chirality(c1)
    c2 = ensure_positive_chirality(c2)

    dx1 = float(c1[1, 0] - c1[0, 0])
    dy1 = float(c1[1, 1] - c1[0, 1])
    if dx1 > 1e-6 or (abs(dx1) <= 1e-6 and dy1 >= 0.0):
        return c1
    return c2


def crop_and_straighten_frame(
    rgb: np.ndarray,
    frame_detections: list[dict[str, Any]],
    geometry_bounds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Apply multi-frame policy, order corners, check geometry bounds, and perspective-warp the OBB crop."""
    bounds = {**DEFAULT_GEOMETRY_BOUNDS, **(geometry_bounds or {})}
    if len(frame_detections) != 1:
        return {
            "cropRgb": None,
            "reason": "no_row" if not frame_detections else "multiple_rows",
            "rowBox": None,
            "orderedPolygon": None,
            "cropSize": None,
            "perspectiveMatrix": None,
        }

    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    poly = np.asarray(frame_detections[0]["polygon"], dtype=np.float32)
    ordered = order_obb_corners(poly)
    tl, tr, br, bl = ordered

    xs = ordered[:, 0]
    ys = ordered[:, 1]
    left_box = int(min(max(math.floor(float(np.min(xs))), 0), width - 1))
    top_box = int(min(max(math.floor(float(np.min(ys))), 0), height - 1))
    right_box = int(min(max(math.ceil(float(np.max(xs))), left_box + 1), width))
    bottom_box = int(min(max(math.ceil(float(np.max(ys))), top_box + 1), height))
    row_box = [left_box, top_box, right_box, bottom_box]

    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    h_left = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))
    crop_w = int(max(round(w_top), round(w_bot)))
    crop_h = int(max(round(h_left), round(h_right)))

    aspect = (float(crop_w) / float(crop_h)) if crop_h > 0 else 0.0
    if (
        crop_w < int(bounds["minWidth"])
        or crop_h < int(bounds["minHeight"])
        or aspect < float(bounds["minAspect"])
        or aspect > float(bounds["maxAspect"])
    ):
        return {
            "cropRgb": None,
            "reason": "unusable_row_geometry",
            "rowBox": None,
            "orderedPolygon": [[float(p[0]), float(p[1])] for p in ordered],
            "cropSize": [crop_w, crop_h],
            "perspectiveMatrix": None,
        }

    dst = np.array(
        [
            [0.0, 0.0],
            [float(crop_w - 1), 0.0],
            [float(crop_w - 1), float(crop_h - 1)],
            [0.0, float(crop_h - 1)],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    crop_rgb = cv2.warpPerspective(
        rgb,
        matrix,
        (crop_w, crop_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return {
        "cropRgb": np.ascontiguousarray(crop_rgb),
        "reason": None,
        "rowBox": row_box,
        "orderedPolygon": [[float(p[0]), float(p[1])] for p in ordered],
        "cropSize": [crop_w, crop_h],
        "perspectiveMatrix": [[float(v) for v in row] for row in matrix],
    }


def axis_iou(box1: list[float] | np.ndarray, box2: list[float] | np.ndarray) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
    denom = area1 + area2 - inter
    return float(inter / denom) if denom > 0.0 else 0.0


def decode_axis_aligned_boxes(
    raw_output: np.ndarray,
    meta: dict[str, Any],
    names: dict[str, str],
    conf_thres: float = 0.25,
    iou_thres: float = 0.50,
    agnostic_nms: bool = True,
    max_det: int = 300,
) -> list[dict[str, Any]]:
    """Decode [1, 4+nc, 2100] detection output, apply NMS, and map boxes to crop coordinates."""
    nc = len(names)
    expected_shape = (1, 4 + nc, 2100)
    if raw_output.shape != expected_shape or not np.isfinite(raw_output).all():
        raise ValueError(
            f"E_INFERENCE: Invalid detection output shape {raw_output.shape}, expected {expected_shape}"
        )

    preds = raw_output[0]
    boxes_xywh = preds[:4]
    class_scores = preds[4:]

    best_cls = np.argmax(class_scores, axis=0)
    best_scores = class_scores[best_cls, np.arange(preds.shape[1])]
    candidate_indices = np.flatnonzero(best_scores >= conf_thres)

    candidates: list[tuple[int, int, str, float, list[float]]] = []
    for idx in candidate_indices:
        anchor = int(idx)
        cls_id = int(best_cls[anchor])
        score = float(best_scores[anchor])
        cx, cy, w, h = (float(boxes_xywh[k, anchor]) for k in range(4))
        if w <= 0.0 or h <= 0.0:
            continue
        key = str(cls_id)
        if key not in names:
            raise ValueError(f"Unsupported class index {cls_id} not in manifest names")
        label = str(names[key])
        net_box = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
        candidates.append((anchor, cls_id, label, score, net_box))

    candidates.sort(key=lambda item: (-item[3], item[0]))

    kept: list[tuple[int, int, str, float, list[float]]] = []
    for cand in candidates:
        if agnostic_nms:
            conflict = any(axis_iou(cand[4], prev[4]) >= iou_thres for prev in kept)
        else:
            conflict = any(cand[1] == prev[1] and axis_iou(cand[4], prev[4]) >= iou_thres for prev in kept)
        if not conflict:
            kept.append(cand)
            if len(kept) >= max_det:
                break

    gain = float(meta["gain"])
    left = float(meta["left"])
    top = float(meta["top"])
    width = float(meta["inputWidth"])
    height = float(meta["inputHeight"])

    results: list[dict[str, Any]] = []
    for anchor, cls_id, label, score, net_box in kept:
        x1 = min(max((net_box[0] - left) / gain, 0.0), width)
        y1 = min(max((net_box[1] - top) / gain, 0.0), height)
        x2 = min(max((net_box[2] - left) / gain, 0.0), width)
        y2 = min(max((net_box[3] - top) / gain, 0.0), height)
        results.append(
            {
                "anchor": anchor,
                "classId": cls_id,
                "class": label,
                "score": score,
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "cx": float((x1 + x2) / 2.0),
                "cy": float((y1 + y2) / 2.0),
            }
        )
    return results


def evaluate_reverse_trigger(
    reverse_detections: list[dict[str, Any]],
    asymmetric_labels: set[str] | list[str] = SUPPORTED_REVERSE_ASYMMETRIC,
    symmetric_labels: set[str] | list[str] = SUPPORTED_REVERSE_SYMMETRIC,
    min_total_boxes: int = 2,
) -> tuple[bool, str]:
    """Evaluate the approved 6-class reverse trigger rule on decoded reverse detections."""
    asym_set = {str(x) for x in asymmetric_labels}
    sym_set = {str(x) for x in symmetric_labels}
    allowed = asym_set | sym_set

    labels: list[str] = []
    for det in reverse_detections:
        label = str(det.get("class", ""))
        if label not in allowed:
            raise ValueError(f"Unsupported reverse check class label: {label!r}")
        labels.append(label)

    if any(lbl in asym_set for lbl in labels):
        return True, "asymmetric_reverse_label"
    if len(labels) >= int(min_total_boxes):
        return True, "multiple_reverse_boxes"
    if len(labels) == 1:
        return False, "single_symmetric_1n_ignored"
    return False, "no_reverse_boxes"


def map_digit_label(label: str, class_to_digit: dict[str, str] | None = None) -> str:
    """Map one of the 30 approved digit classes ('0'..'29') to '0'..'9' via int(label) % 10."""
    mapping = class_to_digit if class_to_digit is not None else DEFAULT_CLASS_TO_DIGIT
    key = str(label)
    if key not in mapping:
        raise ValueError(f"Unsupported digit class label: {label!r}")
    try:
        numeric = int(key)
    except ValueError as exc:
        raise ValueError(f"Non-numeric digit class label: {label!r}") from exc
    if numeric < 0 or numeric >= 30:
        raise ValueError(f"Out-of-range digit class label: {label!r}")
    expected_digit = str(numeric % 10)
    mapped = str(mapping[key])
    if mapped != expected_digit or len(mapped) != 1 or mapped not in "0123456789":
        raise ValueError(f"Invalid digit mapping for label {label!r}: {mapped!r}")
    return mapped


def decode_digits_reading(
    digit_detections: list[dict[str, Any]],
    class_to_digit: dict[str, str] | None = None,
    accepted_lengths: list[int] | None = None,
) -> dict[str, Any]:
    """Sort digit boxes left-to-right, map 30 classes to digits, and validate reading length."""
    valid_lengths = list(accepted_lengths) if accepted_lengths is not None else [5, 6]
    sorted_dets = sorted(
        digit_detections,
        key=lambda d: (float(d["box"][0]), float(d.get("cx", (d["box"][0] + d["box"][2]) / 2.0)), int(d["anchor"])),
    )

    mapped_digits: list[str] = []
    scores: list[float] = []
    classes_ltr: list[str] = []
    enriched_dets: list[dict[str, Any]] = []

    for det in sorted_dets:
        label = str(det["class"])
        digit = map_digit_label(label, class_to_digit)
        mapped_digits.append(digit)
        scores.append(float(det["score"]))
        classes_ltr.append(label)
        enriched_dets.append({**det, "digit": digit})

    raw = "".join(mapped_digits)
    confidence = float(np.mean(np.asarray(scores, dtype=np.float32))) if scores else 0.0

    if not raw:
        reason: str | None = "no_characters"
    elif any(ch not in "0123456789" for ch in raw):
        reason = "non_digit_output"
    elif len(raw) not in valid_lengths:
        reason = "unexpected_length"
    else:
        reason = None

    status = "ok" if reason is None else "uncertain"
    raw_digits = raw if status == "ok" else None
    return {
        "status": status,
        "rawDigits": raw_digits,
        "reason": reason,
        "confidence": confidence,
        "rawDiagnostic": raw,
        "digitClassesLeftToRight": classes_ltr,
        "sortedDetections": enriched_dets,
    }


def execute_three_stage_pipeline(
    rgb: np.ndarray,
    manifest: dict[str, Any],
    forward_fn: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the 3-stage pipeline using a pluggable raw-tensor forward_fn(role, input_tensor)."""
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    models_cfg = manifest["models"]
    policy_cfg = manifest["policy"]

    frame_cfg = models_cfg["frame"]
    reverse_cfg = models_cfg["reverse"]
    digits_cfg = models_cfg["digits"]

    # Stage 1: Frame OBB detection
    frame_input, frame_meta = letterbox_rgb(rgb, int(frame_cfg["imgsz"]))
    frame_output = forward_fn("frame", frame_input)
    frame_detections = decode_frame_obb(
        frame_output,
        frame_meta,
        names=frame_cfg["names"],
        conf_thres=float(frame_cfg["confThres"]),
        iou_thres=float(frame_cfg["iouThres"]),
        max_det=int(frame_cfg["maxDet"]),
    )

    tensors: dict[str, np.ndarray] = {
        "frame_input": frame_input,
        "frame_output": frame_output,
    }

    crop_info = crop_and_straighten_frame(
        rgb,
        frame_detections,
        geometry_bounds=policy_cfg.get("geometryBounds"),
    )

    if crop_info["cropRgb"] is None:
        result = {
            "pipelineId": manifest["pipelineId"],
            "semanticsVersion": manifest["semanticsVersion"],
            "status": "uncertain",
            "rawDigits": None,
            "reason": crop_info["reason"],
            "confidence": 0.0,
            "rawDiagnostic": "",
            "isReversed": False,
            "reverseTriggerReason": "not_run",
            "rowBox": crop_info["rowBox"],
            "orderedPolygon": crop_info["orderedPolygon"],
            "cropSize": crop_info["cropSize"],
            "perspectiveMatrix": crop_info["perspectiveMatrix"],
            "imageSize": [width, height],
            "imagePixelSha256": sha256_bytes(rgb.tobytes()),
            "frameCount": len(frame_detections),
            "frameDetections": frame_detections,
            "reverseDetections": [],
            "revClasses": [],
            "digitDetections": [],
            "digitClassesLeftToRight": [],
        }
        return result, tensors

    crop_rgb = crop_info["cropRgb"]
    tensors["crop_rgb"] = crop_rgb

    # Stage 2: Reverse orientation check on straightened crop
    rev_input, rev_meta = letterbox_rgb(crop_rgb, int(reverse_cfg["imgsz"]))
    rev_output = forward_fn("reverse", rev_input)
    rev_detections = decode_axis_aligned_boxes(
        rev_output,
        rev_meta,
        names=reverse_cfg["names"],
        conf_thres=float(reverse_cfg["confThres"]),
        iou_thres=float(reverse_cfg["iouThres"]),
        agnostic_nms=bool(reverse_cfg.get("agnosticNms", True)),
        max_det=int(reverse_cfg["maxDet"]),
    )
    is_reversed, rev_trigger_reason = evaluate_reverse_trigger(
        rev_detections,
        asymmetric_labels=reverse_cfg["asymmetricLabels"],
        symmetric_labels=reverse_cfg.get("symmetricLabels", ["1n"]),
        min_total_boxes=int(reverse_cfg["minTotalBoxesForSymmetricTrigger"]),
    )

    if is_reversed:
        crop_used = np.ascontiguousarray(cv2.rotate(crop_rgb, cv2.ROTATE_180))
    else:
        crop_used = crop_rgb
    tensors["crop_used_rgb"] = crop_used

    # Stage 3: 30-class digit recognition on upright/rotated crop
    digits_input, digits_meta = letterbox_rgb(crop_used, int(digits_cfg["imgsz"]))
    digits_output = forward_fn("digits", digits_input)
    digit_detections = decode_axis_aligned_boxes(
        digits_output,
        digits_meta,
        names=digits_cfg["names"],
        conf_thres=float(digits_cfg["confThres"]),
        iou_thres=float(digits_cfg["iouThres"]),
        agnostic_nms=bool(digits_cfg.get("agnosticNms", True)),
        max_det=int(digits_cfg["maxDet"]),
    )
    reading = decode_digits_reading(
        digit_detections,
        class_to_digit=digits_cfg["classToDigit"],
        accepted_lengths=policy_cfg["acceptedLengths"],
    )

    tensors.update(
        reverse_input=rev_input,
        reverse_output=rev_output,
        digits_input=digits_input,
        digits_output=digits_output,
    )

    result = {
        "pipelineId": manifest["pipelineId"],
        "semanticsVersion": manifest["semanticsVersion"],
        "status": reading["status"],
        "rawDigits": reading["rawDigits"],
        "reason": reading["reason"],
        "confidence": reading["confidence"],
        "rawDiagnostic": reading["rawDiagnostic"],
        "isReversed": is_reversed,
        "reverseTriggerReason": rev_trigger_reason,
        "rowBox": crop_info["rowBox"],
        "orderedPolygon": crop_info["orderedPolygon"],
        "cropSize": crop_info["cropSize"],
        "perspectiveMatrix": crop_info["perspectiveMatrix"],
        "imageSize": [width, height],
        "imagePixelSha256": sha256_bytes(rgb.tobytes()),
        "frameCount": len(frame_detections),
        "frameDetections": frame_detections,
        "reverseDetections": rev_detections,
        "revClasses": [d["class"] for d in rev_detections],
        "digitDetections": reading["sortedDetections"],
        "digitClassesLeftToRight": reading["digitClassesLeftToRight"],
        "cropPixelSha256": sha256_bytes(crop_rgb.tobytes()),
        "cropUsedPixelSha256": sha256_bytes(crop_used.tobytes()),
    }
    return result, tensors


class OnnxBundleRunner:
    """Host reference runner backed by ONNX Runtime CPU and supplied-yolo-manifest.json."""

    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = bundle_dir
        manifest_path = bundle_dir / "supplied-yolo-manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing bundle manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest_sha256 = sha256_file(manifest_path)

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.inter_op_num_threads = 1

        self.sessions: dict[str, ort.InferenceSession] = {}
        for role in ("frame", "reverse", "digits"):
            cfg = self.manifest["models"][role]
            onnx_path = bundle_dir / cfg["file"]
            if not onnx_path.exists():
                raise FileNotFoundError(f"Missing ONNX model for {role}: {onnx_path}")
            if sha256_file(onnx_path) != cfg["sha256"]:
                raise ValueError(f"ONNX checksum mismatch for {role}: {onnx_path}")
            self.sessions[role] = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

    def forward(self, role: str, tensor: np.ndarray) -> np.ndarray:
        session = self.sessions[role]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        out = session.run([output_name], {input_name: tensor})[0]
        if not np.isfinite(out).all():
            raise ValueError(f"E_INFERENCE: Non-finite ONNX output in {role}")
        return out

    def run_rgb(self, rgb: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        return execute_three_stage_pipeline(rgb, self.manifest, self.forward)


class PytorchCheckpointRunner:
    """Host reference runner backed by PyTorch raw forward pass on the source .pt checkpoints."""

    def __init__(self, manifest: dict[str, Any], repo_root: Path = Path(".")) -> None:
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(2)
        self._torch = torch
        self.manifest = manifest
        self.models: dict[str, Any] = {}
        for role in ("frame", "reverse", "digits"):
            cfg = manifest["models"][role]
            pt_path = repo_root / cfg["sourceFile"]
            if not pt_path.exists():
                raise FileNotFoundError(f"Missing source checkpoint for {role}: {pt_path}")
            if sha256_file(pt_path) != cfg["sourceSha256"]:
                raise ValueError(f"Source checkpoint SHA-256 mismatch for {role}: {pt_path}")
            self.models[role] = YOLO(str(pt_path)).model.eval().cpu()

    def forward(self, role: str, tensor: np.ndarray) -> np.ndarray:
        model = self.models[role]
        with self._torch.no_grad():
            out = model(self._torch.from_numpy(tensor))[0].detach().cpu().numpy()
        if not np.isfinite(out).all():
            raise ValueError(f"E_INFERENCE: Non-finite PyTorch output in {role}")
        return out

    def run_rgb(self, rgb: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        return execute_three_stage_pipeline(rgb, self.manifest, self.forward)


def compare_tensor_pair(
    pt_tensor: np.ndarray,
    ort_tensor: np.ndarray,
    coord_atol: float,
    score_atol: float,
    rtol: float,
) -> dict[str, Any]:
    abs_diff = np.abs(pt_tensor - ort_tensor)
    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    denom = np.maximum(np.abs(pt_tensor), 1e-3)
    max_rel = float(np.max(abs_diff / denom))

    coord_diff = np.abs(pt_tensor[:, :4, :] - ort_tensor[:, :4, :])
    score_diff = np.abs(pt_tensor[:, 4:, :] - ort_tensor[:, 4:, :])
    coord_passed = bool(np.allclose(pt_tensor[:, :4, :], ort_tensor[:, :4, :], atol=coord_atol, rtol=rtol))
    score_passed = bool(np.allclose(pt_tensor[:, 4:, :], ort_tensor[:, 4:, :], atol=score_atol, rtol=rtol))
    passed = coord_passed and score_passed
    return {
        "shape": list(ort_tensor.shape),
        "maxAbsDiff": max_abs,
        "meanAbsDiff": mean_abs,
        "maxRelDiff": max_rel,
        "maxCoordPxDiff": float(np.max(coord_diff)),
        "maxScoreDiff": float(np.max(score_diff)),
        "coordAtol": coord_atol,
        "scoreAtol": score_atol,
        "rtol": rtol,
        "passed": passed,
    }


def generate_exif_fixtures(
    source_image_path: Path,
    onnx_runner: OnnxBundleRunner,
    exif_out_dir: Path,
) -> list[dict[str, Any]]:
    """Create 8 EXIF-tagged PNG test images and their oriented frame_input.f32 tensors for Android parity."""
    exif_out_dir.mkdir(parents=True, exist_ok=True)
    base_rgb, _ = load_rgb_oriented(source_image_path)
    # Use a compact 160x120 resize so 8 EXIF fixtures stay lightweight while asymmetric in W != H
    base_small = cv2.resize(base_rgb, (160, 120), interpolation=cv2.INTER_LINEAR)
    expected_input, _ = letterbox_rgb(base_small, int(onnx_runner.manifest["models"]["frame"]["imgsz"]))
    expected_input_sha = sha256_bytes(expected_input.astype("<f4").tobytes())

    inverse_ops = {
        1: lambda img: img,
        2: lambda img: np.fliplr(img),
        3: lambda img: np.rot90(img, 2),
        4: lambda img: np.flipud(img),
        5: lambda img: np.rot90(np.fliplr(img), 1),
        6: lambda img: np.rot90(img, 1),
        7: lambda img: np.rot90(np.fliplr(img), 3),
        8: lambda img: np.rot90(img, 3),
    }

    records: list[dict[str, Any]] = []
    for orientation in range(1, 9):
        raw_pixels = np.ascontiguousarray(inverse_ops[orientation](base_small))
        pil_img = Image.fromarray(raw_pixels)
        exif = pil_img.getexif()
        exif[274] = orientation
        img_path = exif_out_dir / f"exif-{orientation}.png"
        pil_img.save(img_path, format="PNG", exif=exif.tobytes())

        reloaded_rgb, detected_orientation = load_rgb_oriented(img_path)
        if detected_orientation != orientation:
            raise RuntimeError(f"Failed to round-trip EXIF orientation {orientation}")
        tensor, _ = letterbox_rgb(reloaded_rgb, int(onnx_runner.manifest["models"]["frame"]["imgsz"]))
        f32_path = exif_out_dir / f"exif-{orientation}.frame_input.f32"
        tensor.astype("<f4").tofile(f32_path)
        tensor_sha = sha256_file(f32_path)
        if not np.allclose(tensor, expected_input, atol=1e-6, rtol=0.0):
            raise RuntimeError(f"EXIF {orientation} normalized tensor does not match orientation 1")

        records.append(
            {
                "orientation": orientation,
                "imageFile": f"exif/{img_path.name}",
                "imageSha256": sha256_file(img_path),
                "tensorFile": f"exif/{f32_path.name}",
                "tensorSha256": tensor_sha,
                "matchesCanonicalOrientation1": tensor_sha == expected_input_sha,
            }
        )
    return records


def run_host_reference_suite(
    bundle_dir: Path,
    fixtures_manifest_path: Path,
    out_dir: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir = out_dir / "tensors"
    crops_dir = out_dir / "crops"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    fixtures_manifest = json.loads(fixtures_manifest_path.read_text(encoding="utf-8"))
    onnx_runner = OnnxBundleRunner(bundle_dir)
    pt_runner = PytorchCheckpointRunner(onnx_runner.manifest, repo_root=repo_root)

    tolerances = onnx_runner.manifest["tolerances"]
    coord_atol = float(tolerances.get("rawCoordPxAtol", 0.01))
    raw_score_atol = float(tolerances.get("rawScoreAtol", 0.0005))
    raw_rtol = float(tolerances["rawTensorRtol"])
    score_atol = float(tolerances["hostScoreAtol"])
    box_px_atol = float(tolerances["hostBoxPxAtol"])

    fixture_reports: list[dict[str, Any]] = []
    all_pass = True

    for fixture in fixtures_manifest["fixtures"]:
        fx_id = fixture["id"]
        fx_path = repo_root / fixture["path"]
        if not fx_path.exists():
            raise FileNotFoundError(f"Missing fixture file for {fx_id}: {fx_path}")
        actual_sha = sha256_file(fx_path).upper()
        if actual_sha != str(fixture["sha256"]).upper():
            raise ValueError(
                f"Fixture SHA-256 drift for {fx_id}: expected {fixture['sha256']}, got {actual_sha}"
            )

        expected = fixture["expected"]
        if expected.get("promiseRejected"):
            onnx_err: str | None = None
            pt_err: str | None = None
            try:
                rgb, _ = load_rgb_oriented(fx_path)
                onnx_runner.run_rgb(rgb)
            except ValueError as exc:
                onnx_err = str(exc)
            try:
                rgb, _ = load_rgb_oriented(fx_path)
                pt_runner.run_rgb(rgb)
            except ValueError as exc:
                pt_err = str(exc)

            passed = (
                onnx_err is not None
                and pt_err is not None
                and expected["errorCode"] in onnx_err
                and expected["errorCode"] in pt_err
            )
            all_pass = all_pass and passed
            fx_report = {
                "id": fx_id,
                "category": fixture["category"],
                "path": fixture["path"],
                "sha256": actual_sha,
                "promiseRejected": True,
                "onnxError": onnx_err,
                "pytorchError": pt_err,
                "passed": passed,
            }
            (out_dir / f"{fx_id}.result.json").write_text(
                json.dumps(fx_report, indent=2) + "\n", encoding="utf-8"
            )
            fixture_reports.append(fx_report)
            continue

        rgb, orientation = load_rgb_oriented(fx_path)
        onnx_res, onnx_tensors = onnx_runner.run_rgb(rgb)
        pt_res, _ = pt_runner.run_rgb(rgb)

        # Save crops and private debug tensors with SHA-256 inventory
        saved_tensors: dict[str, dict[str, Any]] = {}
        for t_name in (
            "frame_input",
            "frame_output",
            "reverse_input",
            "reverse_output",
            "digits_input",
            "digits_output",
        ):
            if t_name in onnx_tensors:
                arr = onnx_tensors[t_name].astype("<f4")
                t_file = tensors_dir / f"{fx_id}.{t_name}.f32"
                arr.tofile(t_file)
                saved_tensors[t_name] = {
                    "file": f"tensors/{t_file.name}",
                    "shape": list(arr.shape),
                    "dtype": "float32-le",
                    "sha256": sha256_file(t_file),
                }

        saved_crops: dict[str, str] = {}
        if "crop_rgb" in onnx_tensors:
            crop_file = crops_dir / f"{fx_id}.crop.png"
            cv2.imwrite(str(crop_file), onnx_tensors["crop_rgb"][:, :, ::-1])
            saved_crops["straightenedCrop"] = f"crops/{crop_file.name}"
        if "crop_used_rgb" in onnx_tensors:
            crop_used_file = crops_dir / f"{fx_id}.crop_used.png"
            cv2.imwrite(str(crop_used_file), onnx_tensors["crop_used_rgb"][:, :, ::-1])
            saved_crops["digitsInputCrop"] = f"crops/{crop_used_file.name}"

        # Compare raw tensors between PyTorch and ONNX Runtime on the EXACT SAME input tensor
        tensor_parity: dict[str, Any] = {}
        for role in ("frame", "reverse", "digits"):
            in_key = f"{role}_input"
            out_key = f"{role}_output"
            if in_key in onnx_tensors and out_key in onnx_tensors:
                pt_same_input_out = pt_runner.forward(role, onnx_tensors[in_key])
                tensor_parity[out_key] = compare_tensor_pair(
                    pt_same_input_out,
                    onnx_tensors[out_key],
                    coord_atol=coord_atol,
                    score_atol=raw_score_atol,
                    rtol=raw_rtol,
                )

        # Compare post-NMS detections and geometry between PyTorch and ONNX
        max_polygon_diff = 0.0
        if onnx_res["orderedPolygon"] is not None and pt_res["orderedPolygon"] is not None:
            poly_diff = np.abs(
                np.asarray(onnx_res["orderedPolygon"]) - np.asarray(pt_res["orderedPolygon"])
            )
            max_polygon_diff = float(np.max(poly_diff))

        max_digit_box_diff = 0.0
        max_digit_score_diff = 0.0
        if len(onnx_res["digitDetections"]) == len(pt_res["digitDetections"]):
            for od, pd in zip(onnx_res["digitDetections"], pt_res["digitDetections"]):
                b_diff = float(np.max(np.abs(np.asarray(od["box"]) - np.asarray(pd["box"]))))
                s_diff = abs(float(od["score"]) - float(pd["score"]))
                max_digit_box_diff = max(max_digit_box_diff, b_diff)
                max_digit_score_diff = max(max_digit_score_diff, s_diff)

        pt_vs_onnx_checks = {
            "statusMatch": onnx_res["status"] == pt_res["status"],
            "rawDigitsMatch": onnx_res["rawDigits"] == pt_res["rawDigits"],
            "reasonMatch": onnx_res["reason"] == pt_res["reason"],
            "isReversedMatch": onnx_res["isReversed"] == pt_res["isReversed"],
            "frameCountMatch": onnx_res["frameCount"] == pt_res["frameCount"],
            "rowBoxMatch": onnx_res["rowBox"] == pt_res["rowBox"],
            "cropSizeMatch": onnx_res["cropSize"] == pt_res["cropSize"],
            "revClassesMatch": onnx_res["revClasses"] == pt_res["revClasses"],
            "digitClassesMatch": onnx_res["digitClassesLeftToRight"] == pt_res["digitClassesLeftToRight"],
            "maxPolygonPxDiff": max_polygon_diff,
            "polygonWithinTolerance": max_polygon_diff <= box_px_atol,
            "maxDigitBoxPxDiff": max_digit_box_diff,
            "digitBoxesWithinTolerance": max_digit_box_diff <= box_px_atol,
            "maxDigitScoreDiff": max_digit_score_diff,
            "digitScoresWithinTolerance": max_digit_score_diff <= score_atol,
            "allRawTensorsPassed": all(v["passed"] for v in tensor_parity.values()),
        }

        expected_checks = {
            "statusMatch": onnx_res["status"] == expected["status"],
            "rawDigitsMatch": onnx_res["rawDigits"] == expected["rawDigits"],
            "reasonMatch": onnx_res["reason"] == expected["reason"],
        }
        for field, check_key in (
            ("isReversed", "isReversedMatch"),
            ("rowBox", "rowBoxMatch"),
            ("cropSize", "cropSizeMatch"),
            ("revClasses", "revClassesMatch"),
            ("digitClassesLeftToRight", "digitClassesMatch"),
            ("frameCount", "frameCountMatch"),
        ):
            if field in expected:
                expected_checks[check_key] = onnx_res[field] == expected[field]

        passed = all(
            v if isinstance(v, bool) else True for v in pt_vs_onnx_checks.values()
        ) and all(expected_checks.values())
        all_pass = all_pass and passed

        fx_report = {
            "id": fx_id,
            "category": fixture["category"],
            "path": fixture["path"],
            "sha256": actual_sha,
            "exifOrientation": orientation,
            "expected": expected,
            "onnxResult": onnx_res,
            "pytorchResult": pt_res,
            "tensorParity": tensor_parity,
            "ptVsOnnxChecks": pt_vs_onnx_checks,
            "expectedChecks": expected_checks,
            "debugTensors": saved_tensors,
            "debugCrops": saved_crops,
            "passed": passed,
        }
        (out_dir / f"{fx_id}.result.json").write_text(
            json.dumps(fx_report, indent=2) + "\n", encoding="utf-8"
        )
        fixture_reports.append(fx_report)

    exif_records = generate_exif_fixtures(
        repo_root / "data/1.jfif",
        onnx_runner,
        out_dir / "exif",
    )

    summary_report = {
        "schemaVersion": 1,
        "semanticsVersion": onnx_runner.manifest["semanticsVersion"],
        "pipelineId": onnx_runner.manifest["pipelineId"],
        "manifestSha256": onnx_runner.manifest_sha256,
        "tolerances": tolerances,
        "fixtureCount": len(fixture_reports),
        "allPass": all_pass,
        "exifOrientationParity": {
            "count": len(exif_records),
            "allMatchCanonical": all(r["matchesCanonicalOrientation1"] for r in exif_records),
            "records": exif_records,
        },
        "fixtures": [
            {
                "id": r["id"],
                "category": r["category"],
                "passed": r["passed"],
                "status": r.get("onnxResult", {}).get("status", "rejected"),
                "rawDigits": r.get("onnxResult", {}).get("rawDigits"),
                "reason": r.get("onnxResult", {}).get("reason", r.get("onnxError")),
                "isReversed": r.get("onnxResult", {}).get("isReversed"),
                "rowBox": r.get("onnxResult", {}).get("rowBox"),
                "tensorParity": r.get("tensorParity", {}),
                "ptVsOnnxChecks": r.get("ptVsOnnxChecks", {}),
                "expectedChecks": r.get("expectedChecks", {}),
            }
            for r in fixture_reports
        ],
    }

    report_path = out_dir / "parity-report.json"
    report_path.write_text(json.dumps(summary_report, indent=2) + "\n", encoding="utf-8")

    # Write human-readable markdown summary
    md_lines = [
        "# Host Parity Report (`supplied-yolo-3stage-v1`)",
        "",
        f"- **Manifest SHA-256:** `{onnx_runner.manifest_sha256}`",
        f"- **Total Fixtures:** `{len(fixture_reports)}`",
        f"- **All Checks Passed:** `{all_pass}`",
        f"- **Declared Tolerances:** `rawCoordPxAtol={coord_atol}px`, `rawScoreAtol={raw_score_atol}`, `rawTensorRtol={raw_rtol}`, `hostBoxPxAtol={box_px_atol}px`, `hostScoreAtol={score_atol}`",
        "",
        "| Fixture ID | Category | Status | Raw Digits | Reversed | Reason | Frame Max Abs Diff | Reverse Max Abs Diff | Digits Max Abs Diff | Poly Diff (px) | Pass |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in fixture_reports:
        if r.get("promiseRejected"):
            md_lines.append(
                f"| `{r['id']}` | `{r['category']}` | `rejected` | `null` | `-` | `{r['onnxError']}` | `-` | `-` | `-` | `-` | `{r['passed']}` |"
            )
            continue
        res = r["onnxResult"]
        tp = r["tensorParity"]
        f_diff = f"{tp['frame_output']['maxAbsDiff']:.2e}" if "frame_output" in tp else "-"
        r_diff = f"{tp['reverse_output']['maxAbsDiff']:.2e}" if "reverse_output" in tp else "-"
        d_diff = f"{tp['digits_output']['maxAbsDiff']:.2e}" if "digits_output" in tp else "-"
        p_diff = f"{r['ptVsOnnxChecks']['maxPolygonPxDiff']:.4f}"
        md_lines.append(
            f"| `{r['id']}` | `{r['category']}` | `{res['status']}` | `{res['rawDigits']}` | `{res['isReversed']}` | `{res['reason']}` | `{f_diff}` | `{r_diff}` | `{d_diff}` | `{p_diff}` | `{r['passed']}` |"
        )

    (out_dir / "parity-summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    if not all_pass:
        raise RuntimeError("One or more host parity checks failed! See parity-report.json for details.")
    return summary_report


def stage_android_gate_assets(
    fixtures_manifest_path: Path,
    host_report_dir: Path,
    gate_assets_dir: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Populate .cache/android-gates-assets with versioned Phase 2 fixtures, tensors, EXIF images, and guard images."""
    import shutil

    parity_report_path = host_report_dir / "parity-report.json"
    if not parity_report_path.exists():
        raise FileNotFoundError(f"Host parity report not found: {parity_report_path}. Run host reference suite first.")
    parity_report = json.loads(parity_report_path.read_text(encoding="utf-8"))
    if parity_report.get("semanticsVersion") != "supplied-yolo-3stage-v1" or not parity_report.get("allPass"):
        raise ValueError("Host parity report must be passing supplied-yolo-3stage-v1 before staging gate assets.")

    fixtures_manifest = json.loads(fixtures_manifest_path.read_text(encoding="utf-8"))
    if gate_assets_dir.exists():
        shutil.rmtree(gate_assets_dir)
    gate_assets_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(parity_report_path, gate_assets_dir / "parity-report.json")
    shutil.copy2(fixtures_manifest_path, gate_assets_dir / "fixtures-manifest.json")

    for fixture in fixtures_manifest["fixtures"]:
        fx_id = fixture["id"]
        fx_path = repo_root / fixture["path"]
        res_src = host_report_dir / f"{fx_id}.result.json"
        if not res_src.exists():
            raise FileNotFoundError(f"Missing fixture result JSON: {res_src}")
        shutil.copy2(res_src, gate_assets_dir / f"{fx_id}.result.json")

        if fixture["expected"].get("promiseRejected"):
            shutil.copy2(fx_path, gate_assets_dir / f"{fx_id}.bin")
            shutil.copy2(fx_path, gate_assets_dir / "corrupt.img")
            continue

        rgb, _ = load_rgb_oriented(fx_path)
        png_dst = gate_assets_dir / f"{fx_id}.png"
        cv2.imwrite(str(png_dst), rgb[:, :, ::-1])

        if fx_path.suffix.lower() == ".jfif":
            shutil.copy2(fx_path, gate_assets_dir / f"{fx_id}.jfif")

        for t_name in (
            "frame_input",
            "frame_output",
            "reverse_input",
            "reverse_output",
            "digits_input",
            "digits_output",
        ):
            t_src = host_report_dir / "tensors" / f"{fx_id}.{t_name}.f32"
            if t_src.exists():
                shutil.copy2(t_src, gate_assets_dir / f"{fx_id}.{t_name}.f32")

    shutil.copy2(repo_root / "data/1.jfif", gate_assets_dir / "sample.jfif")
    shutil.copy2(gate_assets_dir / "meter-upright-01.png", gate_assets_dir / "lifecycle.png")

    for orientation in range(1, 9):
        exif_png = host_report_dir / "exif" / f"exif-{orientation}.png"
        exif_f32 = host_report_dir / "exif" / f"exif-{orientation}.frame_input.f32"
        if not exif_png.exists() or not exif_f32.exists():
            raise FileNotFoundError(f"Missing EXIF fixture for orientation {orientation} in {host_report_dir / 'exif'}")
        shutil.copy2(exif_png, gate_assets_dir / exif_png.name)
        shutil.copy2(exif_f32, gate_assets_dir / exif_f32.name)

    # Generate compact valid PNGs that trip the 12 MP pixel limit and 16384 dimension limit
    Image.new("L", (4000, 3001), 0).save(gate_assets_dir / "oversize.png", format="PNG", optimize=True)
    Image.new("L", (16385, 1), 0).save(gate_assets_dir / "oversize-dimension.png", format="PNG", optimize=True)

    staged_files = []
    for item in sorted(gate_assets_dir.iterdir(), key=lambda p: p.name):
        if item.is_file():
            staged_files.append(
                {
                    "name": item.name,
                    "sizeBytes": item.stat().st_size,
                    "sha256": sha256_file(item),
                }
            )

    gates_manifest = {
        "schemaVersion": 1,
        "pipelineId": parity_report["pipelineId"],
        "semanticsVersion": parity_report["semanticsVersion"],
        "manifestSha256": parity_report["manifestSha256"],
        "fileCount": len(staged_files),
        "files": staged_files,
    }
    (gate_assets_dir / "gates-manifest.json").write_text(
        json.dumps(gates_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return gates_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path("artifacts/supplied-yolo"),
        help="Staging bundle directory containing supplied-yolo-manifest.json and .onnx files",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("artifacts/supplied-yolo/fixtures/manifest.json"),
        help="Path to approved fixtures manifest.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/supplied-yolo/host-report"),
        help="Output directory for host parity reports, crops, and debug tensors",
    )
    parser.add_argument(
        "--stage-gate-assets",
        type=Path,
        default=None,
        help="Optional directory (e.g. .cache/android-gates-assets) to stage Android instrumentation test assets",
    )
    parser.add_argument(
        "--stage-gate-only",
        action="store_true",
        help="When combined with --stage-gate-assets, stage from existing --out directory without re-running inference",
    )
    args = parser.parse_args(argv)
    if args.stage_gate_only:
        if args.stage_gate_assets is None:
            raise ValueError("--stage-gate-only requires --stage-gate-assets")
        staged = stage_android_gate_assets(args.fixtures, args.out, args.stage_gate_assets)
        print(
            json.dumps(
                {
                    "semanticsVersion": staged["semanticsVersion"],
                    "gateAssetsDir": str(args.stage_gate_assets),
                    "fileCount": staged["fileCount"],
                },
                indent=2,
            )
        )
        return 0

    report = run_host_reference_suite(args.bundle, args.fixtures, args.out)
    gate_info = None
    if args.stage_gate_assets is not None:
        gate_info = stage_android_gate_assets(args.fixtures, args.out, args.stage_gate_assets)
    print(
        json.dumps(
            {
                "semanticsVersion": report["semanticsVersion"],
                "fixtureCount": report["fixtureCount"],
                "allPass": report["allPass"],
                "exifAllMatch": report["exifOrientationParity"]["allMatchCanonical"],
                "stagedGateAssetCount": gate_info["fileCount"] if gate_info else 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
