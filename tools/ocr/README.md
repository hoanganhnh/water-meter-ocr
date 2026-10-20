# Công cụ Export ONNX & Đối chiếu Host (`tools/ocr/`)

Thư mục `tools/ocr/` chứa các công cụ Python để xuất 3 checkpoint YOLOv8 được cung cấp (`models/best_latest.pt`, `models/best_0608.pt`, `models/best_1608.pt`) sang ONNX và kiểm chứng tính tương đương số học (parity) giữa PyTorch, ONNX Runtime Host và Android.

## Môi trường Python

Sử dụng môi trường ảo đã khóa phiên bản tại `.cache/yolo-env` (thông tin chi tiết được ghi tự động tại `artifacts/supplied-yolo/python-env-lock.json` và `requirements-lock.txt`):
- Python `3.12.10`
- `torch==2.11.0+cpu`
- `ultralytics==8.3.221`
- `onnx==1.21.0`
- `onnxruntime==1.24.4`
- `opencv-python-headless==4.13.0.92`
- `numpy==2.2.6`, `pillow==12.2.0`

## Các lệnh chính

### 1. Xuất ONNX và cập nhật `app/android/app/src/main/assets/ocr`

```powershell
.\.cache\yolo-env\Scripts\python.exe -m tools.ocr.export_supplied_yolo `
  --models models `
  --out artifacts/supplied-yolo `
  --stage-android-assets app/android/app/src/main/assets/ocr
```

Nếu đã có sẵn bundle trong `artifacts/supplied-yolo` và chỉ muốn đồng bộ sang Android assets (đồng thời dọn các file model cũ):

```powershell
.\.cache\yolo-env\Scripts\python.exe -m tools.ocr.export_supplied_yolo `
  --out artifacts/supplied-yolo `
  --stage-android-assets app/android/app/src/main/assets/ocr `
  --stage-only
```

### 2. Chạy Host Reference Suite và tạo tài nguyên cho Android Instrumentation Tests

```powershell
.\.cache\yolo-env\Scripts\python.exe -m tools.ocr.supplied_yolo_reference `
  --bundle artifacts/supplied-yolo `
  --fixtures artifacts/supplied-yolo/fixtures/manifest.json `
  --out artifacts/supplied-yolo/host-report `
  --stage-gate-assets .cache/android-gates-assets
```

Lệnh này thực hiện:
- Chạy cả PyTorch và ONNX Runtime trên toàn bộ 8 fixture trong `artifacts/supplied-yolo/fixtures/manifest.json`.
- Lưu báo cáo tổng hợp `parity-report.json`, bảng tóm tắt `parity-summary.md`, ảnh crop trung gian (`crops/`) và các tensor `.f32` (`tensors/`).
- Sinh 8 ảnh kiểm thử hướng EXIF (`exif/exif-1..8.png` và `exif/exif-1..8.frame_input.f32`).
- Đóng gói toàn bộ dữ liệu kiểm thử kèm `gates-manifest.json` vào `.cache/android-gates-assets` cho `PhaseThreeGatesTest`.

### 3. Chạy bộ kiểm thử hợp đồng Python

```powershell
.\.cache\yolo-env\Scripts\python.exe -m unittest tools.ocr.tests.test_supplied_yolo_contract -v
```
