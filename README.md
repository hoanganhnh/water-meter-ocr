# Water Meter OCR

Ứng dụng React Native đọc chỉ số đồng hồ nước từ ảnh trên thiết bị. Native Android dùng Kotlin, ONNX Runtime và OpenCV để chạy pipeline `supplied-yolo-3stage-v1` hoàn toàn offline; iOS hiện chỉ có scaffold.

Ba checkpoint được cung cấp đảm nhiệm lần lượt **tìm khung OBB → check ngược → đọc chữ số**. `.pt` được giữ nguyên trên máy phát triển, chuyển sang ONNX để đóng vào APK; app không chạy Python. Giao diện nhận chuỗi kết quả và vùng số qua TurboModule.

## Bắt đầu


| Cần làm                                                      | Tài liệu sở hữu                                                                            |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Hiểu công nghệ, pipeline và cách nối React Native với Kotlin | [Kiến trúc tích hợp YOLO](docs/architecture.md)                                            |
| Build APK, chạy app và kiểm thử Android                      | [Hướng dẫn app](app/README.md)                                                             |
| Export checkpoint, tạo host reference và stage assets        | [Công cụ model](tools/ocr/README.md)                                                       |
| Tra model, tensor shape, mapping, policy và hash đang dùng   | [Manifest bundle Android](app/android/app/src/main/assets/ocr/supplied-yolo-manifest.json) |
