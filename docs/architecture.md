# Tích hợp YOLO vào native Android

Pipeline `supplied-yolo-3stage-v1` dùng ba checkpoint được cung cấp để tìm khung số, xác định hướng và đọc chữ số. React Native phụ trách chọn ảnh/hiển thị; Kotlin xử lý ảnh và suy luận trên thiết bị. Android đã có triển khai; iOS hiện chỉ có scaffold.

## Công nghệ và ranh giới

| Thành phần     | Công nghệ                                                       | Vai trò / nơi cấu hình                                                                                                                                                         |
| -------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Giao diện      | React Native 0.87, React 19.2, TypeScript, Hermes               | Picker dùng `react-native-image-crop-picker`; phiên bản và Codegen tại [package.json](../app/package.json), runtime tại [gradle.properties](../app/android/gradle.properties). |
| Bridge         | TurboModule + React Native Codegen                              | [NativeMeterOcr.ts](../app/src/specs/NativeMeterOcr.ts) là hợp đồng JS/native; Codegen tạo lớp spec để Kotlin kế thừa.                                                         |
| Native         | Kotlin 2.2, Android Bitmap/Matrix, AndroidX ExifInterface 1.4.2 | Đọc URI, quản lý tài nguyên, chuẩn hóa EXIF; cấu hình tại [Android build](../app/android/build.gradle) và [app build](../app/android/app/build.gradle).                        |
| Suy luận       | ONNX Runtime Android 1.20, CPU                                  | Chạy model FP32 bằng `OrtSession`, không cần dịch vụ suy luận bên ngoài.                                                                                                       |
| Xử lý ảnh      | OpenCV Android 4.12                                             | Resize, biến đổi phối cảnh, lấy vùng số; đóng gói runtime native qua Gradle.                                                                                                   |
| Chuẩn bị model | Python, Ultralytics/PyTorch, ONNX, ONNX Runtime, OpenCV         | Export và tạo kết quả tham chiếu; môi trường/lệnh tái lập tại [tools/ocr](../tools/ocr/README.md).                                                                             |

Chọn ONNX để tái sử dụng ONNX Runtime Android và Kotlin hiện có. Tách export khỏi app giúp giữ nguyên checkpoint `.pt`, kiểm tra model trước khi đóng APK và chạy offline. TFLite hoặc backend suy luận cần luồng chuyển đổi/kiểm chứng khác, không thuộc kiến trúc này.

## Từ checkpoint tới APK

Nguồn cấu hình model, nhãn, tensor shape, hash và chính sách đọc là [supplied-yolo-manifest.json](../app/android/app/src/main/assets/ocr/supplied-yolo-manifest.json).

| Thứ tự chạy | Nguồn → asset ONNX | Input → raw output | Ý nghĩa |
|---|---|---|---|
| 1. Tìm khung | `models/best_latest.pt` → `frame.onnx` | `[1,3,640,640]` → `[1,6,8400]` | YOLOv8n-OBB, lớp `khung`; hộp có tâm, kích thước và góc xoay. |
| 2. Check ngược | `models/best_1608.pt` → `reverse.onnx` | `[1,3,320,320]` → `[1,10,2100]` | Detector nhãn chữ số ngược, quyết định có xoay crop 180° hay không. |
| 3. Đọc số | `models/best_0608.pt` → `digits.onnx` | `[1,3,320,320]` → `[1,34,2100]` | Detector 30 lớp; ánh xạ nhãn thành chữ số và ghép chuỗi. |

[Exporter](../tools/ocr/export_supplied_yolo.py) xuất ONNX opset 17, kích thước cố định, FP32, `dynamic=False`, `simplify=False`, `nms=False`. Graph trả raw head nên NMS/giải mã nằm trong Kotlin; output chưa phải danh sách box cuối cùng. Model và manifest được stage vào `app/android/app/src/main/assets/ocr/` sau kiểm chứng.

[Gradle](../app/android/app/build.gradle) kiểm tra size/hash và chặn asset cũ hoặc tensor debug trước build. Native kiểm tra cấu hình, chép ONNX từ assets vào cache nếu cần, xác minh hash rồi mở session. Cache là bản sao có thể tái tạo. Không sửa `.pt` hoặc thay riêng một ONNX mà bỏ qua manifest và parity.

## Luồng xử lý một ảnh

Luồng chính: **chọn ảnh đầy đủ → kiểm tra URI/EXIF → tìm OBB → crop chỉnh thẳng → check ngược → đọc chữ số → JSON → overlay/kết quả**.

1. **Chọn và kiểm tra ảnh.** [MeterScreen](../app/src/screens/MeterScreen.tsx) dùng picker với `cropping: false`, gọi `validateImage` rồi tự đọc ảnh. [MeterOcrModule](../app/android/app/src/main/java/com/watermeterocr/ocr/MeterOcrModule.kt) hỗ trợ đường dẫn/file URI và content URI đọc được cục bộ; copy có giới hạn vào tệp tạm, kiểm tra tối đa 25 MiB, 12 MP, mỗi chiều 16.384 px. Bitmap được xoay/lật theo EXIF trước suy luận. Ảnh không truy cập được offline hoặc không giải mã được là lỗi đầu vào.
2. **Chuẩn hóa tensor.** [Engine](../app/android/app/src/main/java/com/watermeterocr/ocr/SuppliedYoloEngine.kt) lấy RGB, resize giữ tỉ lệ và letterbox với padding 114; đóng tensor FP32 NCHW, chia 255. Mỗi stage dùng kích thước riêng trong manifest. Khi port cần giữ đúng kênh màu, resize, làm tròn và padding.
3. **Tìm và nắn khung.** [Decoder](../app/android/app/src/main/java/com/watermeterocr/ocr/SuppliedYoloDecoder.kt) giải mã OBB và ProbIoU NMS. Không có khung hoặc có nhiều khung trả `uncertain`; chỉ một khung mới đi tiếp. Các đỉnh được ánh xạ về ảnh sau EXIF, sắp cạnh dài nằm ngang và giữ chiều để không lật gương. OpenCV `warpPerspective` dùng `INTER_LINEAR`, `BORDER_REPLICATE`, không thêm margin; hình học không dùng được bị từ chối.
4. **Xác định hướng.** `reverse.onnx` chạy trên crop đã nắn. Nhãn bất đối xứng thuộc `2n,3n,4n,5n,7n` hoặc ít nhất hai detection kích hoạt xoay 180°. Một `1n` đơn lẻ không kích hoạt vì dễ xuất hiện trên số 1 xuôi. Stage chữ số nhận crop sau quyết định này; không đảo chuỗi ở cuối.
5. **Đọc và ghép số.** `digits.onnx` dùng class-agnostic NMS để loại box chồng giữa các lớp. Detection được sắp trái sang phải; bảng `classToDigit` ánh xạ lớp 0–29, hiện tương đương `int(label) % 10` (lớp 13 → `3`). Profile nhận 5 hoặc 6 chữ số, giữ kết quả dạng chuỗi, không tự thêm dấu thập phân. Mapping là hợp đồng riêng của bộ model này, không phải quy tắc chung của YOLO.

Ngưỡng confidence/IoU, giới hạn hình học và quy tắc nhãn được sở hữu bởi manifest và `SuppliedYoloDecoder.validateConfig`; tài liệu không thay thế cấu hình đó.

## Bridge và vòng đời native

[NativeMeterOcr.ts](../app/src/specs/NativeMeterOcr.ts) định nghĩa các lời gọi bất đồng bộ trả Promise chứa JSON string:

- `validateImage`: kiểm tra ảnh trước khi nhận vào luồng UI.
- `runImageDiagnostic`: đường đọc ảnh thông thường; tên “diagnostic” vẫn trả kết quả nhận diện thật.
- `runRuntimeSmoke`: kiểm tra ba session bằng input tổng hợp, không phải chỉ số đồng hồ.
- `runImageParity`: đường Debug xuất tensor/metadata; Release từ chối bằng `E_PARITY_DISABLED`.

Bitmap và tensor giữ ở native; bridge truyền URI và JSON để hạn chế sao chép dữ liệu ảnh giữa JS và Android.

[MeterOcrPackage](../app/android/app/src/main/java/com/watermeterocr/ocr/MeterOcrPackage.kt) đăng ký module trong [MainApplication](../app/android/app/src/main/java/com/watermeterocr/MainApplication.kt). Khi đổi tên module hoặc hợp đồng, phải giữ đồng bộ spec TypeScript, cấu hình Codegen, lớp Kotlin và đăng ký package. [getNativeOcr](../app/src/ocr/native.ts) là điểm lấy module phía JS.

`MeterOcrModule` sở hữu một serial executor và engine được nạp khi cần, tái sử dụng ba session giữa các lượt đọc. Một tác vụ hoạt động tại một thời điểm; yêu cầu trùng bị từ chối `E_BUSY`, không xếp hàng vô hạn. `invalidate()` ngăn việc mới, để công việc đang chạy kết thúc trước khi đóng session, rồi shutdown executor; Promise đang chờ nhận `E_MODULE_CLOSED`.

Serial áp dụng ở cấp yêu cầu; số luồng tính toán nội bộ ONNX Runtime được cấu hình riêng trong `SuppliedYoloEngine`.

Bitmap, tệp tạm, OpenCV `Mat`, tensor và kết quả ONNX có phạm vi giải phóng rõ ràng. Các quyết định này tránh giữ bộ nhớ ảnh lớn, dùng session đã đóng hoặc chặn JS/UI thread. [DiagnosticController](../app/src/ocr/controller.ts) dùng generation để bỏ kết quả cũ; `MeterScreen` giữ quyền sở hữu ảnh tới khi native hoàn tất mới dọn tệp picker.

## Kết quả trả UI

Hợp đồng được tạo bởi `SuppliedYoloEngine.run`, bọc thời gian bởi `MeterOcrModule.runInference`, và tiêu thụ tại [readingView](../app/src/ocr/result.ts).

| Nhóm | Ý nghĩa cho bên tích hợp |
|---|---|
| `pipelineId`, `semanticsVersion` | Nhận diện pipeline và ngữ nghĩa decoder; không ghép kết quả của các bundle khác nhau. |
| `status`, `rawDigits`, `reason` | `ok` mang chuỗi chữ số hợp lệ theo profile; `uncertain` mang `rawDigits=null` cùng lý do như `no_row`, `multiple_rows`, `no_characters`, `unexpected_length`. Lỗi URI/runtime dùng Promise rejection, không giả thành kết quả đọc. |
| `rowBox`, `imageSize` | Box `[left,top,right,bottom]` theo pixel của ảnh sau EXIF, không theo tensor letterbox/crop. UI scale và bù khoảng trống của `resizeMode=contain` để vẽ đúng overlay. |
| `isReversed`, `confidence`, timing | Hướng, điểm tổng hợp và thời gian; confidence không phải xác suất chuỗi chắc chắn đúng. `totalMs` bao gồm đọc ảnh và nạp engine khi cần; `invokeMs` là phần pipeline trong engine, gồm tiền/hậu xử lý. |

Giữ `rawDigits` là string để bảo toàn ký tự `0` **đã nhận diện**. Điều này không khôi phục ký tự detector bỏ sót; nếu khung cắt thiếu một số nhưng độ dài còn hợp lệ, kết quả vẫn có thể là `ok`. Không tự pad số 0 để che lỗi model.

## Kiểm chứng khi thay model hoặc port native

Ba điều kiện cần đánh giá riêng:

1. **Đúng artifact:** checkpoint, ONNX và manifest khớp hash; xem [exporter](../tools/ocr/export_supplied_yolo.py) và task kiểm tra assets trong Gradle.
2. **Đúng pipeline:** [host reference](../tools/ocr/supplied_yolo_reference.py) đối chiếu PyTorch/ONNX; [Android gates](../app/android/app/src/androidTest/java/com/watermeterocr/ocr/PhaseThreeGatesTest.kt) đối chiếu ảnh sau EXIF, tensor, OBB, hướng và chuỗi. Đọc dung sai từng stage trong test: tensor sau crop có thể khác do số học tọa độ và nội suy. Test đạt trong dung sai không chứng minh tensor khớp tuyệt đối; nới dung sai không sửa nguyên nhân sai lệch.
3. **Đúng chỉ số và trải nghiệm:** fixture có nhãn chuẩn, ảnh ngoài fixture, picker/overlay Release offline và điện thoại thật. `status=ok` chỉ nói qua điều kiện decoder, không thay thế đối chiếu chỉ số.

Chẩn đoán theo ranh giới: so hash/pixel ảnh và `frame_input` trước; sau đó OBB/crop, input hai detector, nhãn/hướng/chuỗi; cuối cùng JSON và overlay. Cách này phân biệt lỗi export, preprocessing, model và bridge/UI mà không thay đồng thời nhiều thành phần.

Lệnh export/host ở [tools/ocr/README.md](../tools/ocr/README.md); build/test Android ở [app/README.md](../app/README.md). Kết quả chạy, mẫu crop mất số 0 đầu và phần nghiệm thu còn chờ nằm trong [báo cáo kiểm chứng](../plans/261016-0741-supplied-yolo-mobile-integration/reports/test-261020-latest-emulator-verification.md), không phải cam kết độ chính xác của kiến trúc.
