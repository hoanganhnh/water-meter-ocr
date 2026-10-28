# Water Meter OCR — React Native & Android App

Ứng dụng Android đóng gói offline pipeline 3-stage YOLOv8 (`supplied-yolo-3stage-v1`) sử dụng **ONNX Runtime Android `1.20.0`** và **OpenCV `4.12.0`**. Đường chạy native hiện hành là:

`MeterOcrModule` -> `SuppliedYoloEngine` -> `SuppliedYoloDecoder`

Các engine và model thử nghiệm trước đây (`YoloPpocrEngine`, `RowDecoder`, `RowPreprocess`, `AutoRowLocator`, `SwiftYoloDecoder`, `row-detector.onnx`, `row-recognizer.onnx`, `diagnostic-fp32.tflite`) đã được gỡ bỏ hoàn toàn khỏi source tree và APK. iOS vẫn ở trạng thái scaffold-only.

## Cấu hình chiều ảnh Android

Mặc định `ocrReverseEnabled=false` trong `android/gradle.properties`. Cả ảnh tĩnh
và realtime bỏ qua `reverse.onnx`, không khởi tạo session reverse và không tự xoay
crop 180°. Giữ đồng hồ đúng chiều, dãy số nằm ngang. File model vẫn được đóng gói.
Kết quả có `reverseEnabled=false`; khi tìm được crop, `reverseTriggerReason` là
`disabled_by_config` và input/output reverse là null.

Để bật lại, đổi cấu hình thành `true` rồi build/cài lại, hoặc dùng
`gradlew.bat :app:assembleRelease -PocrReverseEnabled=true` từ `app/android`.
Manifest và công cụ Python vẫn mô tả pipeline ba bước gốc; đối chiếu parity ba bước
cần build với `-PocrReverseEnabled=true`. Tắt reverse cần được đánh giá lại trên ảnh
thật, chưa chứng minh đã sửa lỗi `000003 → 300000`.

## Hành vi nghiệp vụ & giao diện

1. Người dùng chọn ảnh có sẵn (JPEG/JFIF/PNG/WebP), **Chụp ảnh đồng hồ**, hoặc **Nhận diện realtime**. Hai đường ảnh tĩnh tự chạy OCR sau xác nhận ảnh; realtime dùng màn hình quét riêng mô tả bên dưới.
2. `MeterOcrModule` kiểm tra giới hạn an toàn (`<= 25 MiB`, `<= 12 MP`, mỗi chiều `<= 16,384 px`), giải mã ảnh và chuẩn hóa hướng theo EXIF (`1..8`) trên serial executor.
3. `SuppliedYoloEngine` chạy tuần tự 3 phiên ONNX Runtime offline:
   - **`frame.onnx` (`640x640` OBB):** Tìm khung số `khung`. Nếu không có khung (`no_row`) hoặc có từ 2 khung trở lên (`multiple_rows`), trả về `status = "uncertain"`. Nếu có đúng 1 khung hợp lệ, sắp xếp 4 đỉnh OBB và nắn thẳng (`warpPerspective`) vùng mặt số.
   - **`reverse.onnx` (`320x320` Detect):** Kiểm tra ảnh crop có bị ngược 180° hay không (`2n,3n,4n,5n,7n` hoặc `>= 2` box ngược). Nếu bị ngược (`isReversed = true`), xoay ảnh crop 180°.
   - **`digits.onnx` (`320x320` Detect):** Nhận diện chữ số trên ảnh crop đã định hướng đúng, ánh xạ 30 lớp `0..29` về chữ số `0..9` theo `int(label) % 10`, ghép chuỗi từ trái sang phải (giữ nguyên số `0` ở đầu) và kiểm tra độ dài hợp lệ (`5` hoặc `6` chữ số).
4. Giao diện `MeterScreen` hiển thị khung vùng đọc (`rowBox`) trên ảnh và chỉ số đọc được (`rawDigits`) hoặc lý do từ chối khi `status = "uncertain"`.

### Chụp ảnh trên Android

Camera dùng ứng dụng camera hệ thống qua `react-native-image-crop-picker.openCamera`, ưu tiên camera sau; không nhận diện video liên tục. Ảnh được thu nhỏ giữ tỷ lệ, tối đa `2560 × 2560`, JPEG quality `0.95`, giữ EXIF hướng rồi đi qua cùng bước kiểm tra/native YOLO như ảnh thư viện. Đây là giới hạn ảnh **đầu vào OCR**, camera hệ thống vẫn có thể chụp ảnh gốc lớn hơn.

- Lần đầu chọn **Chụp ảnh đồng hồ**, cho phép quyền camera. Nếu đã từ chối, bật lại trong Cài đặt → Ứng dụng → WaterMeterOCR → Quyền.
- Hủy chụp không chạy OCR. Thiết bị thiếu camera hoặc ứng dụng camera vẫn dùng được thư viện.
- Khi camera hoặc OCR đang hoạt động, cả hai nút nhận ảnh bị khóa; tệp ảnh trả về được dọn khi thay ảnh/rời màn hình, sau khi native đọc xong.
- Cần build/cài lại APK sau thay đổi quyền trong AndroidManifest; Fast Refresh chỉ cập nhật JavaScript.
- Emulator cần camera được cấu hình trong AVD. Cảnh camera ảo chỉ kiểm tra luồng chụp; đánh giá độ chính xác cần ảnh đồng hồ trên điện thoại thật.

### Nhận diện realtime trên Android

Bấm **Nhận diện realtime**, cấp quyền camera, đưa điện thoại lại gần mặt đồng hồ và giữ yên:
- **Chốt tự động theo chất lượng & độ ổn định ([`ReadingQuality.kt`](android/app/src/main/java/com/watermeterocr/ocr/ReadingQuality.kt), [`ReadingStability.kt`](android/app/src/main/java/com/watermeterocr/ocr/ReadingStability.kt)):**
  - Nhánh `HIGH` (mọi chữ số `>= 0.90`, trung bình `>= 0.95`, khung `>= 0.85`, không có lớp chuyển tiếp `>= 10` và hình học polygon gốc hợp lệ) tự giữ ngay ở frame đầu tiên khi không có lịch sử mâu thuẫn, không chờ cố định 800 ms hay 3 phiếu.
  - Nhánh `MODERATE` (chữ số `>= 0.65`, trung bình `>= 0.80`, khung `>= 0.60` hoặc chữ số cuối đang nhảy giữa hai mức) tự giữ khi đọc khớp **2 quan sát hợp lệ** trên cùng mục tiêu (IoU `>= 0.55` với vùng neo và `>= 0.65` với frame trước, trong cửa sổ thích ứng `3000..10000 ms`). Nếu có chuỗi hợp lệ mâu thuẫn trước đó, hệ thống yêu cầu xác nhận lại theo đa số/liên tiếp.
  - Nhánh `LOW` / `INVALID` không tự chốt; một frame `LOW` cùng vùng hợp lệ chỉ giữ ứng viên tối đa `1200 ms` qua nhịp mờ ngắn mà không cộng phiếu. Mất vùng, có nhiều đồng hồ, lệch vùng, timestamp camera trùng/lùi hoặc đổi `generation` (tạm dừng, **Quét lại**, đóng) đều xóa lịch sử cũ.
- **Giữ ảnh chủ động (`Giữ ảnh`) song song với tự động giữ:** Khi frame mới nhất đọc đủ 6 chữ số hợp lệ (`status = "ok"`, hình học không `INVALID`), nút **Giữ ảnh** sáng tối đa `1500 ms` để người dùng chủ động khóa ngay frame đó trước khi cổng tự động hoàn tất.
- **Khôi phục chữ số & chống đọc thiếu ([`SuppliedYoloDecoder.kt`](android/app/src/main/java/com/watermeterocr/ocr/SuppliedYoloDecoder.kt)):** Hỗ trợ gộp tín hiệu khi vòng quay cơ học nhảy dở giữa 2 số và khóa 6 chữ số khi đã thấy mặt đồng hồ 6 số (không tự điền số `0` đầu bị model bỏ sót).
- **Đường chạy nền nhẹ & báo cáo lười ([`RealtimeMeterActivity.kt`](android/app/src/main/java/com/watermeterocr/ocr/RealtimeMeterActivity.kt), [`AnalysisCadence.kt`](android/app/src/main/java/com/watermeterocr/ocr/AnalysisCadence.kt), [`SuppliedYoloEngine.kt`](android/app/src/main/java/com/watermeterocr/ocr/SuppliedYoloEngine.kt)):** Engine được nạp trước trên `worker` trước khi gắn `ImageAnalysis` (`STRATEGY_KEEP_ONLY_LATEST`). Nhịp phân tích thích ứng theo EWMA (`khởi đầu 200 ms`, `clamp(1.25 * EWMA, 150, 400) ms`, `alpha = 0.25`). Các frame không được giữ dùng `RealtimeFrame` nhẹ, không dựng JSON chi tiết hay băm SHA-256; chỉ frame được giữ mới dựng báo cáo đầy đủ một lần từ chính dữ liệu lượt suy luận đó (không chạy model lần hai).
- **Xác nhận thủ công bắt buộc:** Sau khi máy tự giữ hoặc bấm **Giữ ảnh**, người dùng đối chiếu ảnh minh chứng và bấm **Dùng chỉ số này** (lúc này mới ghi một file JPEG `meter-realtime-*.jpg` vào cache và trả `{uri, width, height, size, result}` với `source = "realtime"`) hoặc **Quét lại** / **Đóng**. Ngưỡng điểm hiện tại là cấu hình khởi tạo bảo thủ (điểm confidence không phải xác suất đúng tuyệt đối) và cần tiếp tục hiệu chỉnh trên bộ mẫu camera thực địa. Xem sơ đồ chi tiết tại [`docs/architecture.md`](../docs/architecture.md).

## Chạy local với emulator (Debug)

Mở thiết bị trong Android Studio → Device Manager, đợi vào màn hình chính. Dùng hai terminal PowerShell từ thư mục `app/`:

```powershell
# Terminal 1: giữ Metro chạy trong suốt phiên phát triển
npm.cmd start
```

```powershell
# Terminal 2: cấu hình công cụ, kết nối Metro và chạy app
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
$env:Path = "$env:JAVA_HOME\bin;$env:ANDROID_HOME\platform-tools;$env:ANDROID_HOME\emulator;$env:Path"
adb devices
adb -s emulator-5554 reverse tcp:8081 tcp:8081
npm.cmd run android -- --no-packager --active-arch-only
```

Thay `emulator-5554` nếu `adb devices` hiển thị serial khác. Nếu Metro đã chạy ở terminal hiện tại, giữ phiên đó thay vì mở thêm server trên cùng cổng.

Nếu gặp **Unable to load script** dù Metro đang chạy: mở React Native Dev Menu bằng `adb -s emulator-5554 shell input keyevent 82`, chọn **Change Bundle Location**, nhập **localhost:8081** rồi **APPLY CHANGES**. Địa chỉ này dùng cổng đã reverse qua ADB; mặc định `10.0.2.2:8081` có thể không truy cập được trên cấu hình máy hiện tại. Bản Debug cần kết nối Metro để tải JavaScript; suy luận YOLO vẫn chạy native. Bản Release đóng gói JS và model để chạy offline.

Để mở lại app đã cài:

```powershell
adb -s emulator-5554 reverse tcp:8081 tcp:8081
adb -s emulator-5554 shell am force-stop com.watermeterocr
adb -s emulator-5554 shell am start -n com.watermeterocr/.MainActivity
```

## APK gọn để cài test

Từ thư mục `app/`:

```powershell
npm.cmd run apk:phone     # ARM64: điện thoại Android 64-bit
npm.cmd run apk:emulator  # x86_64: emulator trên PC
npm.cmd run apk:armv7     # ARM 32-bit: thiết bị cũ
npm.cmd run apk:universal # Gộp cả 4 kiến trúc, chỉ dùng khi cần
```

APK được giữ riêng tại `app/android/app/build/outputs/apk-test/water-meter-ocr-<kiến-trúc>.apk`; bản điện thoại là `water-meter-ocr-arm64-v8a.apk`. Mỗi APK chứa JS và ba model nên chạy độc lập với Metro. Script dùng Gradle incremental, không `clean`, chỉ build kiến trúc được chọn; trên Windows tự tìm JDK Android Studio khi chưa đặt `JAVA_HOME`.

Các lệnh này bật `-PcompressNativeLibs=true`: thư viện `.so` được nén trong APK và Android giải nén khi cài. File APK nhỏ hơn nhưng dung lượng sau cài không giảm tương ứng. Build thông thường/App Bundle giữ cấu hình native không nén mặc định. Không đổi model, lượng tử hóa hoặc loại bỏ thư viện đang dùng.

APK universal trước tối ưu là **284,7 MiB**: native của 4 ABI chiếm **247,6 MiB**, ba model khoảng **30,4 MiB** trong APK. Gửi APK đúng ABI tránh mang theo thư viện của các CPU khác; ARM64 APK không cài được trên emulator x86_64. Có thể kiểm tra thiết bị bằng `adb shell getprop ro.product.cpu.abilist`.

Mốc sau tối ưu đóng gói, trước realtime: `apk:phone` **56,3 MiB**, giảm khoảng **80%**; native ARM64 sau nén khoảng **19,7 MiB**, hash của cả ba model không đổi. Bổ sung CameraX realtime đưa bản điện thoại lên khoảng **57 MiB**, emulator khoảng **71 MiB**. Đây là kích thước file APK, không phải dung lượng cài đặt hoặc bộ nhớ chạy app.

Bản `apk:emulator` đo được **70,5 MiB**, đã cài và chạy YOLO đọc fixture `085918` trên emulator x86_64. ARM64 đã kiểm tra chữ ký, ABI, bundle JS và hash model; chưa kiểm chứng trên điện thoại thật. Cơ chế giải nén thư viện khi cài được mô tả trong [tài liệu Android](https://developer.android.com/guide/practices/page-sizes).

Đây là bản Release ký bằng debug key của project để cài test; chưa phải chữ ký phát hành cửa hàng.

## Tái lập Assets và Build Android

Trước khi build Android, Gradle task `:app:verifySuppliedYoloAssets` tự động kiểm tra sự tồn tại, kích thước, SHA-256 của `frame.onnx`, `reverse.onnx`, `digits.onnx`, `supplied-yolo-manifest.json` và chặn mọi file model cũ hoặc debug tensor `.f32` trong `app/android/app/src/main/assets/ocr/`.

```powershell
# Từ thư mục gốc repository:
# 1. Stage bộ model ONNX đã xác minh vào Android assets
.\.cache\yolo-env\Scripts\python.exe -m tools.ocr.export_supplied_yolo `
  --out artifacts/supplied-yolo `
  --stage-android-assets app/android/app/src/main/assets/ocr `
  --stage-only

# 2. Stage bộ fixture cho Android Instrumentation Tests
.\.cache\yolo-env\Scripts\python.exe -m tools.ocr.supplied_yolo_reference `
  --bundle artifacts/supplied-yolo `
  --fixtures artifacts/supplied-yolo/fixtures/manifest.json `
  --out artifacts/supplied-yolo/host-report `
  --stage-gate-assets .cache/android-gates-assets `
  --stage-gate-only

# 3. Kiểm tra JS/TS
npm.cmd --prefix app run typecheck
npm.cmd --prefix app run lint
npm.cmd --prefix app test -- --runInBand

# 4. Chạy JVM Unit Tests và đóng gói APK Debug / Release (yêu cầu JDK 17+)
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
app\android\gradlew.bat -p app/android testDebugUnitTest assembleDebug assembleRelease

# 5. Chạy Instrumentation Tests trên thiết bị hoặc emulator đang kết nối
app\android\gradlew.bat -p app/android connectedDebugAndroidTest
```

## Bảo mật & Vòng đời Runtime

- Không phụ thuộc mạng hoặc Google Play Services; ứng dụng hoạt động offline ngay từ lần mở đầu tiên.
- `runImageParity` chỉ hoạt động ở bản Debug (`ApplicationInfo.FLAG_DEBUGGABLE`) để xuất các tensor `.f32` vào thư mục riêng của app (`files/ocr-parity/latest`) phục vụ kiểm thử; bản Release từ chối ngay với mã lỗi `E_PARITY_DISABLED`.
- Mọi tệp tạm (`meter-inspect-*`, `meter-crop-*`) trong `cacheDir` đều được xóa trong khối `finally`. Khi module bị `invalidate()`, yêu cầu đang chạy trên serial executor được hoàn tất an toàn trước khi đóng cả 3 `OrtSession`.
