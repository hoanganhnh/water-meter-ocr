# Water Meter OCR — React Native & Android App

Ứng dụng Android đóng gói offline pipeline 3-stage YOLOv8 (`supplied-yolo-3stage-v1`) sử dụng **ONNX Runtime Android `1.20.0`** và **OpenCV `4.12.0`**. Đường chạy native hiện hành là:

`MeterOcrModule` -> `SuppliedYoloEngine` -> `SuppliedYoloDecoder`

Các engine và model thử nghiệm trước đây (`YoloPpocrEngine`, `RowDecoder`, `RowPreprocess`, `AutoRowLocator`, `SwiftYoloDecoder`, `row-detector.onnx`, `row-recognizer.onnx`, `diagnostic-fp32.tflite`) đã được gỡ bỏ hoàn toàn khỏi source tree và APK. iOS vẫn ở trạng thái scaffold-only.

## Hành vi nghiệp vụ & giao diện

1. Người dùng chọn một ảnh đồng hồ nước có sẵn trên máy (JPEG/JFIF/PNG/WebP).
2. `MeterOcrModule` kiểm tra giới hạn an toàn (`<= 25 MiB`, `<= 12 MP`, mỗi chiều `<= 16,384 px`), giải mã ảnh và chuẩn hóa hướng theo EXIF (`1..8`) trên serial executor.
3. `SuppliedYoloEngine` chạy tuần tự 3 phiên ONNX Runtime offline:
   - **`frame.onnx` (`640x640` OBB):** Tìm khung số `khung`. Nếu không có khung (`no_row`) hoặc có từ 2 khung trở lên (`multiple_rows`), trả về `status = "uncertain"`. Nếu có đúng 1 khung hợp lệ, sắp xếp 4 đỉnh OBB và nắn thẳng (`warpPerspective`) vùng mặt số.
   - **`reverse.onnx` (`320x320` Detect):** Kiểm tra ảnh crop có bị ngược 180° hay không (`2n,3n,4n,5n,7n` hoặc `>= 2` box ngược). Nếu bị ngược (`isReversed = true`), xoay ảnh crop 180°.
   - **`digits.onnx` (`320x320` Detect):** Nhận diện chữ số trên ảnh crop đã định hướng đúng, ánh xạ 30 lớp `0..29` về chữ số `0..9` theo `int(label) % 10`, ghép chuỗi từ trái sang phải (giữ nguyên số `0` ở đầu) và kiểm tra độ dài hợp lệ (`5` hoặc `6` chữ số).
4. Giao diện `MeterScreen` hiển thị khung vùng đọc (`rowBox`) trên ảnh và chỉ số đọc được (`rawDigits`) hoặc lý do từ chối khi `status = "uncertain"`.

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
