package com.watermeterocr.ocr

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.os.Bundle
import android.os.SystemClock
import android.util.Size
import android.view.Gravity
import android.view.View
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.camera.view.transform.CoordinateTransform
import androidx.camera.view.transform.ImageProxyTransformFactory
import androidx.core.content.ContextCompat
import java.io.File
import java.util.concurrent.Executors
import org.json.JSONObject

/** Camera and pixels stay native. Only a user-confirmed reading crosses the RN bridge. */
class RealtimeMeterActivity : ComponentActivity() {
  private val worker = Executors.newSingleThreadExecutor()
  private val stability = ReadingStability()
  private var engine: SuppliedYoloEngine? = null // Access only on worker.
  private lateinit var camera: LifecycleCameraController
  private lateinit var preview: PreviewView
  private lateinit var frozenImage: ImageView
  private lateinit var overlay: RowOverlay
  private lateinit var status: TextView
  private lateinit var confirm: Button
  private lateinit var retry: Button
  @Volatile private var active = false
  @Volatile private var generation = 0
  @Volatile private var frozen = false
  private var nextAnalysisAt = 0L // Worker owned.
  private var workerGeneration = -1
  private var frozenBitmap: Bitmap? = null // Main thread owned.
  private var frozenResult: JSONObject? = null
  private var torch = false

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    buildUi()
    camera = LifecycleCameraController(this).apply {
      setEnabledUseCases(CameraController.IMAGE_ANALYSIS)
      imageAnalysisBackpressureStrategy = ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
      imageAnalysisOutputImageFormat = ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
      imageAnalysisTargetSize = CameraController.OutputSize(Size(1280, 960))
      setImageAnalysisAnalyzer(worker, ::analyze)
    }
    preview.controller = camera
    if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) bindCamera()
    else requestPermissions(arrayOf(Manifest.permission.CAMERA), 40)
  }

  private fun buildUi() {
    val root = LinearLayout(this).apply {
      orientation = LinearLayout.VERTICAL
      setBackgroundColor(Color.rgb(20, 39, 30))
      setPadding(18, 12, 18, 12)
    }
    // Respect status/navigation bars and display cutouts on edge-to-edge Android.
    if (android.os.Build.VERSION.SDK_INT >= 30) root.setOnApplyWindowInsetsListener { view, insets ->
      val bars = insets.getInsets(android.view.WindowInsets.Type.systemBars() or android.view.WindowInsets.Type.displayCutout())
      view.setPadding(18 + bars.left, 12 + bars.top, 18 + bars.right, 12 + bars.bottom)
      insets
    }
    else root.setOnApplyWindowInsetsListener { view, insets ->
      @Suppress("DEPRECATION")
      view.setPadding(18 + insets.systemWindowInsetLeft, 12 + insets.systemWindowInsetTop,
        18 + insets.systemWindowInsetRight, 12 + insets.systemWindowInsetBottom)
      insets
    }
    root.addView(TextView(this).apply {
      text = "Quét đồng hồ realtime"; textSize = 22f; setTextColor(Color.WHITE)
    })
    val frame = FrameLayout(this)
    preview = PreviewView(this).apply {
      implementationMode = PreviewView.ImplementationMode.COMPATIBLE
      scaleType = PreviewView.ScaleType.FIT_CENTER
    }
    frozenImage = ImageView(this).apply {
      scaleType = ImageView.ScaleType.FIT_CENTER; visibility = View.GONE; setBackgroundColor(Color.BLACK)
    }
    overlay = RowOverlay()
    for (view in listOf(preview, frozenImage, overlay)) frame.addView(view,
      FrameLayout.LayoutParams(-1, -1))
    root.addView(frame, LinearLayout.LayoutParams(-1, 0, 1f))
    status = TextView(this).apply {
      text = "Đưa gần toàn bộ dãy số, giữ yên khoảng 2–3 giây. Chạm ảnh để lấy nét."
      textSize = 17f; setTextColor(Color.WHITE); setPadding(8, 12, 8, 12)
      accessibilityLiveRegion = View.ACCESSIBILITY_LIVE_REGION_POLITE
    }
    root.addView(status)
    confirm = Button(this).apply {
      text = "Dùng chỉ số này"; isEnabled = false; setOnClickListener { acceptReading() }
    }
    root.addView(confirm)
    retry = Button(this).apply {
      text = "Quét lại"
      setOnClickListener {
        resetScan()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
          requestPermissions(arrayOf(Manifest.permission.CAMERA), 40)
        else bindCamera()
      }
    }
    root.addView(retry)
    val actions = LinearLayout(this).apply { gravity = Gravity.CENTER }
    actions.addView(Button(this).apply {
      text = "Đèn pin"
      setOnClickListener {
        if (::camera.isInitialized && camera.cameraInfo?.hasFlashUnit() == true) {
          torch = !torch; camera.enableTorch(torch)
        } else status.text = "Thiết bị không có đèn pin khả dụng."
      }
    }, LinearLayout.LayoutParams(0, -2, 1f))
    actions.addView(Button(this).apply { text = "Đóng"; setOnClickListener { finish() } },
      LinearLayout.LayoutParams(0, -2, 1f))
    root.addView(actions)
    setContentView(root)
  }

  private fun bindCamera() {
    try { camera.bindToLifecycle(this) }
    catch (error: Exception) {
      status.text = "Không mở được camera. Bạn có thể đóng để chụp hoặc chọn ảnh."
    }
  }

  override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (requestCode == 40) {
      if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) bindCamera()
      else status.text = "Chưa có quyền camera. Cho phép trong Cài đặt → Ứng dụng → WaterMeterOCR → Quyền; hoặc đóng để chọn ảnh."
    }
  }

  private fun resetScan() {
    generation++
    frozen = false
    stability.reset()
    frozenImage.setImageDrawable(null); frozenImage.visibility = View.GONE
    frozenBitmap?.recycle(); frozenBitmap = null; frozenResult = null
    confirm.isEnabled = false; overlay.box = null; overlay.invalidate()
    status.text = "Đưa gần toàn bộ dãy số, giữ yên khoảng 2–3 giây. Chạm ảnh để lấy nét."
  }

  override fun onStart() { super.onStart(); active = true; resetScan() }
  override fun onStop() {
    active = false; resetScan()
    if (::camera.isInitialized) { camera.enableTorch(false); torch = false }
    super.onStop()
  }

  private fun analyze(image: ImageProxy) {
    val token = generation
    var bitmap: Bitmap? = null
    try {
      val now = SystemClock.elapsedRealtime()
      if (!active || frozen || now < nextAnalysisAt) return
      nextAnalysisAt = now + 400
      if (workerGeneration != token) { stability.reset(); workerGeneration = token }
      val started = SystemClock.elapsedRealtime()
      val sourceTransform = ImageProxyTransformFactory().apply {
        isUsingRotationDegrees = true
      }.getOutputTransform(image)
      val raw = image.toBitmap()
      val rotated = if (image.imageInfo.rotationDegrees == 0) raw else Bitmap.createBitmap(
        raw, 0, 0, raw.width, raw.height,
        Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }, true)
      if (rotated !== raw) raw.recycle()
      bitmap = rotated
      // Keep the analysis lease until finally: CameraX overwrites its waiting
      // frame instead of queuing another analyzer call behind this inference.
      val runtime = engine ?: SuppliedYoloEngine(applicationContext).also { engine = it }
      val result = JSONObject(runtime.run(rotated, false, false))
        .put("source", "realtime").put("totalMs", SystemClock.elapsedRealtime() - started)
      if (!active || token != generation) return
      val row = result.optJSONArray("rowBox")
      val box = if (row != null && row.length() == 4) RectF(
        row.getDouble(0).toFloat(), row.getDouble(1).toFloat(),
        row.getDouble(2).toFloat(), row.getDouble(3).toFloat()) else null
      val digits = if (result.optString("status") == "ok") result.optString("rawDigits") else null
      val normalized = box?.let { floatArrayOf(it.left / rotated.width, it.top / rotated.height,
        it.right / rotated.width, it.bottom / rotated.height) }
      val stable = stability.observe(digits, normalized, SystemClock.elapsedRealtime())
      if (stable) {
        // Transfer exactly this evidence frame to the UI. It is never silently updated.
        bitmap = null
      }
      runOnUiThread {
        if (!active || token != generation || isFinishing || frozen) {
          if (stable) rotated.recycle()
          return@runOnUiThread
        }
        overlay.box = null
        if (stable) {
          frozen = true
          frozenBitmap = rotated; frozenResult = result
          frozenImage.setImageBitmap(rotated); frozenImage.visibility = View.VISIBLE
          overlay.frozenSize = Size(rotated.width, rotated.height); overlay.box = box
          confirm.isEnabled = true
          status.text = "Đã ổn định: $digits\nĐối chiếu chỉ số với ảnh trước khi xác nhận."
        } else {
          overlay.frozenSize = null
          val target = preview.outputTransform
          if (box != null && target != null) {
            try { CoordinateTransform(sourceTransform, target).mapRect(box); overlay.box = box }
            catch (_: IllegalArgumentException) { /* Wait for matching preview geometry. */ }
          }
          status.text = when {
            digits != null -> "Đang kiểm tra: $digits · giữ yên…"
            result.optString("reason") == "multiple_rows" -> "Có nhiều đồng hồ. Đưa gần một dãy số."
            box != null -> "Đã thấy vùng số. Giữ yên, tránh lóa và lấy nét lại."
            else -> "Đang tìm vùng số. Đưa gần đồng hồ và tránh lóa."
          }
        }
        overlay.invalidate()
      }
    } catch (error: Exception) {
      android.util.Log.w("RealtimeMeter", "Frame analysis failed", error)
      stability.reset()
      runOnUiThread {
        if (active && token == generation) {
          overlay.box = null; overlay.invalidate()
          status.text = "Chưa xử lý được camera. Bấm Quét lại hoặc đóng để chọn ảnh."
        }
      }
    } finally { bitmap?.recycle(); image.close() }
  }

  private fun acceptReading() {
    val bitmap = frozenBitmap ?: return
    val result = frozenResult ?: return
    var file: File? = null
    try {
      val savedFile = File.createTempFile("meter-realtime-", ".jpg", cacheDir).also { file = it }
      savedFile.outputStream().use { check(bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it)) }
      val payload = JSONObject().put("uri", android.net.Uri.fromFile(savedFile).toString())
        .put("width", bitmap.width).put("height", bitmap.height).put("size", savedFile.length())
        .put("result", result).toString()
      setResult(Activity.RESULT_OK, Intent().putExtra("reading", payload))
      finish()
    } catch (_: Exception) { file?.delete(); status.text = "Không lưu được ảnh. Hãy thử xác nhận lại." }
  }

  override fun onDestroy() {
    active = false; generation++
    if (::camera.isInitialized) { camera.clearImageAnalysisAnalyzer(); camera.unbind() }
    frozenImage.setImageDrawable(null); frozenBitmap?.recycle(); frozenBitmap = null
    worker.execute { engine?.close(); engine = null }
    worker.shutdown()
    super.onDestroy()
  }

  private inner class RowOverlay : View(this@RealtimeMeterActivity) {
    var box: RectF? = null
    var frozenSize: Size? = null
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
      color = Color.rgb(30, 230, 130); strokeWidth = 5f; style = Paint.Style.STROKE
    }
    override fun onDraw(canvas: Canvas) {
      super.onDraw(canvas)
      val rect = box?.let { RectF(it) } ?: return
      frozenSize?.let {
        val scale = minOf(width.toFloat() / it.width, height.toFloat() / it.height)
        rect.set((width-it.width*scale)/2 + rect.left*scale, (height-it.height*scale)/2 + rect.top*scale,
          (width-it.width*scale)/2 + rect.right*scale, (height-it.height*scale)/2 + rect.bottom*scale)
      }
      canvas.drawRect(rect, paint)
    }
  }
}
