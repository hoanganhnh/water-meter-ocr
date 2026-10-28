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
  private val cadence = AnalysisCadence()
  private val scanLock = Any()
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
  @Volatile private var destroyed = false
  @Volatile private var cameraBound = false
  private var captureOffsetNs: Long? = null // Worker owned; camera timebase need not equal elapsed realtime.
  private var lastSubmittedCaptureNs = 0L
  @Volatile private var scanStartedMs = 0L
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
    }
    preview.controller = camera
    // Start preview immediately, but attach analysis only after preload: an old first image
    // must not wait in the executor behind model initialization and establish a stale baseline.
    // Run analyzer directly on worker so RGBA conversion stays off the UI thread and CameraX's
    // STRATEGY_KEEP_ONLY_LATEST lease stays held until analyze() closes the ImageProxy.
    worker.execute {
      if (!destroyed) {
        try { readyEngine() }
        catch (error: Exception) { android.util.Log.w("RealtimeMeter", "Engine initialization failed", error) }
        finally {
          runOnUiThread {
            if (!destroyed) camera.setImageAnalysisAnalyzer(worker, ::analyze)
          }
        }
      }
    }
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
      text = "Đưa gần toàn bộ dãy số, giữ đúng chiều và tránh lóa. Chạm ảnh để lấy nét."
      textSize = 17f; setTextColor(Color.WHITE); setPadding(8, 12, 8, 12)
      minLines = 2; maxLines = 2
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
        } else updateStatus("Thiết bị không có đèn pin khả dụng.")
      }
    }, LinearLayout.LayoutParams(0, -2, 1f))
    actions.addView(Button(this).apply { text = "Đóng"; setOnClickListener { finish() } },
      LinearLayout.LayoutParams(0, -2, 1f))
    root.addView(actions)
    setContentView(root)
  }

  private fun updateStatus(message: String) {
    if (status.text.toString() != message) status.text = message
  }

  private fun bindCamera() {
    if (cameraBound) return
    try {
      camera.bindToLifecycle(this)
      cameraBound = true
    } catch (error: Exception) {
      updateStatus("Không mở được camera. Bạn có thể đóng để chụp hoặc chọn ảnh.")
    }
  }

  override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (requestCode == 40) {
      if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) bindCamera()
      else updateStatus("Chưa có quyền camera. Cho phép trong Cài đặt → Ứng dụng → WaterMeterOCR → Quyền; hoặc đóng để chọn ảnh.")
    }
  }

  private fun resetScan() {
    synchronized(scanLock) { generation++; frozen = false; scanStartedMs = SystemClock.elapsedRealtime() }
    // The worker clears its own evidence on generation change before cadence checks.
    frozenImage.setImageDrawable(null); frozenImage.visibility = View.GONE
    frozenBitmap?.recycle(); frozenBitmap = null; frozenResult = null
    confirm.isEnabled = false; overlay.box = null; overlay.frozenSize = null; overlay.invalidate()
    updateStatus("Đưa gần toàn bộ dãy số, giữ đúng chiều và tránh lóa. Chạm ảnh để lấy nét.")
  }

  override fun onStart() { super.onStart(); active = true; resetScan() }
  override fun onStop() {
    active = false; resetScan()
    if (::camera.isInitialized) { camera.enableTorch(false); torch = false }
    super.onStop()
  }

  private fun readyEngine(): SuppliedYoloEngine {
    engine?.let { return it }
    val started = SystemClock.elapsedRealtime()
    return SuppliedYoloEngine(applicationContext).also {
      engine = it
      if (com.watermeterocr.BuildConfig.DEBUG)
        android.util.Log.d("RealtimeMeter", "engineReadyMs=${SystemClock.elapsedRealtime() - started}")
    }
  }

  /** Admission runs on main so its generation cannot race with reset or executor shutdown. */
  private fun admitFrame(image: ImageProxy) {
    val token = generation
    val admittedAt = SystemClock.elapsedRealtime()
    if (!active || destroyed || frozen) { image.close(); return }
    try { worker.execute { analyze(image, token, admittedAt) } }
    catch (_: java.util.concurrent.RejectedExecutionException) { image.close() }
  }

  // Also used by the labelled-frame instrumentation path already running on the worker.
  private fun analyze(image: ImageProxy) = analyze(image, generation, SystemClock.elapsedRealtime())

  private fun analyze(image: ImageProxy, token: Int, admittedAt: Long) {
    var bitmap: Bitmap? = null
    var started = -1L
    var reserved = false
    try {
      if (!active || destroyed || frozen || token != generation) return
      val now = SystemClock.elapsedRealtime()
      if (workerGeneration != token) {
        stability.reset(); cadence.reset(); captureOffsetNs = null; lastSubmittedCaptureNs = 0L
        workerGeneration = token
      }
      if (now - admittedAt > 3000) { stability.reset(); return }
      if (!cadence.shouldAnalyze(now)) return
      val capturedAtNs = image.imageInfo.timestamp
      if (capturedAtNs <= 0 || capturedAtNs < lastSubmittedCaptureNs) { stability.reset(); return }
      if (capturedAtNs == lastSubmittedCaptureNs) return
      lastSubmittedCaptureNs = capturedAtNs
      // Compare relative clock offsets, not absolute timestamps from potentially different timebases.
      // The first delivered frame establishes the baseline; CameraX bounds its pending queue.
      val offset = SystemClock.elapsedRealtimeNanos() - capturedAtNs
      val baseline = minOf(captureOffsetNs ?: offset, offset)
      captureOffsetNs = baseline
      if (offset - baseline > 3_000_000_000L) { stability.reset(); return }
      started = now
      val sourceTransform = ImageProxyTransformFactory().apply {
        isUsingRotationDegrees = true
      }.getOutputTransform(image)
      val raw = image.toBitmap()
      bitmap = raw
      val rotated = if (image.imageInfo.rotationDegrees == 0) raw else Bitmap.createBitmap(
        raw, 0, 0, raw.width, raw.height,
        Matrix().apply { postRotate(image.imageInfo.rotationDegrees.toFloat()) }, true)
      if (rotated !== raw) raw.recycle()
      bitmap = rotated
      val convertedAt = SystemClock.elapsedRealtime()
      val evidence = readyEngine().runRealtime(rotated)
      val inferredAt = SystemClock.elapsedRealtime()
      if (!active || token != generation || destroyed) return
      val row = evidence.cropPlan.rowBox
      val box = row?.let { RectF(it[0].toFloat(), it[1].toFloat(), it[2].toFloat(), it[3].toFloat()) }
      val observation = ReadingQuality.evaluate(evidence.reading, evidence.cropPlan,
        evidence.frameDetections, evidence.width, evidence.height)
      // Include the current slow inference in the valid-pair window, rather than resetting forever.
      val duration = inferredAt - now
      val pairWindow = maxOf(cadence.pairWindowMs, (3 * duration).coerceIn(3000, 10000))
      val stable = stability.observe(observation, capturedAtNs, inferredAt, pairWindow)
      if (stable) {
        reserved = synchronized(scanLock) {
          if (active && token == generation && !destroyed && !frozen) { frozen = true; true }
          else false
        }
        if (!reserved) return
        val result = evidence.report().put("source", "realtime")
          .put("totalMs", SystemClock.elapsedRealtime() - now)
        val reportedAt = SystemClock.elapsedRealtime()
        if (com.watermeterocr.BuildConfig.DEBUG) android.util.Log.d("RealtimeMeter",
          "conversionMs=${convertedAt-now} inferenceMs=${inferredAt-convertedAt} " +
            "reportMs=${reportedAt-inferredAt} frameMs=${reportedAt-now} " +
            "scanToFreezeMs=${reportedAt-scanStartedMs} " +
            "credibleToFreezeMs=${reportedAt-(stability.firstCredibleAtMs ?: inferredAt)}")
        // Bitmap ownership transfers only after report construction and successful UI scheduling.
        runOnUiThread { showFrozenReading(token, rotated, result, box) }
        bitmap = null
      } else {
        val digits = evidence.reading.rawDigits
        val pending = stability.pendingDigits
        val reason = evidence.reading.reason
        val tooSmall = box != null && (evidence.cropPlan.cropWidth < 130 || evidence.cropPlan.cropHeight < 24)
        runOnUiThread {
          if (!active || token != generation || destroyed || isFinishing || frozen) return@runOnUiThread
          overlay.box = null; overlay.frozenSize = null
          val target = preview.outputTransform
          if (box != null && target != null) {
            try { CoordinateTransform(sourceTransform, target).mapRect(box); overlay.box = box }
            catch (_: IllegalArgumentException) { /* Wait for matching preview geometry. */ }
          }
          updateStatus(when {
            tooSmall -> "Đồng hồ còn xa · đưa điện thoại lại gần dãy số hơn."
            pending != null -> "Đang xác nhận: $pending · giữ yên máy để lấy nét rõ…"
            digits != null -> "Đang kiểm tra: $digits · giữ rõ toàn bộ 6 chữ số…"
            reason == "multiple_rows" -> "Có nhiều đồng hồ. Đưa gần một dãy số."
            box != null -> "Đã thấy vùng số. Giữ yên, tránh lóa và lấy nét đủ chữ số."
            else -> "Đang tìm vùng số. Đưa gần đồng hồ và tránh lóa."
          })
          overlay.invalidate()
        }
      }
    } catch (error: Exception) {
      if (reserved) synchronized(scanLock) { if (token == generation) frozen = false }
      android.util.Log.w("RealtimeMeter", "Frame analysis failed", error)
      stability.reset()
      runOnUiThread {
        if (active && token == generation && !destroyed && !frozen) {
          overlay.box = null; overlay.invalidate()
          updateStatus("Chưa xử lý được camera. Bấm Quét lại hoặc đóng để chọn ảnh.")
        }
      }
    } finally {
      if (started >= 0 && workerGeneration == token) {
        val duration = SystemClock.elapsedRealtime() - started
        cadence.recordDuration(duration)
        if (com.watermeterocr.BuildConfig.DEBUG)
          android.util.Log.d("RealtimeMeter", "analysisMs=$duration nextIntervalMs=${cadence.intervalMs}")
      }
      bitmap?.recycle(); image.close()
    }
  }

  /** Main-thread handoff also owns disposal of rejected evidence. */
  private fun showFrozenReading(token: Int, bitmap: Bitmap, result: JSONObject, box: RectF?): Boolean {
    synchronized(scanLock) {
      if (!active || token != generation || destroyed || isFinishing || frozenBitmap != null) {
        bitmap.recycle(); return false
      }
      frozen = true
      frozenBitmap = bitmap; frozenResult = result
    }
    frozenImage.setImageBitmap(bitmap); frozenImage.visibility = View.VISIBLE
    overlay.frozenSize = Size(bitmap.width, bitmap.height); overlay.box = box
    confirm.isEnabled = true
    updateStatus("Đã giữ ảnh: ${result.optString("rawDigits")}\nĐối chiếu chỉ số với ảnh trước khi xác nhận.")
    overlay.invalidate()
    return true
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
    } catch (_: Exception) { file?.delete(); updateStatus("Không lưu được ảnh. Hãy thử xác nhận lại.") }
  }

  override fun onDestroy() {
    synchronized(scanLock) { active = false; destroyed = true; cameraBound = false; generation++; frozen = false }
    if (::camera.isInitialized) { camera.clearImageAnalysisAnalyzer(); camera.unbind() }
    frozenImage.setImageDrawable(null); frozenBitmap?.recycle(); frozenBitmap = null
    try {
      worker.execute {
        try { engine?.close(); engine = null }
        finally { worker.shutdown() }
      }
    } catch (_: java.util.concurrent.RejectedExecutionException) {
      worker.shutdown()
    }
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
