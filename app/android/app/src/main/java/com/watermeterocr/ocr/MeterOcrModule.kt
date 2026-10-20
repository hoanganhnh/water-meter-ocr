package com.watermeterocr.ocr

import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.annotations.ReactModule
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.content.pm.ApplicationInfo
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import com.watermeterocr.NativeMeterOcrSpec
import java.io.File
import java.io.FileOutputStream
import java.io.FileInputStream
import java.io.InputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONObject

@ReactModule(name = MeterOcrModule.NAME)
class MeterOcrModule(context: ReactApplicationContext) : NativeMeterOcrSpec(context) {
  companion object {
    const val NAME = "NativeMeterOcr"
    private const val MAX_SOURCE_BYTES = 25L * 1024L * 1024L
    private const val MAX_SOURCE_PIXELS = 12_000_000L
    private const val MAX_SOURCE_DIMENSION = 16_384
  }

  private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor()
  private val lifecycleLock = Any()
  private val inferenceActive = AtomicBoolean(false)
  private var engine: SuppliedYoloEngine? = null
  @Volatile private var invalidated = false

  override fun getName() = NAME

  override fun runRuntimeSmoke(promise: Promise) {
    enqueue(promise) { runInference(null) }
  }

  override fun runImageDiagnostic(uri: String, promise: Promise) {
    if (uri.isBlank()) {
      promise.reject("E_IMAGE_URI", "Image URI is empty")
      return
    }
    enqueue(promise) { runInference(uri) }
  }

  override fun runImageParity(uri: String, promise: Promise) {
    if ((reactApplicationContext.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) == 0) {
      promise.reject("E_PARITY_DISABLED", "Parity tensor export is available only in debug builds")
      return
    }
    if (uri.isBlank()) {
      promise.reject("E_IMAGE_URI", "Image URI is empty")
      return
    }
    enqueue(promise) { runInference(uri, true) }
  }

  override fun validateImage(uri: String, promise: Promise) {
    if (uri.isBlank()) {
      promise.reject("E_IMAGE_URI", "Image URI is empty")
      return
    }
    enqueue(promise) { inspectImage(uri) }
  }

  private fun enqueue(promise: Promise, operation: () -> String) {
    synchronized(lifecycleLock) {
      if (invalidated) {
        promise.reject("E_MODULE_CLOSED", "OCR module is shutting down")
        return
      }
      if (!inferenceActive.compareAndSet(false, true)) {
        promise.reject("E_BUSY", "An OCR runtime check is already running")
        return
      }
      try {
        inferenceExecutor.execute {
          var result: String? = null
          var failure: Throwable? = null
          try {
            if (invalidated) throw IllegalStateException("OCR module is shutting down")
            result = operation()
          } catch (error: Throwable) {
            failure = error
          } finally {
            inferenceActive.set(false)
          }
          if (invalidated) promise.reject("E_MODULE_CLOSED", "OCR module is shutting down")
          else if (failure != null) promise.reject("E_INFERENCE", failure.message, failure)
          else promise.resolve(result)
        }
      } catch (error: Throwable) {
        inferenceActive.set(false)
        promise.reject("E_EXECUTOR_CLOSED", error.message, error)
      }
    }
  }

  private fun runInference(uri: String?, saveParity: Boolean = false): String {
    val started = System.nanoTime()
    if (invalidated) throw IllegalStateException("OCR module is shutting down")
    val runtime = engine ?: SuppliedYoloEngine(reactApplicationContext).also { engine = it }
    if (uri == null) {
      val bitmap = Bitmap.createBitmap(SuppliedYoloDecoder.FRAME_IMG_SIZE, SuppliedYoloDecoder.FRAME_IMG_SIZE, Bitmap.Config.ARGB_8888)
      try { return runtime.run(bitmap, true, false) } finally { bitmap.recycle() }
    }
    val temp = File.createTempFile("meter-crop-", ".img", reactApplicationContext.cacheDir)
    var decoded: Bitmap? = null
    var oriented: Bitmap? = null
    try {
      copyUriWithLimit(uri, temp)
      val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
      BitmapFactory.decodeFile(temp.absolutePath, bounds)
      require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Image file is corrupt or unsupported" }
      require(bounds.outWidth <= MAX_SOURCE_DIMENSION && bounds.outHeight <= MAX_SOURCE_DIMENSION) { "Image dimension exceeds the 16384 pixel limit" }
      require(bounds.outWidth.toLong()*bounds.outHeight <= MAX_SOURCE_PIXELS) { "Cropped image exceeds the 12 MP pixel limit" }
      decoded = BitmapFactory.decodeFile(temp.absolutePath, BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 })
        ?: throw IllegalArgumentException("Image could not be decoded")
      val exif = FileInputStream(temp).use { ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
      oriented = orient(decoded, exif)
      return JSONObject(runtime.run(oriented, false, saveParity))
        .put("totalMs", (System.nanoTime()-started)/1_000_000.0).toString()
    } finally {
      if (oriented !== decoded) oriented?.recycle()
      decoded?.recycle()
      temp.delete()
    }
  }

  private fun inspectImage(uriText: String): String {
    val temp = File.createTempFile("meter-inspect-", ".img", reactApplicationContext.cacheDir)
    try {
      copyUriWithLimit(uriText, temp)
      val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
      BitmapFactory.decodeFile(temp.absolutePath, bounds)
      require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Image file is corrupt or unsupported" }
      require(bounds.outWidth <= MAX_SOURCE_DIMENSION && bounds.outHeight <= MAX_SOURCE_DIMENSION) { "Image dimension exceeds the 16384 pixel limit" }
      val pixelCount = bounds.outWidth.toLong() * bounds.outHeight.toLong()
      require(pixelCount <= MAX_SOURCE_PIXELS) { "Image exceeds the 12 MP pixel limit; choose a smaller image" }
      val orientation = FileInputStream(temp).use { ExifInterface(it).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
      return "{\"width\":${bounds.outWidth},\"height\":${bounds.outHeight},\"bytes\":${temp.length()},\"exifOrientation\":$orientation}"
    } finally {
      temp.delete()
    }
  }

  private fun copyUriWithLimit(uriText: String, target: File) {
    val uri = Uri.parse(uriText)
    val stream = when (uri.scheme) {
      "file" -> FileInputStream(File(requireNotNull(uri.path)))
      "content" -> try {
        reactApplicationContext.contentResolver.openInputStream(uri)
      } catch (error: Exception) {
        throw IllegalArgumentException("Photo is unavailable offline or access expired; choose a locally available photo", error)
      }
      null, "" -> FileInputStream(File(uriText))
      else -> throw IllegalArgumentException("Unsupported image URI scheme: ${uri.scheme}")
    } ?: throw IllegalArgumentException("Image is unavailable offline; choose a locally available photo")
    stream.use { copyWithLimit(it, target) }
  }

  private fun copyWithLimit(input: InputStream, target: File) {
    var total = 0L
    FileOutputStream(target).use { output ->
      val buffer = ByteArray(64 * 1024)
      while (true) {
        val count = input.read(buffer)
        if (count < 0) break
        total += count
        require(total <= MAX_SOURCE_BYTES) { "Cropped image exceeds the 25 MB file limit" }
        output.write(buffer, 0, count)
      }
    }
  }

  private fun orient(source: Bitmap, orientation: Int): Bitmap {
    val matrix = Matrix()
    when (orientation) {
      ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.setScale(-1f, 1f)
      ExifInterface.ORIENTATION_ROTATE_180 -> matrix.setRotate(180f)
      ExifInterface.ORIENTATION_FLIP_VERTICAL -> { matrix.setRotate(180f); matrix.postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_TRANSPOSE -> { matrix.setRotate(90f); matrix.postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_ROTATE_90 -> matrix.setRotate(90f)
      ExifInterface.ORIENTATION_TRANSVERSE -> { matrix.setRotate(-90f); matrix.postScale(-1f, 1f) }
      ExifInterface.ORIENTATION_ROTATE_270 -> matrix.setRotate(-90f)
      else -> return source
    }
    return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
  }

  override fun invalidate() {
    synchronized(lifecycleLock) {
      if (invalidated) return
      invalidated = true
      inferenceExecutor.execute {
        try {
          engine?.close()
          engine = null
        } finally {
          inferenceExecutor.shutdown()
        }
      }
    }
  }
}
