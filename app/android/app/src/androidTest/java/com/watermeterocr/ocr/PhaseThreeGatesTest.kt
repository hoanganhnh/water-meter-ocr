package com.watermeterocr.ocr

import android.content.Context
import android.content.ContextWrapper
import android.content.pm.ApplicationInfo
import android.os.Build
import android.os.Debug
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.facebook.react.bridge.BridgeReactContext
import com.facebook.react.bridge.Promise
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.lang.reflect.Proxy
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.math.max

@RunWith(AndroidJUnit4::class)
class PhaseThreeGatesTest {
  companion object {
    private const val PIPELINE_ID = "supplied-yolo-3stage-v1"
    private const val SEMANTICS_VERSION = "supplied-yolo-3stage-v1"
    private const val PREPROCESS_VERSION = "opencv-linear-byte-letterbox-rgb-div255-pad114-obb-warp-v1"
    private const val ANDROID_BOX_PX_ATOL = 0.5
    private const val SCORE_ATOL = 0.005
    private const val CHAINED_SCORE_ATOL = 0.02
  }

  private val instrumentation = InstrumentationRegistry.getInstrumentation()
  private val context = instrumentation.targetContext
  private fun asset(name: String) = instrumentation.context.assets.open(name).use { it.readBytes() }
  private fun seed(name: String) = File(context.cacheDir, "gate-$name").also { it.writeBytes(asset(name)) }
  private fun module() = MeterOcrModule(BridgeReactContext(context))

  private class Reply {
    val done = CountDownLatch(1)
    var result: String? = null
    var error: String? = null
    var message: String? = null
    fun await(): Reply {
      assertTrue("Native promise must settle", done.await(60, TimeUnit.SECONDS))
      return this
    }
  }

  private fun call(operation: (Promise) -> Unit): Reply {
    val reply = Reply()
    val promise = Proxy.newProxyInstance(Promise::class.java.classLoader, arrayOf(Promise::class.java)) { _, method, args ->
      when (method.name) {
        "resolve" -> {
          reply.result = args?.get(0) as? String
          reply.done.countDown()
        }
        "reject" -> {
          reply.error = args?.get(0) as? String
          reply.message = args?.getOrNull(1)?.toString()
          reply.done.countDown()
        }
      }
      null
    } as Promise
    operation(promise)
    return reply
  }

  private fun assertTensor(expected: ByteArray, actual: ByteArray, atol: Double, rtol: Double, label: String = "tensor") {
    assertEquals("$label byte size", expected.size, actual.size)
    val h = ByteBuffer.wrap(expected).order(ByteOrder.LITTLE_ENDIAN)
    val a = ByteBuffer.wrap(actual).order(ByteOrder.LITTLE_ENDIAN)
    var index = 0
    while (h.hasRemaining()) {
      val reference = h.float.toDouble()
      val value = a.float.toDouble()
      assertTrue(
        "$label[$index] expected=$reference actual=$value",
        value.isFinite() && abs(value - reference) <= atol + rtol * abs(reference),
      )
      index++
    }
  }

  private fun assertYoloOutputTensor(
    expected: ByteArray,
    actual: ByteArray,
    channels: Int,
    anchors: Int,
    coordAtol: Double,
    scoreAtol: Double,
    rtol: Double,
    label: String,
  ) {
    assertEquals("$label byte size", channels * anchors * 4, expected.size)
    assertEquals("$label byte size", expected.size, actual.size)
    val h = ByteBuffer.wrap(expected).order(ByteOrder.LITTLE_ENDIAN)
    val a = ByteBuffer.wrap(actual).order(ByteOrder.LITTLE_ENDIAN)
    for (c in 0 until channels) {
      val atol = if (c < 4) coordAtol else scoreAtol
      for (anchor in 0 until anchors) {
        val reference = h.float.toDouble()
        val value = a.float.toDouble()
        assertTrue(
          "$label[c=$c,a=$anchor] expected=$reference actual=$value",
          value.isFinite() && abs(value - reference) <= atol + rtol * abs(reference),
        )
      }
    }
  }

  private fun compareChainedTensorStats(
    expected: ByteArray,
    actual: ByteArray,
    maxAtol: Double,
    meanAtol: Double,
    label: String,
  ): JSONObject {
    assertEquals("$label byte size", expected.size, actual.size)
    val count = expected.size / 4
    val h = ByteBuffer.wrap(expected).order(ByteOrder.LITTLE_ENDIAN)
    val a = ByteBuffer.wrap(actual).order(ByteOrder.LITTLE_ENDIAN)
    var maxDiff = 0.0
    var sumDiff = 0.0
    for (i in 0 until count) {
      val reference = h.float.toDouble()
      val value = a.float.toDouble()
      assertTrue("$label[$i] finite", value.isFinite())
      val diff = abs(value - reference)
      maxDiff = max(maxDiff, diff)
      sumDiff += diff
    }
    val meanDiff = if (count > 0) sumDiff / count else 0.0
    assertTrue("$label maxDiff=$maxDiff > $maxAtol", maxDiff <= maxAtol)
    assertTrue("$label meanDiff=$meanDiff > $meanAtol", meanDiff <= meanAtol)
    return JSONObject()
      .put("maxAbsDiff", maxDiff)
      .put("meanAbsDiff", meanDiff)
      .put("maxAtol", maxAtol)
      .put("meanAtol", meanAtol)
  }

  private fun assertJsonStringArrayEquals(label: String, expected: JSONArray, actual: JSONArray) {
    assertEquals("$label length", expected.length(), actual.length())
    for (i in 0 until expected.length()) {
      assertEquals("$label[$i]", expected.getString(i), actual.getString(i))
    }
  }

  @Test fun runtimeSmokeVerifiesThreeOnnxSessionsAndManifestHashes() {
    val native = module()
    try {
      val reply = call { native.runRuntimeSmoke(it) }.await()
      assertNull(reply.message, reply.error)
      val actual = JSONObject(reply.result!!)
      val gatesManifest = JSONObject(String(asset("gates-manifest.json")))
      assertEquals(PIPELINE_ID, gatesManifest.getString("pipelineId"))
      assertEquals(SEMANTICS_VERSION, gatesManifest.getString("semanticsVersion"))

      assertEquals(PIPELINE_ID, actual.getString("pipelineId"))
      assertEquals(SEMANTICS_VERSION, actual.getString("semanticsVersion"))
      assertEquals(SEMANTICS_VERSION, actual.getString("decoderVersion"))
      assertEquals(PREPROCESS_VERSION, actual.getString("preprocessVersion"))
      assertEquals(gatesManifest.getString("manifestSha256"), actual.getString("modelManifestSha256"))
      assertEquals("[1,3,640,640]", actual.getJSONArray("inputShape").toString())
      assertEquals("[1,6,8400]", actual.getJSONArray("outputShape").toString())
      assertEquals("[1,3,320,320]", actual.getJSONArray("reverseInputShape").toString())
      assertEquals("[1,10,2100]", actual.getJSONArray("reverseOutputShape").toString())
      assertEquals("[1,3,320,320]", actual.getJSONArray("digitsInputShape").toString())
      assertEquals("[1,34,2100]", actual.getJSONArray("digitsOutputShape").toString())
      assertEquals(8400 * 6 + 2100 * 10 + 2100 * 34, actual.getInt("finiteOutputs"))
      assertEquals("diagnostic", actual.getString("status"))
      assertEquals("diagnostic", actual.getString("source"))
      assertEquals("unverified", actual.getString("authenticity"))
      File(context.filesDir, "runtime-smoke.native.json").writeText(actual.toString(2))
    } finally {
      native.invalidate()
    }
  }

  @Test fun allApprovedFixturesMatchHostTensorsObbPolygonCropDetectionsOrientationAndFinalJson() {
    val native = module()
    try {
      val gatesManifest = JSONObject(String(asset("gates-manifest.json")))
      assertEquals(PIPELINE_ID, gatesManifest.getString("pipelineId"))
      assertEquals(SEMANTICS_VERSION, gatesManifest.getString("semanticsVersion"))

      val fixturesManifest = JSONObject(String(asset("fixtures-manifest.json")))
      val fixtures = fixturesManifest.getJSONArray("fixtures")
      assertEquals(8, fixtures.length())

      for (idx in 0 until fixtures.length()) {
        val fx = fixtures.getJSONObject(idx)
        val fxId = fx.getString("id")
        val expectedSpec = fx.getJSONObject("expected")

        if (expectedSpec.optBoolean("promiseRejected", false)) {
          val corrupt = seed("$fxId.bin")
          val reply = call { native.runImageDiagnostic(corrupt.toURI().toString(), it) }.await()
          assertEquals(fxId, expectedSpec.getString("errorCode"), reply.error)
          corrupt.delete()
          continue
        }

        val image = seed("$fxId.png")
        val reply = call { native.runImageParity(image.toURI().toString(), it) }.await()
        assertNull("$fxId: ${reply.message}", reply.error)
        val actual = JSONObject(reply.result!!)
        val hostDetail = JSONObject(String(asset("$fxId.result.json")))
        val host = hostDetail.getJSONObject("onnxResult")
        val root = File(context.filesDir, "ocr-parity/latest")

        assertEquals(fxId, PIPELINE_ID, actual.getString("pipelineId"))
        assertEquals(fxId, SEMANTICS_VERSION, actual.getString("semanticsVersion"))
        assertEquals(fxId, SEMANTICS_VERSION, actual.getString("decoderVersion"))
        assertEquals(fxId, PREPROCESS_VERSION, actual.getString("preprocessVersion"))
        assertEquals(fxId, "gallery", actual.getString("source"))
        assertEquals(fxId, "unverified", actual.getString("authenticity"))
        assertEquals(fxId, "parity-image", actual.getString("inputSource"))
        assertEquals("$fxId imagePixelSha256", host.getString("imagePixelSha256"), actual.getString("imagePixelSha256"))

        // Exact string, status, orientation, and class sequence checks (zero tolerance)
        assertEquals("$fxId status", host.getString("status"), actual.getString("status"))
        assertEquals("$fxId rawDigits", host.opt("rawDigits")?.toString(), actual.opt("rawDigits")?.toString())
        assertEquals("$fxId reason", host.opt("reason")?.toString(), actual.opt("reason")?.toString())
        assertEquals("$fxId isReversed", host.getBoolean("isReversed"), actual.getBoolean("isReversed"))
        assertEquals("$fxId reverseTriggerReason", host.getString("reverseTriggerReason"), actual.getString("reverseTriggerReason"))
        assertEquals("$fxId frameCount", host.getInt("frameCount"), actual.getInt("frameCount"))
        assertJsonStringArrayEquals("$fxId revClasses", host.getJSONArray("revClasses"), actual.getJSONArray("revClasses"))
        assertJsonStringArrayEquals(
          "$fxId digitClassesLeftToRight",
          host.getJSONArray("digitClassesLeftToRight"),
          actual.getJSONArray("digitClassesLeftToRight"),
        )

        // Stage 1 tensor parity
        assertTensor(
          asset("$fxId.frame_input.f32"),
          File(root, "frame_input.f32").readBytes(),
          1e-6,
          0.0,
          "$fxId.frame_input",
        )
        assertYoloOutputTensor(
          asset("$fxId.frame_output.f32"),
          File(root, "frame_output.f32").readBytes(),
          channels = 6,
          anchors = 8400,
          coordAtol = 0.02,
          scoreAtol = 5e-4,
          rtol = 1e-4,
          label = "$fxId.frame_output",
        )

        // Frame OBB detections parity
        val hostFrames = host.getJSONArray("frameDetections")
        val actualFrames = actual.getJSONArray("frameDetections")
        assertEquals("$fxId frameDetections length", hostFrames.length(), actualFrames.length())
        for (i in 0 until hostFrames.length()) {
          val hf = hostFrames.getJSONObject(i)
          val af = actualFrames.getJSONObject(i)
          assertEquals("$fxId frame[$i] anchor", hf.getInt("anchor"), af.getInt("anchor"))
          assertEquals("$fxId frame[$i] classId", hf.getInt("classId"), af.getInt("classId"))
          assertEquals("$fxId frame[$i] class", hf.getString("class"), af.getString("class"))
          assertEquals("$fxId frame[$i] score", hf.getDouble("score"), af.getDouble("score"), SCORE_ATOL)
          val hPoly = hf.getJSONArray("polygon")
          val aPoly = af.getJSONArray("polygon")
          for (p in 0 until 4) {
            assertEquals(hPoly.getJSONArray(p).getDouble(0), aPoly.getJSONArray(p).getDouble(0), ANDROID_BOX_PX_ATOL)
            assertEquals(hPoly.getJSONArray(p).getDouble(1), aPoly.getJSONArray(p).getDouble(1), ANDROID_BOX_PX_ATOL)
          }
        }

        if (host.getInt("frameCount") == 1) {
          // Verify ordered OBB polygon, rowBox, and cropSize
          val hOrdered = host.getJSONArray("orderedPolygon")
          val aOrdered = actual.getJSONArray("orderedPolygon")
          for (p in 0 until 4) {
            assertEquals("$fxId orderedPolygon[$p].x", hOrdered.getJSONArray(p).getDouble(0), aOrdered.getJSONArray(p).getDouble(0), ANDROID_BOX_PX_ATOL)
            assertEquals("$fxId orderedPolygon[$p].y", hOrdered.getJSONArray(p).getDouble(1), aOrdered.getJSONArray(p).getDouble(1), ANDROID_BOX_PX_ATOL)
          }
          val hRowBox = host.getJSONArray("rowBox")
          val aRowBox = actual.getJSONArray("rowBox")
          for (c in 0..3) {
            assertTrue("$fxId rowBox[$c]", abs(hRowBox.getInt(c) - aRowBox.getInt(c)) <= 1)
          }
          val hCropSize = host.getJSONArray("cropSize")
          val aCropSize = actual.getJSONArray("cropSize")
          assertTrue("$fxId cropSize[0]", abs(hCropSize.getInt(0) - aCropSize.getInt(0)) <= 1)
          assertTrue("$fxId cropSize[1]", abs(hCropSize.getInt(1) - aCropSize.getInt(1)) <= 1)

          // Stage 2 & Stage 3 chained tensors parity (warped crop has <=0.01px sub-pixel shift from Stage 1 FP32 OBB)
          val revInStats = compareChainedTensorStats(
            asset("$fxId.reverse_input.f32"),
            File(root, "reverse_input.f32").readBytes(),
            maxAtol = 0.20,
            meanAtol = 0.005,
            label = "$fxId.reverse_input",
          )
          val revOutStats = compareChainedTensorStats(
            asset("$fxId.reverse_output.f32"),
            File(root, "reverse_output.f32").readBytes(),
            maxAtol = 8.0,
            meanAtol = 0.05,
            label = "$fxId.reverse_output",
          )
          val digInStats = compareChainedTensorStats(
            asset("$fxId.digits_input.f32"),
            File(root, "digits_input.f32").readBytes(),
            maxAtol = 0.20,
            meanAtol = 0.005,
            label = "$fxId.digits_input",
          )
          val digOutStats = compareChainedTensorStats(
            asset("$fxId.digits_output.f32"),
            File(root, "digits_output.f32").readBytes(),
            maxAtol = 8.0,
            meanAtol = 0.05,
            label = "$fxId.digits_output",
          )
          actual.put(
            "chainedTensorStats",
            JSONObject()
              .put("reverse_input", revInStats)
              .put("reverse_output", revOutStats)
              .put("digits_input", digInStats)
              .put("digits_output", digOutStats),
          )

          // Stage 2 reverse detections (chained after Stage 1 perspective warp)
          val hRev = host.getJSONArray("reverseDetections")
          val aRev = actual.getJSONArray("reverseDetections")
          assertEquals("$fxId reverseDetections length", hRev.length(), aRev.length())
          for (i in 0 until hRev.length()) {
            val hd = hRev.getJSONObject(i)
            val ad = aRev.getJSONObject(i)
            assertEquals("$fxId rev[$i] classId", hd.getInt("classId"), ad.getInt("classId"))
            assertEquals("$fxId rev[$i] class", hd.getString("class"), ad.getString("class"))
            assertEquals("$fxId rev[$i] score", hd.getDouble("score"), ad.getDouble("score"), CHAINED_SCORE_ATOL)
            for (c in 0..3) {
              assertEquals(hd.getJSONArray("box").getDouble(c), ad.getJSONArray("box").getDouble(c), ANDROID_BOX_PX_ATOL)
            }
          }

          // Stage 3 digit detections (chained after Stage 1 perspective warp + Stage 2 rotation)
          val hDigits = host.getJSONArray("digitDetections")
          val aDigits = actual.getJSONArray("digitDetections")
          assertEquals("$fxId digitDetections length", hDigits.length(), aDigits.length())
          for (i in 0 until hDigits.length()) {
            val hd = hDigits.getJSONObject(i)
            val ad = aDigits.getJSONObject(i)
            assertEquals("$fxId digit[$i] classId", hd.getInt("classId"), ad.getInt("classId"))
            assertEquals("$fxId digit[$i] class", hd.getString("class"), ad.getString("class"))
            assertEquals("$fxId digit[$i] digit", hd.getString("digit"), ad.getString("digit"))
            assertEquals("$fxId digit[$i] score", hd.getDouble("score"), ad.getDouble("score"), CHAINED_SCORE_ATOL)
            for (c in 0..3) {
              assertEquals(hd.getJSONArray("box").getDouble(c), ad.getJSONArray("box").getDouble(c), ANDROID_BOX_PX_ATOL)
            }
          }
        } else {
          // Verify no stale Stage 2/3 tensors when Stage 1 rejects with no_row or multiple_rows
          assertTrue("$fxId rowBox is null", actual.isNull("rowBox"))
          assertTrue("$fxId orderedPolygon is null", actual.isNull("orderedPolygon"))
          assertTrue("$fxId cropSize is null", actual.isNull("cropSize"))
          for (stale in listOf("reverse_input.f32", "reverse_output.f32", "digits_input.f32", "digits_output.f32")) {
            assertFalse("$fxId must not leave stale $stale", File(root, stale).exists())
          }
        }

        val saved = File(context.filesDir, "ocr-parity/$fxId").apply { mkdirs() }
        root.listFiles()?.forEach { it.copyTo(File(saved, it.name), overwrite = true) }
        File(context.filesDir, "$fxId.native.json").writeText(actual.toString(2))
        image.delete()
      }

      // Also verify raw gallery .jfif files end-to-end via runImageDiagnostic
      for ((jfifName, expectedDigits) in listOf("meter-upright-01.jfif" to "085918", "meter-transition-02.jfif" to "156883")) {
        val jfif = seed(jfifName)
        val reply = call { native.runImageDiagnostic(jfif.toURI().toString(), it) }.await()
        assertNull(jfifName, reply.error)
        val actual = JSONObject(reply.result!!)
        assertEquals(jfifName, PIPELINE_ID, actual.getString("pipelineId"))
        assertEquals(jfifName, "ok", actual.getString("status"))
        assertEquals(jfifName, expectedDigits, actual.getString("rawDigits"))
        assertFalse(jfifName, actual.getBoolean("isReversed"))
        assertTrue(jfifName, actual.getDouble("totalMs") >= actual.getDouble("invokeMs"))
        File(context.filesDir, "$jfifName.native.json").writeText(actual.toString(2))
        jfif.delete()
      }
    } finally {
      native.invalidate()
    }
  }

  @Test fun approvedFixturesProduceExpectedReadingsAndRejections() {
    val native = module()
    val results = JSONArray()
    val failures = mutableListOf<String>()
    try {
      val fixtures = JSONObject(String(asset("fixtures-manifest.json"))).getJSONArray("fixtures")
      assertEquals(8, fixtures.length())
      for (idx in 0 until fixtures.length()) {
        val fx = fixtures.getJSONObject(idx)
        val id = fx.getString("id")
        val expected = fx.getJSONObject("expected")
        val rejected = expected.optBoolean("promiseRejected", false)
        val image = seed("$id.${if (rejected) "bin" else "png"}")
        try {
          val reply = call { native.runImageParity(image.toURI().toString(), it) }.await()
          val record = JSONObject().put("id", id)
          if (rejected) {
            record.put("errorCode", reply.error ?: JSONObject.NULL)
            if (reply.error != expected.getString("errorCode")) failures.add("$id: ${reply.error}")
          } else if (reply.error != null) {
            record.put("errorCode", reply.error).put("message", reply.message)
            failures.add("$id: ${reply.error}: ${reply.message}")
          } else {
            val actual = JSONObject(reply.result!!)
            record.put("actual", actual)
            // Keep end-to-end behavior separate from the stricter tensor parity gate.
            for (field in listOf("status", "rawDigits", "reason", "isReversed")) {
              if (expected.has(field) && expected.opt(field)?.toString() != actual.opt(field)?.toString()) {
                failures.add("$id $field: expected=${expected.opt(field)} actual=${actual.opt(field)}")
              }
            }
            val saved = File(context.filesDir, "integration-fixtures/$id").apply { mkdirs() }
            File(context.filesDir, "ocr-parity/latest").listFiles()?.forEach {
              it.copyTo(File(saved, it.name), overwrite = true)
            }
          }
          results.put(record)
        } finally {
          image.delete()
        }
      }
      File(context.filesDir, "integration-fixtures-results.json").writeText(results.toString(2))
      assertTrue(failures.joinToString("\n"), failures.isEmpty())
    } finally {
      native.invalidate()
    }

    // Separate cold vs warm timing + memory trend measurement (fresh MeterOcrModule instance)
    val benchModule = module()
    val sample = seed("meter-upright-01.jfif")
    try {
      fun memSnapshot(stage: String): JSONObject {
        val rt = Runtime.getRuntime()
        val memInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memInfo)
        return JSONObject()
          .put("stage", stage)
          .put("totalPssKb", memInfo.totalPss)
          .put("nativePssKb", memInfo.nativePss)
          .put("dalvikPssKb", memInfo.dalvikPss)
          .put("nativeHeapAllocatedBytes", Debug.getNativeHeapAllocatedSize())
          .put("jvmHeapUsedBytes", rt.totalMemory() - rt.freeMemory())
      }

      val memBefore = memSnapshot("before_cold")
      val coldWallStart = System.nanoTime()
      val coldReply = call { benchModule.runImageDiagnostic(sample.toURI().toString(), it) }.await()
      val coldWallMs = (System.nanoTime() - coldWallStart) / 1_000_000.0
      assertNull(coldReply.message, coldReply.error)
      val coldJson = JSONObject(coldReply.result!!)
      assertEquals("085918", coldJson.getString("rawDigits"))
      val memAfterCold = memSnapshot("after_cold")

      val warmRuns = JSONArray()
      for (runIdx in 1..5) {
        val warmWallStart = System.nanoTime()
        val warmReply = call { benchModule.runImageDiagnostic(sample.toURI().toString(), it) }.await()
        val warmWallMs = (System.nanoTime() - warmWallStart) / 1_000_000.0
        assertNull(warmReply.message, warmReply.error)
        val warmJson = JSONObject(warmReply.result!!)
        assertEquals("085918", warmJson.getString("rawDigits"))
        warmRuns.put(
          JSONObject()
            .put("iteration", runIdx)
            .put("wallMs", warmWallMs)
            .put("totalMs", warmJson.getDouble("totalMs"))
            .put("invokeMs", warmJson.getDouble("invokeMs"))
            .put("memory", memSnapshot("warm_$runIdx")),
        )
      }

      val benchReport = JSONObject()
        .put("deviceModel", Build.MODEL)
        .put("deviceManufacturer", Build.MANUFACTURER)
        .put("deviceProduct", Build.PRODUCT)
        .put("deviceHardware", Build.HARDWARE)
        .put("sdkInt", Build.VERSION.SDK_INT)
        .put("release", Build.VERSION.RELEASE)
        .put("supportedAbis", JSONArray(Build.SUPPORTED_ABIS))
        .put("isEmulator", Build.FINGERPRINT.contains("generic") || Build.PRODUCT.contains("sdk_gphone") || Build.HARDWARE.contains("ranchu"))
        .put(
          "cold",
          JSONObject()
            .put("wallMs", coldWallMs)
            .put("totalMs", coldJson.getDouble("totalMs"))
            .put("invokeMs", coldJson.getDouble("invokeMs"))
            .put("memoryBefore", memBefore)
            .put("memoryAfter", memAfterCold),
        )
        .put("warmRuns", warmRuns)
      File(context.filesDir, "benchmark.native.json").writeText(benchReport.toString(2))
    } finally {
      sample.delete()
      benchModule.invalidate()
    }
  }

  @Test fun allEightExifOrientationsMatchIndependentHostInput() {
    val native = module()
    try {
      for (orientation in 1..8) {
        val image = seed("exif-$orientation.png")
        val reply = call { native.runImageParity(image.toURI().toString(), it) }.await()
        assertNull("EXIF $orientation: ${reply.message}", reply.error)
        assertTensor(
          asset("exif-$orientation.frame_input.f32"),
          File(context.filesDir, "ocr-parity/latest/frame_input.f32").readBytes(),
          1e-6,
          0.0,
          "exif-$orientation.frame_input",
        )
        image.delete()
      }
    } finally {
      native.invalidate()
    }
  }

  @Test fun releaseDebugGuardRejectsParityExportWhenNotDebuggable() {
    val releaseContext = object : ContextWrapper(context) {
      // ReactApplicationContext unwraps its input through applicationContext.
      override fun getApplicationContext(): android.content.Context = this

      override fun getApplicationInfo(): ApplicationInfo {
        return ApplicationInfo(super.getApplicationInfo()).apply {
          flags = flags and ApplicationInfo.FLAG_DEBUGGABLE.inv()
        }
      }
    }
    val reactReleaseContext = BridgeReactContext(releaseContext)
    assertEquals(0, reactReleaseContext.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE)
    val releaseModule = MeterOcrModule(reactReleaseContext)
    try {
      val sample = seed("sample.jfif")
      val reply = call { releaseModule.runImageParity(sample.toURI().toString(), it) }.await()
      assertEquals("E_PARITY_DISABLED", reply.error)
      assertTrue(reply.message.orEmpty().contains("debug builds"))
      sample.delete()
    } finally {
      releaseModule.invalidate()
    }
  }

  @Test fun jfifCorruptMissingOversizeAndUnsupportedUriSettleAndCleanTemporaryFiles() {
    val native = module()
    try {
      val jfif = seed("sample.jfif")
      val good = call { native.validateImage(jfif.toURI().toString(), it) }.await()
      assertNull(good.error)
      assertEquals(540, JSONObject(good.result!!).getInt("width"))
      for (name in listOf("corrupt.img", "oversize.png", "oversize-dimension.png")) {
        val image = seed(name)
        val reply = call { native.validateImage(image.toURI().toString(), it) }.await()
        assertNotNull(name, reply.error)
        image.delete()
      }
      for (uri in listOf("file:///not-present.jpg", "https://example.invalid/cloud-only.jpg", "content://unavailable-offline/image")) {
        val reply = call { native.validateImage(uri, it) }.await()
        assertNotNull(uri, reply.error)
        if (uri.startsWith("content:")) assertTrue(reply.message.orEmpty().contains("locally available photo"))
      }
      val huge = File(context.cacheDir, "gate-too-large.img")
      java.io.RandomAccessFile(huge, "rw").use { it.setLength(25L * 1024 * 1024 + 1) }
      val tooLarge = call { native.validateImage(huge.toURI().toString(), it) }.await()
      assertNotNull(tooLarge.error)
      assertTrue(tooLarge.message.orEmpty().contains("25 MB"))
      assertTrue(context.cacheDir.listFiles()!!.none { it.name.startsWith("meter-inspect-") || it.name.startsWith("meter-crop-") })
      huge.delete()
      jfif.delete()
    } finally {
      native.invalidate()
    }
  }

  @Test fun rapidRequestsAreNotQueuedAndTeardownIsIdempotent() {
    val native = module()
    val huge = File(context.cacheDir, "gate-busy.img")
    java.io.RandomAccessFile(huge, "rw").use { it.setLength(25L * 1024 * 1024 + 1) }
    val first = call { native.validateImage(huge.toURI().toString(), it) }
    val second = call { native.validateImage(huge.toURI().toString(), it) }.await()
    assertEquals("E_BUSY", second.error)
    native.invalidate()
    native.invalidate()
    first.await()
    assertEquals("E_MODULE_CLOSED", call { native.validateImage(huge.toURI().toString(), it) }.await().error)
    huge.delete()
  }

  @Test fun invalidationDuringImageWorkLetsTheActiveInterpreterFinishBeforeClose() {
    val native = module()
    assertNull(call { native.runRuntimeSmoke(it) }.await().error)
    val image = seed("lifecycle.png")
    val metadata = File(context.filesDir, "ocr-parity/latest/metadata.json")
    val previous = if (metadata.exists()) JSONObject(metadata.readText()).optString("runId") else ""
    val active = call { native.runImageParity(image.toURI().toString(), it) }
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5)
    while (System.nanoTime() < deadline && context.cacheDir.listFiles()!!.none { it.name.startsWith("meter-crop-") }) Thread.sleep(1)
    assertTrue("must observe image work before teardown", context.cacheDir.listFiles()!!.any { it.name.startsWith("meter-crop-") })
    native.invalidate()
    native.invalidate()
    assertEquals("E_MODULE_CLOSED", active.await().error)
    // Export is after all 3 OrtSessions finish; a premature close would prevent this fresh completed fixture.
    val finished = JSONObject(metadata.readText())
    assertNotEquals(previous, finished.getString("runId"))
    assertEquals(PIPELINE_ID, finished.getString("pipelineId"))
    assertEquals("085918", finished.getString("rawDigits"))
    assertEquals(125, finished.getJSONArray("cropSize").getInt(0))
    assertEquals(27, finished.getJSONArray("cropSize").getInt(1))
    assertEquals("E_MODULE_CLOSED", call { native.runRuntimeSmoke(it) }.await().error)
    image.delete()
  }
}
