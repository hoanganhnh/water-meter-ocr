package com.watermeterocr.ocr

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.watermeterocr.BuildConfig
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/** Report lifetime checks plus model-backed compatibility using the staged, labelled fixture. */
@RunWith(AndroidJUnit4::class)
class RealtimeEngineReportTest {
  @Test fun rejectedEvidenceDoesNotBuildReportAndRetainedEvidenceBuildsOnlyOnce() {
    var builds = 0
    val reading = SuppliedYoloDecoder.DigitReading(
      "uncertain", null, "no_characters", 0.0, "", emptyList(), emptyList())
    val crop = SuppliedYoloDecoder.CropPlan("no_row", null, null, 0, 0)
    val frame = SuppliedYoloEngine.RealtimeFrame(1280, 960, reading, crop, emptyList()) {
      builds++
      JSONObject().put("imagePixelSha256", "same-frame-hash")
    }

    assertEquals(1280, frame.width)
    assertEquals(960, frame.height)
    assertSame(reading, frame.reading)
    assertSame(crop, frame.cropPlan)
    assertTrue(frame.frameDetections.isEmpty())
    assertEquals("Inspecting evidence must not serialize or hash it", 0, builds)

    val retained = frame.report()
    assertEquals(1, builds)
    assertEquals("same-frame-hash", retained.getString("imagePixelSha256"))
    assertSame("Repeated access must reuse the same report", retained, frame.report())
    assertEquals("Repeated report access must not rebuild evidence", 1, builds)
  }

  @Test fun realtimeDetailedReportMatchesStillImageReportFromSameBitmap() {
    val instrumentation = InstrumentationRegistry.getInstrumentation()
    // Prepared by the existing gate-assets workflow; a missing fixture is a failed prerequisite.
    val bitmap = instrumentation.context.assets.open("meter-upright-01.png").use {
      checkNotNull(BitmapFactory.decodeStream(it)) { "Cannot decode meter-upright-01.png" }
    }
    try {
      SuppliedYoloEngine(instrumentation.targetContext).use { engine ->
        val still = JSONObject(engine.run(bitmap, synthetic = false, saveParity = false))
        val frame = engine.runRealtime(bitmap)
        val realtime = frame.report()

        assertEquals("ok", still.getString("status"))
        assertEquals("085918", still.getString("rawDigits"))
        assertEquals(bitmap.width, frame.width)
        assertEquals(bitmap.height, frame.height)
        assertEquals(still.getString("rawDigits"), frame.reading.rawDigits)
        assertEquals(BuildConfig.OCR_REVERSE_ENABLED, realtime.getBoolean("reverseEnabled"))
        if (!BuildConfig.OCR_REVERSE_ENABLED) {
          assertEquals("disabled_by_config", realtime.getString("reverseTriggerReason"))
          assertTrue(realtime.isNull("reverseInputShape"))
          assertTrue(realtime.isNull("reverseOutputShape"))
        }
        for (field in listOf("imagePixelSha256", "cropPixelSha256", "cropUsedPixelSha256")) {
          assertEquals("$field must retain the complete digest", 64, realtime.getString(field).length)
          assertEquals(still.getString(field), realtime.getString(field))
        }
        assertJsonEquals("result", still, realtime)
        assertSame(realtime, frame.report())
      }
    } finally {
      bitmap.recycle()
    }
  }

  private fun assertJsonEquals(path: String, expected: Any?, actual: Any?) {
    when (expected) {
      is JSONObject -> {
        assertTrue("$path must be an object", actual is JSONObject)
        val objectValue = actual as JSONObject
        val expectedKeys = expected.keys().asSequence().toSet()
        assertEquals("$path fields changed", expectedKeys, objectValue.keys().asSequence().toSet())
        for (key in expectedKeys) {
          // Elapsed timings vary across two actual inference calls; all structural data must match.
          if (path == "result" && key == "invokeMs") continue
          assertJsonEquals("$path.$key", expected.get(key), objectValue.get(key))
        }
      }
      is JSONArray -> {
        assertTrue("$path must be an array", actual is JSONArray)
        val arrayValue = actual as JSONArray
        assertEquals("$path length changed", expected.length(), arrayValue.length())
        for (index in 0 until expected.length()) {
          assertJsonEquals("$path[$index]", expected.get(index), arrayValue.get(index))
        }
      }
      is Number -> {
        assertTrue("$path must be numeric", actual is Number)
        assertEquals(path, expected.toDouble(), (actual as Number).toDouble(), 0.0)
      }
      else -> assertEquals(path, expected, actual)
    }
  }
}
