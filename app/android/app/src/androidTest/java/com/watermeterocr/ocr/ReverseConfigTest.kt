package com.watermeterocr.ocr

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.watermeterocr.BuildConfig
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class ReverseConfigTest {
  @Test fun disabledReverseSkipsInferenceForPhotoAndDiagnostic() {
    assertFalse("Run this test with -PocrReverseEnabled=false", BuildConfig.OCR_REVERSE_ENABLED)
    val instrumentation = InstrumentationRegistry.getInstrumentation()
    val photo = instrumentation.context.assets.open("sample.jfif").use { BitmapFactory.decodeStream(it) }
    val dummy = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
    try {
      SuppliedYoloEngine(instrumentation.targetContext).use { engine ->
        val diagnostic = JSONObject(engine.run(dummy, true, false))
        assertFalse(diagnostic.getBoolean("reverseEnabled"))
        assertTrue(diagnostic.isNull("reverseOutputShape"))
        val result = JSONObject(engine.run(photo, false, false))
        assertFalse(result.getBoolean("reverseEnabled"))
        assertEquals("disabled_by_config", result.getString("reverseTriggerReason"))
        assertFalse(result.getBoolean("isReversed"))
        assertTrue(result.isNull("reverseInputShape"))
        assertTrue(result.isNull("reverseOutputShape"))
        assertEquals(0, result.getJSONArray("reverseDetections").length())
        assertFalse("Photo must reach the digit model", result.isNull("digitsOutputShape"))
      }
    } finally {
      photo.recycle()
      dummy.recycle()
    }
  }
}
