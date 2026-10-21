package com.watermeterocr.ocr

import android.Manifest
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.widget.Button
import android.widget.TextView
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import androidx.camera.view.LifecycleCameraController
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import java.lang.reflect.Proxy
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

/** Injects labelled Bitmap frames through the real analyzer/engine/UI. Not a sensor accuracy test. */
class RealtimeMeterActivityTest {
  @Test fun stableFramesFreezeAndReturnConfirmedEvidence() {
    val instrumentation = InstrumentationRegistry.getInstrumentation()
    val context = instrumentation.targetContext
    instrumentation.uiAutomation.grantRuntimePermission(context.packageName, Manifest.permission.CAMERA)
    val source = instrumentation.context.assets.open("meter-upright-01.png").use { BitmapFactory.decodeStream(it) }
    val scenario = ActivityScenario.launchActivityForResult<RealtimeMeterActivity>(
      Intent(context, RealtimeMeterActivity::class.java))
    try {
      lateinit var activity: RealtimeMeterActivity
      scenario.onActivity {
        activity = it
        val field = it.javaClass.getDeclaredField("camera").apply { isAccessible = true }
        (field.get(it) as LifecycleCameraController).clearImageAnalysisAnalyzer()
      }
      val executor = activity.javaClass.getDeclaredField("worker").apply { isAccessible = true }
        .get(activity) as ExecutorService
      val analyzer = activity.javaClass.getDeclaredMethod("analyze", ImageProxy::class.java).apply { isAccessible = true }
      var confirmed = false
      val diagnostics = mutableListOf<String>()
      for (attempt in 0 until 6) {
        val done = CountDownLatch(1)
        var failure: Throwable? = null
        executor.execute {
          try { analyzer.invoke(activity, bitmapFrame(source)) }
          catch (error: Throwable) { failure = error }
          finally { done.countDown() }
        }
        assertTrue("Analyzer did not finish", done.await(30, TimeUnit.SECONDS))
        failure?.let { throw AssertionError("Frame failed", it) }
        instrumentation.waitForIdleSync()
        scenario.onActivity {
          confirmed = findButton(it.findViewById(android.R.id.content), "Dùng chỉ số này")!!.isEnabled
          val status = it.javaClass.getDeclaredField("status").apply { isAccessible = true }.get(it) as TextView
          val gate = it.javaClass.getDeclaredField("stability").apply { isAccessible = true }.get(it)
          val history = gate.javaClass.getDeclaredField("samples").apply { isAccessible = true }.get(gate)
          diagnostics.add("${status.text}: $history")
        }
        if (confirmed) break
        Thread.sleep(450)
      }
      scenario.onActivity {
        val confirm = findButton(it.findViewById(android.R.id.content), "Dùng chỉ số này")!!
        assertTrue("Matching frames should enable confirmation: $diagnostics", confirmed && confirm.isEnabled)
        confirm.performClick()
      }
      val result = scenario.result
      assertEquals(android.app.Activity.RESULT_OK, result.resultCode)
      val payload = JSONObject(result.resultData.getStringExtra("reading")!!)
      assertEquals("085918", payload.getJSONObject("result").getString("rawDigits"))
      assertEquals("realtime", payload.getJSONObject("result").getString("source"))
      val file = java.io.File(android.net.Uri.parse(payload.getString("uri")).path!!)
      try {
        assertTrue(file.length() > 0)
        val saved = BitmapFactory.decodeFile(file.path)
        assertEquals(source.width, saved.width); assertEquals(source.height, saved.height)
        saved.recycle()
      } finally { file.delete() }
    } finally { scenario.close(); source.recycle() }
  }

  private fun findButton(view: View, label: String): Button? {
    if (view is Button && view.text.toString() == label) return view
    if (view is ViewGroup) for (i in 0 until view.childCount) {
      findButton(view.getChildAt(i), label)?.let { return it }
    }
    return null
  }

  private fun bitmapFrame(bitmap: Bitmap): ImageProxy {
    val info = Proxy.newProxyInstance(ImageInfo::class.java.classLoader, arrayOf(ImageInfo::class.java)) { _, method, _ ->
      when (method.name) { "getRotationDegrees" -> 0; "getTimestamp" -> System.nanoTime(); else -> null }
    } as ImageInfo
    return Proxy.newProxyInstance(ImageProxy::class.java.classLoader, arrayOf(ImageProxy::class.java)) { _, method, _ ->
      when (method.name) {
        "getWidth" -> bitmap.width
        "getHeight" -> bitmap.height
        "getCropRect" -> Rect(0, 0, bitmap.width, bitmap.height)
        "getImageInfo" -> info
        "toBitmap" -> bitmap.copy(Bitmap.Config.ARGB_8888, false)
        "close" -> null
        else -> null
      }
    } as ImageProxy
  }
}
