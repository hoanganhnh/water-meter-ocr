package com.watermeterocr.ocr

import android.Manifest
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.drawable.BitmapDrawable
import android.os.SystemClock
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.ImageInfo
import androidx.camera.core.ImageProxy
import androidx.camera.view.LifecycleCameraController
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import androidx.lifecycle.Lifecycle
import java.lang.reflect.Proxy
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

/** Deterministic lifecycle/ownership checks plus labelled analyzer replay; not a sensor accuracy test. */
class RealtimeMeterActivityTest {
  @Test fun frozenFrameWaitsForConfirmationAndLaterFramesCannotReplaceIt() {
    val instrumentation = InstrumentationRegistry.getInstrumentation()
    val context = instrumentation.targetContext
    instrumentation.uiAutomation.grantRuntimePermission(context.packageName, Manifest.permission.CAMERA)
    val before = context.cacheDir.listFiles()!!.filter { it.name.startsWith("meter-realtime-") }.map { it.name }.toSet()
    val scenario = ActivityScenario.launchActivityForResult<RealtimeMeterActivity>(Intent(context, RealtimeMeterActivity::class.java))
    val evidence = solidBitmap(Color.RED)
    try {
      awaitPreload(scenario)
      scenario.onActivity { activity ->
        detachAnalyzer(activity)
        val token = generation(activity)
        val firstReading = ReadingObservation(floatArrayOf(0.1f, 0.2f, 0.8f, 0.7f),
          "085918", ReadingQualityLevel.HIGH)
        assertTrue("The first qualified frame must reach the freeze handoff without another frame",
          ReadingStability().observe(firstReading, System.nanoTime(), SystemClock.elapsedRealtime()))
        assertTrue(showFrozen(activity, token, evidence, reading("085918")))
        val button = findButton(activity.findViewById(android.R.id.content), "Dùng chỉ số này")!!
        assertTrue(button.isEnabled)
        assertFalse("Freeze must not finish before explicit confirmation", activity.isFinishing)
        assertSame(evidence, frozenBitmap(activity))
        val image = field(activity, "frozenImage") as ImageView
        assertSame(evidence, (image.drawable as BitmapDrawable).bitmap)
        assertEquals(before, context.cacheDir.listFiles()!!.filter { it.name.startsWith("meter-realtime-") }.map { it.name }.toSet())
        val later = solidBitmap(Color.BLUE)
        assertFalse(showFrozen(activity, token, later, reading("999999")))
        assertTrue("Rejected evidence must be recycled", later.isRecycled)
        assertSame(evidence, frozenBitmap(activity))
        assertEquals("085918", (field(activity, "frozenResult") as JSONObject).getString("rawDigits"))
        button.performClick()
      }
      val result = scenario.result
      assertEquals(android.app.Activity.RESULT_OK, result.resultCode)
      val payload = JSONObject(result.resultData.getStringExtra("reading")!!)
      assertEquals("085918", payload.getJSONObject("result").getString("rawDigits"))
      val savedFile = java.io.File(android.net.Uri.parse(payload.getString("uri")).path!!)
      try {
        val saved = BitmapFactory.decodeFile(savedFile.path)
        assertEquals(72, saved.width)
        assertEquals(48, saved.height)
        val pixel = saved.getPixel(36, 24)
        assertTrue("JPEG must contain the retained red frame", Color.red(pixel) > 240 && Color.blue(pixel) < 15)
        saved.recycle()
      } finally { savedFile.delete() }
    } finally { scenario.close(); if (!evidence.isRecycled) evidence.recycle() }
  }

  @Test fun retryDiscardsFrozenEvidenceAndRejectsOldGeneration() {
    val scenario = launchScanner()
    val evidence = solidBitmap(Color.RED)
    try {
      scenario.onActivity { activity ->
        detachAnalyzer(activity)
        val oldToken = generation(activity)
        assertTrue(showFrozen(activity, oldToken, evidence, reading("085918")))
        findButton(activity.findViewById(android.R.id.content), "Quét lại")!!.performClick()
        detachAnalyzer(activity)
        val statusView = field(activity, "status") as TextView
        assertEquals(2, statusView.minLines)
        assertEquals(2, statusView.maxLines)
        assertTrue(evidence.isRecycled)
        assertNull(frozenBitmap(activity))
        assertNull(field(activity, "frozenResult"))
        assertFalse(findButton(activity.findViewById(android.R.id.content), "Dùng chỉ số này")!!.isEnabled)
        val stale = solidBitmap(Color.BLUE)
        assertFalse(showFrozen(activity, oldToken, stale, reading("999999")))
        assertTrue(stale.isRecycled)
        assertNull(frozenBitmap(activity))
      }
    } finally { scenario.close(); if (!evidence.isRecycled) evidence.recycle() }
  }

  @Test fun backgroundDiscardsEvidenceAndRejectsPendingFrame() {
    val scenario = launchScanner()
    val evidence = solidBitmap(Color.RED)
    try {
      lateinit var activity: RealtimeMeterActivity
      scenario.onActivity {
        activity = it
        detachAnalyzer(it)
        assertTrue(showFrozen(it, generation(it), evidence, reading("085918")))
      }
      scenario.moveToState(Lifecycle.State.CREATED)
      InstrumentationRegistry.getInstrumentation().runOnMainSync {
        assertTrue(evidence.isRecycled)
        assertNull(frozenBitmap(activity))
        val pending = solidBitmap(Color.BLUE)
        assertFalse(showFrozen(activity, generation(activity), pending, reading("999999")))
        assertTrue(pending.isRecycled)
      }
      scenario.moveToState(Lifecycle.State.RESUMED)
      scenario.onActivity {
        detachAnalyzer(it)
        assertNull(frozenBitmap(it))
        assertFalse(findButton(it.findViewById(android.R.id.content), "Dùng chỉ số này")!!.isEnabled)
      }
    } finally { scenario.close(); if (!evidence.isRecycled) evidence.recycle() }
  }

  @Test fun staleGenerationAndQueuedFramesCloseBeforeReadingPixels() {
    val scenario = launchScanner()
    try {
      lateinit var activity: RealtimeMeterActivity
      var token = 0
      scenario.onActivity {
        activity = it
        detachAnalyzer(it)
        token = generation(it)
      }
      val executor = field(activity, "worker") as ExecutorService
      val analyzer = activity.javaClass.getDeclaredMethod("analyze", ImageProxy::class.java,
        Int::class.javaPrimitiveType, Long::class.javaPrimitiveType).apply { isAccessible = true }
      val now = SystemClock.elapsedRealtime()
      for ((frameToken, admittedAt) in listOf(token - 1 to now, token to now - 3001)) {
        val closed = AtomicInteger()
        val reads = AtomicInteger()
        val frame = Proxy.newProxyInstance(ImageProxy::class.java.classLoader, arrayOf(ImageProxy::class.java)) { _, method, _ ->
          if (method.name == "close") closed.incrementAndGet() else reads.incrementAndGet()
          null
        } as ImageProxy
        val done = CountDownLatch(1)
        var failure: Throwable? = null
        executor.execute {
          try { analyzer.invoke(activity, frame, frameToken, admittedAt) }
          catch (error: Throwable) { failure = error }
          finally { done.countDown() }
        }
        assertTrue("Admission rejection did not finish", done.await(30, TimeUnit.SECONDS))
        failure?.let { throw AssertionError("Admission rejection failed", it) }
        assertEquals("Every rejected frame must close exactly once", 1, closed.get())
        assertEquals("Stale frames must not access metadata or pixels", 0, reads.get())
      }
      scenario.onActivity {
        assertNull(frozenBitmap(it))
        assertFalse(findButton(it.findViewById(android.R.id.content), "Dùng chỉ số này")!!.isEnabled)
      }
    } finally { scenario.close() }
  }

  @Test fun stableFramesFreezeAndReturnConfirmedEvidence() {
    val instrumentation = InstrumentationRegistry.getInstrumentation()
    val context = instrumentation.targetContext
    instrumentation.uiAutomation.grantRuntimePermission(context.packageName, Manifest.permission.CAMERA)
    val source = instrumentation.context.assets.open("meter-upright-01.png").use { BitmapFactory.decodeStream(it) }
    val scenario = ActivityScenario.launchActivityForResult<RealtimeMeterActivity>(
      Intent(context, RealtimeMeterActivity::class.java))
    try {
      awaitPreload(scenario)
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
          diagnostics.add(status.text.toString())
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

  private fun launchScanner(): ActivityScenario<RealtimeMeterActivity> {
    val instrumentation = InstrumentationRegistry.getInstrumentation()
    val context = instrumentation.targetContext
    instrumentation.uiAutomation.grantRuntimePermission(context.packageName, Manifest.permission.CAMERA)
    return ActivityScenario.launch<RealtimeMeterActivity>(Intent(context, RealtimeMeterActivity::class.java)).also { awaitPreload(it) }
  }

  private fun awaitPreload(scenario: ActivityScenario<RealtimeMeterActivity>) {
    lateinit var executor: ExecutorService
    scenario.onActivity { executor = field(it, "worker") as ExecutorService }
    val done = CountDownLatch(1)
    executor.execute { done.countDown() }
    assertTrue("Engine preload did not finish", done.await(30, TimeUnit.SECONDS))
    InstrumentationRegistry.getInstrumentation().waitForIdleSync()
  }

  private fun detachAnalyzer(activity: RealtimeMeterActivity) {
    (field(activity, "camera") as LifecycleCameraController).clearImageAnalysisAnalyzer()
  }

  private fun field(activity: RealtimeMeterActivity, name: String): Any? =
    activity.javaClass.getDeclaredField(name).apply { isAccessible = true }.get(activity)

  private fun generation(activity: RealtimeMeterActivity) = field(activity, "generation") as Int
  private fun frozenBitmap(activity: RealtimeMeterActivity) = field(activity, "frozenBitmap") as Bitmap?
  private fun solidBitmap(color: Int) = Bitmap.createBitmap(72, 48, Bitmap.Config.ARGB_8888).apply { eraseColor(color) }
  private fun reading(digits: String) = JSONObject().put("status", "ok").put("rawDigits", digits)
    .put("source", "realtime").put("imageSize", org.json.JSONArray(intArrayOf(72, 48)))

  private fun showFrozen(activity: RealtimeMeterActivity, token: Int, bitmap: Bitmap, result: JSONObject): Boolean {
    val method = activity.javaClass.getDeclaredMethod("showFrozenReading", Int::class.javaPrimitiveType,
      Bitmap::class.java, JSONObject::class.java, RectF::class.java).apply { isAccessible = true }
    return method.invoke(activity, token, bitmap, result, RectF(0f, 0f, 72f, 48f)) as Boolean
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
