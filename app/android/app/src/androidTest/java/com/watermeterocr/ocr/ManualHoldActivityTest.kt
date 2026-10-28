package com.watermeterocr.ocr

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.RectF
import android.os.SystemClock
import android.view.View
import android.widget.Button
import androidx.camera.view.LifecycleCameraController
import androidx.lifecycle.Lifecycle
import androidx.test.core.app.ActivityScenario
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutorService
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

/** Exercises actual hold/confirm clicks with deterministic same-frame evidence, not sensor accuracy. */
class ManualHoldActivityTest {
  @Test fun liveButtonsStartWithDisabledHoldAndHiddenConfirmation() {
    val scenario = launchScanner()
    try {
      scenario.onActivity {
        assertEquals(View.VISIBLE, hold(it).visibility)
        assertFalse(hold(it).isEnabled)
        assertEquals(View.GONE, confirm(it).visibility)
        assertFalse(confirm(it).isEnabled)
      }
    } finally { scenario.close() }
  }

  @Test fun manualHoldKeepsLatestExactFrameAndRequiresSeparateConfirmation() {
    val instrumentation = InstrumentationRegistry.getInstrumentation()
    val context = instrumentation.targetContext
    val before = savedReadings()
    val scenario = launchScanner(forResult = true)
    val reportBuilds = AtomicInteger()
    val red = solidBitmap(Color.RED)
    val blue = solidBitmap(Color.BLUE)
    val blocked = CountDownLatch(1)
    val release = CountDownLatch(1)
    lateinit var activity: RealtimeMeterActivity
    try {
      scenario.onActivity { activity = it }
      worker(activity).execute {
        blocked.countDown()
        release.await(20, TimeUnit.SECONDS)
      }
      assertTrue("Worker barrier did not start", blocked.await(30, TimeUnit.SECONDS))
      scenario.onActivity {
        // A valid LOW reading may be inspected manually without weakening automatic capture.
        assertTrue(offer(it, red, frame("174305", score = 0.4f) { reportBuilds.incrementAndGet() }))
        assertTrue(hold(it).isEnabled)
        assertEquals(View.GONE, confirm(it).visibility)
        assertTrue(offer(it, blue, frame("092928") { reportBuilds.incrementAndGet() }))
        assertTrue("Replacing a candidate must release the old bitmap", red.isRecycled)
        assertEquals("Candidate publication must not serialize reports", 0, reportBuilds.get())
        hold(it).performClick()
        assertSame("The selected bitmap must freeze immediately even while worker is busy", blue, frozenBitmap(it))
        assertEquals(View.GONE, hold(it).visibility)
        assertEquals(View.GONE, confirm(it).visibility)
        assertFalse(confirm(it).isEnabled)
        assertNull(field(it, "frozenResult"))
        assertFalse("Holding must not submit the reading", it.isFinishing)
        assertEquals("Holding must not create a JPEG", before, savedReadings())
        val later = solidBitmap(Color.GREEN)
        assertFalse(offer(it, later, frame("111111")))
        assertTrue(later.isRecycled)
        assertSame(blue, frozenBitmap(it))
      }
      release.countDown()
      awaitWorker(activity)
      scenario.onActivity {
        assertEquals(1, reportBuilds.get())
        assertSame(blue, frozenBitmap(it))
        assertEquals("092928", (field(it, "frozenResult") as JSONObject).getString("rawDigits"))
        assertEquals(View.VISIBLE, confirm(it).visibility)
        assertTrue(confirm(it).isEnabled)
        assertEquals(before, savedReadings())
        assertFalse(it.isFinishing)
        confirm(it).performClick()
      }
      val result = scenario.result
      assertEquals(Activity.RESULT_OK, result.resultCode)
      val payload = JSONObject(result.resultData.getStringExtra("reading")!!)
      assertEquals("092928", payload.getJSONObject("result").getString("rawDigits"))
      assertEquals("realtime", payload.getJSONObject("result").getString("source"))
      val file = File(android.net.Uri.parse(payload.getString("uri")).path!!)
      try {
        val saved = BitmapFactory.decodeFile(file.path)
        assertEquals(72, saved.width); assertEquals(48, saved.height)
        val pixel = saved.getPixel(36, 24)
        assertTrue("Saved JPEG must contain the selected blue frame", Color.blue(pixel) > 240 && Color.red(pixel) < 15)
        saved.recycle()
      } finally { file.delete() }
    } finally {
      release.countDown(); scenario.close()
      for (bitmap in listOf(red, blue)) if (!bitmap.isRecycled) bitmap.recycle()
    }
  }

  @Test fun missingDigitsUncertainOrMalformedEvidenceCannotLeaveHoldEnabled() {
    val scenario = launchScanner()
    try {
      for (invalid in listOf(frame("92928"), frame("174305", status = "uncertain"),
        frame("174305", invalidGeometry = true), frame("174305", score = Float.NaN))) {
        scenario.onActivity {
          val previous = solidBitmap(Color.RED)
          assertTrue(offer(it, previous, frame()))
          val rejected = solidBitmap(Color.BLUE)
          assertFalse(offer(it, rejected, invalid))
          assertTrue(previous.isRecycled)
          assertTrue(rejected.isRecycled)
          assertFalse(hold(it).isEnabled)
          assertNull(field(it, "manualCandidate"))
          assertNull(frozenBitmap(it))
          assertEquals(View.GONE, confirm(it).visibility)
        }
      }
    } finally { scenario.close() }
  }

  @Test fun staleGenerationCannotReplaceOrDisableCurrentCandidate() {
    val scenario = launchScanner()
    val current = solidBitmap(Color.RED)
    try {
      scenario.onActivity {
        assertTrue(offer(it, current, frame()))
        val stale = solidBitmap(Color.BLUE)
        assertFalse(offer(it, stale, frame("111111"), token = generation(it) - 1))
        assertTrue(stale.isRecycled)
        assertFalse(current.isRecycled)
        assertTrue(hold(it).isEnabled)
      }
    } finally { scenario.close(); if (!current.isRecycled) current.recycle() }
  }

  @Test fun candidateExpiresWithoutAnotherCameraResult() {
    val scenario = launchScanner()
    val candidate = solidBitmap(Color.RED)
    try {
      scenario.onActivity { assertTrue(offer(it, candidate, frame())) }
      SystemClock.sleep(1700)
      InstrumentationRegistry.getInstrumentation().waitForIdleSync()
      scenario.onActivity {
        assertTrue(candidate.isRecycled)
        assertNull(field(it, "manualCandidate"))
        assertFalse(hold(it).isEnabled)
        val alreadyOld = solidBitmap(Color.BLUE)
        assertFalse(offer(it, alreadyOld, frame(), finishedAt = SystemClock.elapsedRealtime() - 2000))
        assertTrue(alreadyOld.isRecycled)
        assertNull(frozenBitmap(it))
      }
    } finally { scenario.close(); if (!candidate.isRecycled) candidate.recycle() }
  }

  @Test fun retryAndBackgroundClearCandidatesAndRestoreLiveButtons() {
    val scenario = launchScanner()
    val red = solidBitmap(Color.RED)
    val blue = solidBitmap(Color.BLUE)
    try {
      scenario.onActivity {
        val previousToken = generation(it)
        assertTrue(offer(it, red, frame()))
        (field(it, "retry") as Button).performClick()
        detach(it)
        assertTrue(red.isRecycled)
        assertNull(field(it, "manualCandidate"))
        assertFalse(hold(it).isEnabled)
        assertEquals(View.VISIBLE, hold(it).visibility)
        assertEquals(View.GONE, confirm(it).visibility)
        val stale = solidBitmap(Color.GREEN)
        assertFalse(offer(it, stale, frame(), token = previousToken))
        assertTrue(stale.isRecycled)
        assertTrue(offer(it, blue, frame()))
      }
      scenario.moveToState(Lifecycle.State.CREATED)
      assertTrue(blue.isRecycled)
      scenario.moveToState(Lifecycle.State.RESUMED)
      scenario.onActivity {
        detach(it)
        assertNull(field(it, "manualCandidate"))
        assertFalse(hold(it).isEnabled)
        assertEquals(View.VISIBLE, hold(it).visibility)
        assertEquals(View.GONE, confirm(it).visibility)
      }
    } finally {
      scenario.close()
      for (bitmap in listOf(red, blue)) if (!bitmap.isRecycled) bitmap.recycle()
    }
  }

  @Test fun automaticFreezeStillWorksWithoutClickingHoldAndDiscardsCandidate() {
    val scenario = launchScanner()
    val candidate = solidBitmap(Color.RED)
    val automatic = solidBitmap(Color.BLUE)
    val builds = AtomicInteger()
    try {
      scenario.onActivity {
        assertTrue(offer(it, candidate, frame { builds.incrementAndGet() }))
        val method = it.javaClass.getDeclaredMethod("showFrozenReading", Int::class.javaPrimitiveType,
          Bitmap::class.java, JSONObject::class.java, RectF::class.java).apply { isAccessible = true }
        assertTrue(method.invoke(it, generation(it), automatic, result("174305"), RectF(0f, 8f, 72f, 40f)) as Boolean)
        assertTrue(candidate.isRecycled)
        assertNull(field(it, "manualCandidate"))
        assertEquals("Unused manual candidate must not build its report", 0, builds.get())
        assertSame(automatic, frozenBitmap(it))
        assertEquals(View.GONE, hold(it).visibility)
        assertEquals(View.VISIBLE, confirm(it).visibility)
        assertTrue(confirm(it).isEnabled)
        assertFalse(it.isFinishing)
      }
    } finally {
      scenario.close()
      for (bitmap in listOf(candidate, automatic)) if (!bitmap.isRecycled) bitmap.recycle()
    }
  }

  @Test fun reportFromPreviousHoldCannotReappearAfterRetry() {
    val scenario = launchScanner()
    val candidate = solidBitmap(Color.RED)
    val blocked = CountDownLatch(1)
    val release = CountDownLatch(1)
    lateinit var activity: RealtimeMeterActivity
    try {
      scenario.onActivity { activity = it }
      worker(activity).execute { blocked.countDown(); release.await(20, TimeUnit.SECONDS) }
      assertTrue(blocked.await(30, TimeUnit.SECONDS))
      scenario.onActivity {
        assertTrue(offer(it, candidate, frame()))
        hold(it).performClick()
        assertSame(candidate, frozenBitmap(it))
        (field(it, "retry") as Button).performClick()
        detach(it)
        assertTrue(candidate.isRecycled)
        assertNull(frozenBitmap(it))
        assertNull(field(it, "frozenResult"))
      }
      release.countDown()
      awaitWorker(activity)
      scenario.onActivity {
        assertNull(frozenBitmap(it))
        assertNull(field(it, "frozenResult"))
        assertEquals(View.GONE, confirm(it).visibility)
        assertFalse(confirm(it).isEnabled)
        assertEquals(View.VISIBLE, hold(it).visibility)
      }
    } finally {
      release.countDown(); scenario.close()
      if (!candidate.isRecycled) candidate.recycle()
    }
  }

  private fun launchScanner(forResult: Boolean = false): ActivityScenario<RealtimeMeterActivity> {
    val instrumentation = InstrumentationRegistry.getInstrumentation()
    val context = instrumentation.targetContext
    instrumentation.uiAutomation.grantRuntimePermission(context.packageName, Manifest.permission.CAMERA)
    val intent = Intent(context, RealtimeMeterActivity::class.java)
    val scenario = if (forResult) ActivityScenario.launchActivityForResult<RealtimeMeterActivity>(intent)
      else ActivityScenario.launch<RealtimeMeterActivity>(intent)
    lateinit var activity: RealtimeMeterActivity
    scenario.onActivity { activity = it }
    awaitWorker(activity)
    scenario.onActivity { detach(it) }
    awaitWorker(activity)
    scenario.onActivity {
      it.javaClass.getDeclaredMethod("resetScan").apply { isAccessible = true }.invoke(it)
      detach(it)
    }
    return scenario
  }

  private fun awaitWorker(activity: RealtimeMeterActivity) {
    val done = CountDownLatch(1)
    worker(activity).execute { done.countDown() }
    assertTrue("Worker did not finish", done.await(30, TimeUnit.SECONDS))
    InstrumentationRegistry.getInstrumentation().waitForIdleSync()
  }

  private fun worker(activity: RealtimeMeterActivity) = field(activity, "worker") as ExecutorService
  private fun detach(activity: RealtimeMeterActivity) {
    (field(activity, "camera") as LifecycleCameraController).clearImageAnalysisAnalyzer()
  }
  private fun field(activity: RealtimeMeterActivity, name: String): Any? =
    activity.javaClass.getDeclaredField(name).apply { isAccessible = true }.get(activity)
  private fun generation(activity: RealtimeMeterActivity) = field(activity, "generation") as Int
  private fun hold(activity: RealtimeMeterActivity) = field(activity, "hold") as Button
  private fun confirm(activity: RealtimeMeterActivity) = field(activity, "confirm") as Button
  private fun frozenBitmap(activity: RealtimeMeterActivity) = field(activity, "frozenBitmap") as Bitmap?
  private fun solidBitmap(color: Int) = Bitmap.createBitmap(72, 48, Bitmap.Config.ARGB_8888).apply { eraseColor(color) }
  private fun savedReadings() = InstrumentationRegistry.getInstrumentation().targetContext.cacheDir.listFiles()
    .orEmpty().filter { it.name.startsWith("meter-realtime-") }.map { it.name }.toSet()

  private fun result(digits: String) = JSONObject().put("status", "ok").put("rawDigits", digits)
    .put("source", "realtime").put("imageSize", JSONArray(intArrayOf(72, 48)))

  private fun frame(
    digits: String = "174305",
    status: String = "ok",
    invalidGeometry: Boolean = false,
    score: Float = 0.8f,
    onReport: () -> Unit = {},
  ): SuppliedYoloEngine.RealtimeFrame {
    val points = arrayOf(floatArrayOf(0f, 8f), floatArrayOf(72f, 8f),
      floatArrayOf(72f, 40f), floatArrayOf(0f, 40f))
    val detections = digits.mapIndexed { index, digit ->
      val left = 1f + index * 11f
      SuppliedYoloDecoder.BoxDetection(index, digit - '0', digit.toString(), score,
        floatArrayOf(left, 3f, left + 8f, 29f), left + 4f, 16f, digit.toString())
    }
    val raw = if (status == "ok") digits else null
    val reading = SuppliedYoloDecoder.DigitReading(status, raw,
      if (raw == null) "no_characters" else null, score.toDouble(), digits,
      digits.map { it.toString() }, detections)
    val crop = SuppliedYoloDecoder.CropPlan(null,
      if (invalidGeometry) intArrayOf(72, 8, 0, 40) else intArrayOf(0, 8, 72, 40), points, 72, 32)
    val rows = listOf(SuppliedYoloDecoder.FrameDetection(0, 0, "khung", 0.75f,
      floatArrayOf(36f, 24f, 72f, 32f, 0f), points))
    return SuppliedYoloEngine.RealtimeFrame(72, 48, reading, crop, rows) {
      onReport()
      result(digits)
    }
  }

  private fun offer(
    activity: RealtimeMeterActivity,
    bitmap: Bitmap,
    frame: SuppliedYoloEngine.RealtimeFrame,
    token: Int = generation(activity),
    finishedAt: Long = SystemClock.elapsedRealtime(),
  ): Boolean {
    val method = activity.javaClass.getDeclaredMethod("offerManualReading", Int::class.javaPrimitiveType,
      Bitmap::class.java, SuppliedYoloEngine.RealtimeFrame::class.java, RectF::class.java,
      Long::class.javaPrimitiveType, Long::class.javaPrimitiveType).apply { isAccessible = true }
    return method.invoke(activity, token, bitmap, frame, RectF(0f, 8f, 72f, 40f), finishedAt - 30, finishedAt) as Boolean
  }
}
