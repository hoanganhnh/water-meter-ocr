package com.watermeterocr.ocr

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.hypot

class SuppliedYoloDecoderTest {

  private fun det(
    anchor: Int,
    classId: Int,
    label: String,
    score: Float,
    x1: Float,
    y1: Float = 2f,
    x2: Float = x1 + 14f,
    y2: Float = 26f,
  ): SuppliedYoloDecoder.BoxDetection = SuppliedYoloDecoder.BoxDetection(
    anchor = anchor,
    classId = classId,
    label = label,
    score = score,
    box = floatArrayOf(x1, y1, x2, y2),
    cx = (x1 + x2) / 2f,
    cy = (y1 + y2) / 2f,
  )

  @Test
  fun mapDigitLabelCoversAllThirtyClassesAndRejectsUnsupportedLabels() {
    for (clsId in 0 until 30) {
      assertEquals((clsId % 10).toString(), SuppliedYoloDecoder.mapDigitLabel(clsId.toString()))
    }
    for (invalid in listOf("-1", "30", "1n", "blank", "")) {
      var threw = false
      try {
        SuppliedYoloDecoder.mapDigitLabel(invalid)
      } catch (_: IllegalArgumentException) {
        threw = true
      }
      assertTrue("Expected IllegalArgumentException for label '$invalid'", threw)
    }
  }

  @Test
  fun validateConfigAcceptsApprovedManifestAndRejectsTamperedMetadata() {
    val valid = SuppliedYoloDecoder.BundleConfig(
      schemaVersion = 1,
      pipelineId = SuppliedYoloDecoder.PIPELINE_ID,
      semanticsVersion = SuppliedYoloDecoder.SEMANTICS_VERSION,
      contractStatus = "approved",
      runtime = "ONNX Runtime CPU",
      preprocessVersion = SuppliedYoloDecoder.PREPROCESS_VERSION,
      colorOrder = "RGB",
      padValue = 114,
      frame = SuppliedYoloDecoder.FrameModelConfig(
        file = "frame.onnx",
        sha256 = "b".repeat(64),
        sizeBytes = 12_463_574L,
        imgsz = 640,
        inputShape = listOf(1L, 3L, 640L, 640L),
        outputShape = listOf(1L, 6L, 8400L),
        confThres = 0.25f,
        iouThres = 0.50f,
        maxDet = 300,
        names = SuppliedYoloDecoder.DEFAULT_FRAME_NAMES,
      ),
      reverse = SuppliedYoloDecoder.ReverseModelConfig(
        file = "reverse.onnx",
        sha256 = "e".repeat(64),
        sizeBytes = 12_116_287L,
        imgsz = 320,
        inputShape = listOf(1L, 3L, 320L, 320L),
        outputShape = listOf(1L, 10L, 2100L),
        confThres = 0.25f,
        iouThres = 0.50f,
        maxDet = 300,
        agnosticNms = true,
        names = SuppliedYoloDecoder.DEFAULT_REVERSE_NAMES,
        asymmetricLabels = SuppliedYoloDecoder.DEFAULT_ASYMMETRIC_REVERSE_LABELS,
        symmetricLabels = SuppliedYoloDecoder.DEFAULT_SYMMETRIC_REVERSE_LABELS,
        minTotalBoxesForSymmetricTrigger = 2,
      ),
      digits = SuppliedYoloDecoder.DigitsModelConfig(
        file = "digits.onnx",
        sha256 = "f".repeat(64),
        sizeBytes = 12_135_238L,
        imgsz = 320,
        inputShape = listOf(1L, 3L, 320L, 320L),
        outputShape = listOf(1L, 34L, 2100L),
        confThres = 0.25f,
        iouThres = 0.50f,
        maxDet = 300,
        agnosticNms = true,
        names = SuppliedYoloDecoder.DEFAULT_DIGITS_NAMES,
        classToDigit = SuppliedYoloDecoder.DEFAULT_CLASS_TO_DIGIT,
      ),
      geometryBounds = SuppliedYoloDecoder.GeometryBounds(),
      acceptedLengths = SuppliedYoloDecoder.DEFAULT_ACCEPTED_LENGTHS,
      preserveLeadingZeros = true,
    )
    SuppliedYoloDecoder.validateConfig(valid)

    val badVariants = listOf(
      valid.copy(pipelineId = "yolo11n-ppocr-v4-research-v1"),
      valid.copy(frame = valid.frame.copy(outputShape = listOf(1L, 5L, 3549L))),
      valid.copy(reverse = valid.reverse.copy(names = mapOf(0 to "0n"))),
      valid.copy(digits = valid.digits.copy(classToDigit = valid.digits.classToDigit + ("13" to "9"))),
    )
    for (bad in badVariants) {
      var threw = false
      try {
        SuppliedYoloDecoder.validateConfig(bad)
      } catch (_: IllegalArgumentException) {
        threw = true
      }
      assertTrue("Expected validateConfig to reject tampered variant: $bad", threw)
    }
  }

  @Test
  fun decodeDigitsReadingPreservesLeadingZerosTransitionClassesAndLeftToRightOrder() {
    // Out-of-order detections with transition classes: "10"->"0", "0"->"0", "12"->"2", "23"->"3", "17"->"7"
    val detections = listOf(
      det(anchor = 40, classId = 17, label = "17", score = 0.88f, x1 = 92f),
      det(anchor = 10, classId = 10, label = "10", score = 0.91f, x1 = 6f),
      det(anchor = 30, classId = 23, label = "23", score = 0.85f, x1 = 70f),
      det(anchor = 20, classId = 0, label = "0", score = 0.94f, x1 = 28f),
      det(anchor = 25, classId = 12, label = "12", score = 0.89f, x1 = 49f),
    )
    val reading = SuppliedYoloDecoder.decodeDigitsReading(detections)
    assertEquals("ok", reading.status)
    assertEquals("00237", reading.rawDigits)
    assertNull(reading.reason)
    assertEquals(listOf("10", "0", "12", "23", "17"), reading.digitClassesLeftToRight)
  }

  @Test
  fun decodeDigitsReadingAbstainsOnEmptyOrUnexpectedLengthWithoutDroppingDiagnostic() {
    val emptyReading = SuppliedYoloDecoder.decodeDigitsReading(emptyList())
    assertEquals("uncertain", emptyReading.status)
    assertNull(emptyReading.rawDigits)
    assertEquals("no_characters", emptyReading.reason)

    val fourDigits = (0 until 4).map { i -> det(i, i, i.toString(), 0.9f, x1 = i * 20f) }
    val shortReading = SuppliedYoloDecoder.decodeDigitsReading(fourDigits)
    assertEquals("uncertain", shortReading.status)
    assertNull(shortReading.rawDigits)
    assertEquals("unexpected_length", shortReading.reason)
    assertEquals("0123", shortReading.rawDiagnostic)

    val sevenDigits = (0 until 7).map { i -> det(i, i, i.toString(), 0.9f, x1 = i * 20f) }
    val longReading = SuppliedYoloDecoder.decodeDigitsReading(sevenDigits)
    assertEquals("uncertain", longReading.status)
    assertNull(longReading.rawDigits)
    assertEquals("unexpected_length", longReading.reason)
    assertEquals("0123456", longReading.rawDiagnostic)
  }

  @Test
  fun evaluateReverseTriggerFollowsApprovedSixClassPolicy() {
    val none = SuppliedYoloDecoder.evaluateReverseTrigger(emptyList())
    assertFalse(none.isReversed)
    assertEquals("no_reverse_boxes", none.reason)

    val single1n = SuppliedYoloDecoder.evaluateReverseTrigger(listOf(det(1, 0, "1n", 0.72f, 10f)))
    assertFalse(single1n.isReversed)
    assertEquals("single_symmetric_1n_ignored", single1n.reason)

    for ((idx, asym) in listOf("2n", "3n", "4n", "5n", "7n").withIndex()) {
      val res = SuppliedYoloDecoder.evaluateReverseTrigger(listOf(det(idx, idx + 1, asym, 0.45f, 15f)))
      assertTrue("Expected $asym to trigger reverse", res.isReversed)
      assertEquals("asymmetric_reverse_label", res.reason)
    }

    val two1n = SuppliedYoloDecoder.evaluateReverseTrigger(
      listOf(det(1, 0, "1n", 0.65f, 10f), det(2, 0, "1n", 0.58f, 35f))
    )
    assertTrue(two1n.isReversed)
    assertEquals("multiple_reverse_boxes", two1n.reason)

    var threwOnUnsupported = false
    try {
      SuppliedYoloDecoder.evaluateReverseTrigger(listOf(det(1, 9, "6n", 0.8f, 10f)))
    } catch (_: IllegalArgumentException) {
      threwOnUnsupported = true
    }
    assertTrue(threwOnUnsupported)
  }

  @Test
  fun orderObbCornersEnforcesLongHorizontalEdgeAndPositiveChirality() {
    val base = arrayOf(
      floatArrayOf(100f, 200f),
      floatArrayOf(280f, 185f),
      floatArrayOf(284f, 230f),
      floatArrayOf(104f, 245f),
    )
    for (shift in 0..3) {
      val cycled = Array(4) { i -> base[(i + shift) % 4] }
      val ordered = SuppliedYoloDecoder.orderObbCorners(cycled)
      val w = hypot((ordered[1][0] - ordered[0][0]).toDouble(), (ordered[1][1] - ordered[0][1]).toDouble())
      val h = hypot((ordered[3][0] - ordered[0][0]).toDouble(), (ordered[3][1] - ordered[0][1]).toDouble())
      val cross = (ordered[1][0] - ordered[0][0]) * (ordered[3][1] - ordered[0][1]) -
        (ordered[1][1] - ordered[0][1]) * (ordered[3][0] - ordered[0][0])
      assertTrue(w >= h)
      assertTrue(cross > 0f)
      assertTrue(ordered[1][0] > ordered[0][0])

      val mirrored = arrayOf(cycled[0], cycled[3], cycled[2], cycled[1])
      val orderedMirrored = SuppliedYoloDecoder.orderObbCorners(mirrored)
      val crossMirrored = (orderedMirrored[1][0] - orderedMirrored[0][0]) * (orderedMirrored[3][1] - orderedMirrored[0][1]) -
        (orderedMirrored[1][1] - orderedMirrored[0][1]) * (orderedMirrored[3][0] - orderedMirrored[0][0])
      assertTrue(crossMirrored > 0f)
      assertTrue(orderedMirrored[1][0] > orderedMirrored[0][0])
    }
  }

  @Test
  fun planFrameCropHandlesNoRowMultipleRowsDegenerateAndClampedAabb() {
    assertEquals("no_row", SuppliedYoloDecoder.planFrameCrop(480, 640, emptyList()).reason)

    val validPoly = arrayOf(
      floatArrayOf(120.2f, 210.4f),
      floatArrayOf(310.7f, 212.1f),
      floatArrayOf(309.9f, 255.8f),
      floatArrayOf(119.5f, 254.0f),
    )
    val frame = SuppliedYoloDecoder.FrameDetection(
      anchor = 100,
      classId = 0,
      label = "khung",
      score = 0.92f,
      netObb = floatArrayOf(215f, 233f, 190f, 44f, 0.01f),
      polygon = validPoly,
    )
    assertEquals("multiple_rows", SuppliedYoloDecoder.planFrameCrop(480, 640, listOf(frame, frame)).reason)

    val singlePlan = SuppliedYoloDecoder.planFrameCrop(480, 640, listOf(frame))
    assertNull(singlePlan.reason)
    assertArrayEquals(intArrayOf(119, 210, 311, 256), singlePlan.rowBox)
    assertEquals(191, singlePlan.cropWidth)
    assertEquals(44, singlePlan.cropHeight)

    val tinyPoly = arrayOf(
      floatArrayOf(10f, 10f),
      floatArrayOf(18f, 10f),
      floatArrayOf(18f, 14f),
      floatArrayOf(10f, 14f),
    )
    val tinyFrame = frame.copy(polygon = tinyPoly)
    assertEquals("unusable_row_geometry", SuppliedYoloDecoder.planFrameCrop(480, 640, listOf(tinyFrame)).reason)
  }

  @Test
  fun rotateRgb180ReversesPixelOrderWhilePreservingChannels() {
    // 2x2 RGB image: pixel0=(1,2,3), pixel1=(4,5,6), pixel2=(7,8,9), pixel3=(10,11,12)
    val src = byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    val rot = SuppliedYoloDecoder.rotateRgb180(src, 2, 2)
    assertArrayEquals(byteArrayOf(10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3), rot)
    assertArrayEquals(src, SuppliedYoloDecoder.rotateRgb180(rot, 2, 2))
  }

  private fun readFloat32Le(file: File): FloatArray {
    val bytes = file.readBytes()
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    val out = FloatArray(buf.remaining())
    buf.get(out)
    return out
  }

  @Test
  fun hostStagingTensorsDecodeToExactApprovedFixtureReadings() {
    val repoRoot = File("../..").canonicalFile
    val tensorsDir = File(repoRoot, "artifacts/supplied-yolo/host-report/tensors")
    if (!tensorsDir.exists()) return

    data class ExpectedFixture(
      val id: String,
      val width: Int,
      val height: Int,
      val status: String,
      val rawDigits: String?,
      val reason: String?,
      val isReversed: Boolean,
      val rowBox: IntArray?,
      val cropW: Int,
      val cropH: Int,
    )

    val fixtures = listOf(
      ExpectedFixture("meter-clean-01", 480, 640, "ok", "00237", null, false, intArrayOf(196, 267, 323, 303), 127, 29),
      ExpectedFixture("meter-clean-02", 472, 1047, "ok", "034184", null, false, intArrayOf(188, 396, 336, 434), 147, 34),
      ExpectedFixture("meter-tilted-01", 960, 1280, "ok", "02660", null, false, intArrayOf(289, 513, 667, 738), 413, 109),
      ExpectedFixture("meter-upside-down-01", 480, 640, "ok", "00237", null, true, intArrayOf(156, 336, 283, 372), 121, 28),
      ExpectedFixture("meter-transition-01", 726, 987, "ok", "01563", null, false, intArrayOf(270, 567, 500, 623), 229, 51),
      ExpectedFixture("meter-transition-02", 480, 640, "ok", "004387", null, false, intArrayOf(186, 276, 374, 331), 184, 43),
      ExpectedFixture("meter-blank-01", 480, 640, "uncertain", null, "no_row", false, null, 0, 0),
      ExpectedFixture("meter-multi-01", 960, 640, "uncertain", null, "multiple_rows", false, null, 0, 0),
    )

    for (fx in fixtures) {
      val frameOutFile = File(tensorsDir, "${fx.id}.frame_output.f32")
      assertTrue("Missing tensor ${frameOutFile.absolutePath}", frameOutFile.exists())
      val frameMeta = SuppliedYoloDecoder.letterboxMeta(fx.width, fx.height, SuppliedYoloDecoder.FRAME_IMG_SIZE)
      val frameDetections = SuppliedYoloDecoder.decodeFrameObb(readFloat32Le(frameOutFile), frameMeta)
      val cropPlan = SuppliedYoloDecoder.planFrameCrop(fx.width, fx.height, frameDetections)

      if (fx.rowBox == null) {
        assertEquals(fx.id, fx.reason, cropPlan.reason)
        assertNull(fx.id, cropPlan.rowBox)
        continue
      }

      assertNull(fx.id, cropPlan.reason)
      assertNotNull(fx.id, cropPlan.rowBox)
      assertArrayEquals(fx.id, fx.rowBox, cropPlan.rowBox)
      assertEquals(fx.id, fx.cropW, cropPlan.cropWidth)
      assertEquals(fx.id, fx.cropH, cropPlan.cropHeight)

      val revMeta = SuppliedYoloDecoder.letterboxMeta(cropPlan.cropWidth, cropPlan.cropHeight, SuppliedYoloDecoder.DETECT_IMG_SIZE)
      val revOutFile = File(tensorsDir, "${fx.id}.reverse_output.f32")
      val revDetections = SuppliedYoloDecoder.decodeAxisAlignedBoxes(
        rawOutput = readFloat32Le(revOutFile),
        anchors = SuppliedYoloDecoder.DETECT_ANCHORS,
        meta = revMeta,
        names = SuppliedYoloDecoder.DEFAULT_REVERSE_NAMES,
      )
      val revDecision = SuppliedYoloDecoder.evaluateReverseTrigger(revDetections)
      assertEquals(fx.id, fx.isReversed, revDecision.isReversed)

      val digitsMeta = SuppliedYoloDecoder.letterboxMeta(cropPlan.cropWidth, cropPlan.cropHeight, SuppliedYoloDecoder.DETECT_IMG_SIZE)
      val digitsOutFile = File(tensorsDir, "${fx.id}.digits_output.f32")
      val digitDetections = SuppliedYoloDecoder.decodeAxisAlignedBoxes(
        rawOutput = readFloat32Le(digitsOutFile),
        anchors = SuppliedYoloDecoder.DETECT_ANCHORS,
        meta = digitsMeta,
        names = SuppliedYoloDecoder.DEFAULT_DIGITS_NAMES,
      )
      val reading = SuppliedYoloDecoder.decodeDigitsReading(digitDetections)
      assertEquals(fx.id, fx.status, reading.status)
      assertEquals(fx.id, fx.rawDigits, reading.rawDigits)
      assertEquals(fx.id, fx.reason, reading.reason)
    }
  }
}
