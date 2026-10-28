package com.watermeterocr.ocr

import org.junit.Assert.*
import org.junit.Test

class ReadingQualityTest {
  private val polygon = arrayOf(
    floatArrayOf(100f, 100f), floatArrayOf(700f, 100f),
    floatArrayOf(700f, 200f), floatArrayOf(100f, 200f),
  )
  private val crop = SuppliedYoloDecoder.CropPlan(null, intArrayOf(100, 100, 700, 200), polygon, 600, 100)
  private fun frames(score: Float = 0.99f, points: Array<FloatArray> = polygon) = listOf(
    SuppliedYoloDecoder.FrameDetection(0, 0, "khung", score, floatArrayOf(400f, 150f, 600f, 100f, 0f), points),
  )

  private fun reading(digits: String = "085918", scores: List<Float> = List(digits.length) { 0.99f }): SuppliedYoloDecoder.DigitReading {
    val detections = digits.mapIndexed { index, digit ->
      val left = 10f + index * 90f
      SuppliedYoloDecoder.BoxDetection(index, digit - '0', digit.toString(), scores[index],
        floatArrayOf(left, 10f, left + 60f, 90f), left + 30f, 50f, digit.toString())
    }
    return SuppliedYoloDecoder.DigitReading("ok", digits, null, scores.map { it.toDouble() }.average(),
      digits, digits.map { it.toString() }, detections)
  }

  private fun evaluate(
    value: SuppliedYoloDecoder.DigitReading = reading(),
    plan: SuppliedYoloDecoder.CropPlan = crop,
    rows: List<SuppliedYoloDecoder.FrameDetection> = frames(),
  ) = ReadingQuality.evaluate(value, plan, rows, 800, 400)

  @Test fun completeStrongReadingHasHighQualityAndNormalizedBox() {
    val result = evaluate()
    assertEquals(ReadingQualityLevel.HIGH, result.quality)
    assertEquals("085918", result.digits)
    assertArrayEquals(floatArrayOf(0.125f, 0.25f, 0.875f, 0.5f), result.box, 0.00001f)
  }

  @Test fun acceptedFiveDigitReadingRemainsStringWithLeadingZeros() {
    val result = evaluate(reading("00237"))
    assertEquals(ReadingQualityLevel.HIGH, result.quality)
    assertEquals("00237", result.digits)
  }

  @Test fun completeHighQualityCameraFrameFreezesOnItsFirstObservation() {
    val observation = ReadingQuality.evaluate(reading(), crop, frames(), 1280, 960)
    assertEquals(ReadingQualityLevel.HIGH, observation.quality)
    assertTrue("Passing camera geometry and quality must not add a second-frame delay",
      ReadingStability().observe(observation, 100_000_000, 100))
  }

  @Test fun moderateScoresRequireTemporalConfirmation() {
    assertEquals(ReadingQualityLevel.MODERATE,
      evaluate(reading(scores = List(6) { 0.85f }), rows = frames(0.75f)).quality)
  }

  @Test fun floatThresholdEqualityQualifiesAtBothQualityLevels() {
    assertEquals(ReadingQualityLevel.HIGH,
      evaluate(reading(scores = List(6) { 0.95f }), rows = frames(0.85f)).quality)
    assertEquals(ReadingQualityLevel.MODERATE,
      evaluate(reading(scores = List(6) { 0.80f }), rows = frames(0.60f)).quality)
  }

  @Test fun weakDigitCannotHideBehindStrongMean() {
    val result = evaluate(reading(scores = listOf(0.99f, 0.99f, 0.99f, 0.40f, 0.99f, 0.99f)))
    assertEquals(ReadingQualityLevel.LOW, result.quality)
  }

  @Test fun weakRowCannotUseStrongDigitScoresToFreeze() {
    assertEquals(ReadingQualityLevel.LOW, evaluate(rows = frames(0.40f)).quality)
  }

  @Test fun nonFiniteOrOutOfRangeConfidenceNeverQualifies() {
    for (score in listOf(Float.NaN, Float.POSITIVE_INFINITY, -0.1f, 1.1f)) {
      assertEquals(ReadingQualityLevel.INVALID,
        evaluate(reading(scores = listOf(score, 0.99f, 0.99f, 0.99f, 0.99f, 0.99f))).quality)
      assertEquals(ReadingQualityLevel.INVALID, evaluate(rows = frames(score)).quality)
    }
  }

  @Test fun uncertainDigitsWithValidRowRetainOnlyLowQualityTarget() {
    val value = reading().copy(status = "uncertain", rawDigits = null, reason = "no_characters", sortedDetections = emptyList())
    val result = evaluate(value)
    assertEquals(ReadingQualityLevel.LOW, result.quality)
    assertNotNull(result.box)
    assertNull(result.digits)
  }

  @Test fun missingAndMultipleRowsInvalidateTarget() {
    assertEquals(ReadingQualityLevel.INVALID, evaluate(rows = emptyList()).quality)
    assertEquals(ReadingQualityLevel.INVALID, evaluate(rows = frames() + frames()).quality)
  }

  @Test fun originalPolygonOutsideImageIsRejectedDespiteClampedRowBox() {
    val clipped = polygon.map { it.copyOf() }.toTypedArray().also { it[0][0] = -20f }
    val result = evaluate(plan = crop.copy(rowBox = intArrayOf(0, 100, 700, 200), orderedPolygon = clipped), rows = frames(points = clipped))
    assertEquals(ReadingQualityLevel.INVALID, result.quality)
  }

  @Test fun malformedOrNonFinitePolygonInvalidatesTarget() {
    for (points in listOf(arrayOf(floatArrayOf(1f)), polygon.map { it.copyOf() }.toTypedArray().also { it[1][0] = Float.NaN })) {
      assertEquals(ReadingQualityLevel.INVALID,
        evaluate(plan = crop.copy(orderedPolygon = points), rows = frames(points = points)).quality)
    }
  }

  @Test fun degenerateAndSelfCrossingPolygonsInvalidateTarget() {
    val collapsed = arrayOf(floatArrayOf(100f, 100f), floatArrayOf(300f, 100f),
      floatArrayOf(500f, 100f), floatArrayOf(700f, 100f))
    val crossing = arrayOf(polygon[0], polygon[2], polygon[1], polygon[3])
    for (points in listOf(collapsed, crossing)) {
      assertEquals(ReadingQualityLevel.INVALID,
        evaluate(plan = crop.copy(orderedPolygon = points), rows = frames(points = points)).quality)
    }
  }

  @Test fun invalidCropOrRowGeometryInvalidatesTarget() {
    for (plan in listOf(
      crop.copy(reason = "unusable_row_geometry"), crop.copy(cropWidth = 0),
      crop.copy(rowBox = intArrayOf(700, 100, 100, 200)), crop.copy(rowBox = null),
    )) assertEquals(ReadingQualityLevel.INVALID, evaluate(plan = plan).quality)
  }

  @Test fun clippedOrMalformedCharacterCannotQualify() {
    val original = reading()
    for (box in listOf(floatArrayOf(-10f, 10f, 60f, 90f), floatArrayOf(10f, 10f, 610f, 90f),
      floatArrayOf(10f, 10f, 5f, 90f), floatArrayOf(Float.NaN, 10f, 60f, 90f), floatArrayOf(1f))) {
      val changed = original.copy(sortedDetections = original.sortedDetections.mapIndexed { index, detection ->
        if (index == 0) detection.copy(box = box) else detection
      })
      assertEquals(ReadingQualityLevel.INVALID, evaluate(changed).quality)
    }
  }

  @Test fun inconsistentDigitCountOrUnsupportedLengthCannotQualify() {
    val original = reading()
    for (value in listOf(original.copy(sortedDetections = original.sortedDetections.dropLast(1)),
      original.copy(rawDigits = "1234"), original.copy(rawDigits = "1234567"), original.copy(rawDigits = "abc123"))) {
      assertEquals(ReadingQualityLevel.INVALID, evaluate(value).quality)
    }
  }

  @Test fun malformedSameRowEvidenceClearsPreviouslyCredibleCandidate() {
    val valid = reading(scores = List(6) { 0.85f })
    val credible = evaluate(valid, rows = frames(0.75f))
    val malformed = valid.copy(sortedDetections = valid.sortedDetections.mapIndexed { index, detection ->
      if (index == 0) detection.copy(box = floatArrayOf(-10f, 10f, 60f, 90f)) else detection
    })
    val invalid = evaluate(malformed, rows = frames(0.75f))
    assertEquals(ReadingQualityLevel.MODERATE, credible.quality)
    assertEquals(ReadingQualityLevel.INVALID, invalid.quality)
    val gate = ReadingStability()
    assertFalse(gate.observe(credible, 100_000_000, 100))
    assertFalse(gate.observe(invalid, 200_000_000, 200))
    assertFalse("Malformed evidence must erase the first vote", gate.observe(credible, 300_000_000, 300))
    assertTrue(gate.observe(credible, 400_000_000, 400))
  }

  @Test fun transitioningRollerDigitQualifiesAsModerate() {
    val transitioning = reading("085914", listOf(0.96f, 0.95f, 0.94f, 0.96f, 0.93f, 0.42f)).let { base ->
      base.copy(
        digitClassesLeftToRight = listOf("0", "8", "5", "9", "1", "14"),
        sortedDetections = base.sortedDetections.mapIndexed { idx, det ->
          if (idx == 5) det.copy(classId = 14, label = "14") else det
        }
      )
    }
    assertEquals(ReadingQualityLevel.MODERATE, evaluate(transitioning).quality)
  }

  @Test fun fiveDigitReadingWithMissingSixthSlotOnCameraFrameIsDowngradedToLow() {
    val partialFive = reading("08591")
    // On a 1280x960 camera frame, 5 digits spanning x=10..430 in a 600px crop leaves a 170px empty slot on the right.
    val result = ReadingQuality.evaluate(partialFive, crop, frames(), 1280, 960)
    assertEquals(ReadingQualityLevel.LOW, result.quality)
  }
}
