package com.watermeterocr.ocr

internal enum class ReadingQualityLevel { INVALID, LOW, MODERATE, HIGH }

internal data class ReadingObservation(
  val box: FloatArray?,
  val digits: String?,
  val quality: ReadingQualityLevel,
)

/** Realtime-only policy. These initial score cutoffs require field calibration. */
internal object ReadingQuality {
  private const val PIXEL_TOLERANCE = 0.5f
  private const val HIGH_MIN_DIGIT = 0.90f
  private const val HIGH_MEAN_DIGIT = 0.95f
  private const val HIGH_ROW = 0.85f
  private const val MODERATE_MIN_DIGIT = 0.65f
  private const val MODERATE_MEAN_DIGIT = 0.80f
  private const val MODERATE_ROW = 0.60f
  private const val TRANSITION_CLASS_MIN_DIGIT = 0.35f
  private const val TRANSITION_ROLLER_MIN_DIGIT = 0.42f
  private const val TRANSITION_OTHER_MIN_DIGIT = 0.78f
  private const val TRANSITION_MEAN_DIGIT = 0.76f
  private const val MIN_CAMERA_CROP_WIDTH = 130
  private const val MIN_CAMERA_CROP_HEIGHT = 24

  fun evaluate(
    reading: SuppliedYoloDecoder.DigitReading,
    crop: SuppliedYoloDecoder.CropPlan,
    frames: List<SuppliedYoloDecoder.FrameDetection>,
    width: Int,
    height: Int,
  ): ReadingObservation {
    val invalid = ReadingObservation(null, null, ReadingQualityLevel.INVALID)
    val row = crop.rowBox ?: return invalid
    val polygon = crop.orderedPolygon ?: return invalid
    if (width <= 0 || height <= 0 || frames.size != 1 || crop.reason != null ||
      crop.cropWidth <= 0 || crop.cropHeight <= 0 || row.size != 4 ||
      row[0] < 0 || row[1] < 0 || row[2] > width || row[3] > height ||
      row[2] <= row[0] || row[3] <= row[1] || polygon.size != 4 ||
      polygon.any { it.size != 2 || !inside(it[0], width) || !inside(it[1], height) } ||
      !convexPolygon(polygon)) return invalid
    val box = floatArrayOf(row[0].toFloat()/width, row[1].toFloat()/height,
      row[2].toFloat()/width, row[3].toFloat()/height)
    val low = ReadingObservation(box, null, ReadingQualityLevel.LOW)
    val rowScore = frames.single().score
    if (!rowScore.isFinite() || rowScore !in 0f..1f) return invalid
    val detections = reading.sortedDetections
    var previousLeft = -Float.MAX_VALUE
    for (det in detections) {
      val b = det.box
      if (!det.score.isFinite() || det.score !in 0f..1f || b.size != 4 ||
        !inside(b[0], crop.cropWidth) || !inside(b[2], crop.cropWidth) ||
        !inside(b[1], crop.cropHeight) || !inside(b[3], crop.cropHeight) ||
        b[2] <= b[0] || b[3] <= b[1] || b[0] < previousLeft) return invalid
      previousLeft = b[0]
    }
    val digits = reading.rawDigits ?: return low
    if (reading.status != "ok" || !digits.matches(Regex("[0-9]{5,6}")) ||
      detections.size != digits.length) return invalid
    val isCameraFrame = width >= 960 || height >= 960
    if (isCameraFrame && (crop.cropWidth < MIN_CAMERA_CROP_WIDTH || crop.cropHeight < MIN_CAMERA_CROP_HEIGHT)) {
      return low
    }
    if (!validDigitRowLayout(detections, crop.cropWidth, crop.cropHeight, isCameraFrame)) {
      return low
    }
    val sortedScores = detections.map { it.score }.sorted()
    val minimum = sortedScores.first()
    val secondMinimum = sortedScores.getOrElse(1) { minimum }
    val mean = detections.sumOf { it.score.toDouble() } / detections.size
    val hasTransitionClass = detections.any { it.classId >= 10 }
    val weakestIsTransitionRoller = hasTransitionClass || detections.last().score == minimum
    val transitioningRollerOk = rowScore >= MODERATE_ROW &&
      secondMinimum >= TRANSITION_OTHER_MIN_DIGIT &&
      mean >= TRANSITION_MEAN_DIGIT.toDouble() &&
      ((hasTransitionClass && minimum >= TRANSITION_CLASS_MIN_DIGIT) ||
        (weakestIsTransitionRoller && minimum >= TRANSITION_ROLLER_MIN_DIGIT) ||
        minimum >= 0.50f)
    val quality = when {
      !hasTransitionClass && minimum >= HIGH_MIN_DIGIT && mean >= HIGH_MEAN_DIGIT.toDouble() && rowScore >= HIGH_ROW -> ReadingQualityLevel.HIGH
      (minimum >= MODERATE_MIN_DIGIT && mean >= MODERATE_MEAN_DIGIT.toDouble() && rowScore >= MODERATE_ROW) || transitioningRollerOk -> ReadingQualityLevel.MODERATE
      else -> ReadingQualityLevel.LOW
    }
    return ReadingObservation(box, digits, quality)
  }

  /**
   * Rejects rows where a middle digit was skipped (leaving a ~2x pitch hole), boxes overlap on the
   * same roller, vertical centers/heights are erratic, or a 5-digit reading on a camera frame
   * leaves an empty 6th-digit slot at the left/right edge of the cropped meter window.
   */
  private fun validDigitRowLayout(
    detections: List<SuppliedYoloDecoder.BoxDetection>,
    cropWidth: Int,
    cropHeight: Int,
    isCameraFrame: Boolean,
  ): Boolean {
    if (detections.size < 5) return false
    val steps = FloatArray(detections.size - 1) { i -> detections[i + 1].cx - detections[i].cx }
    val sortedSteps = steps.sortedArray()
    val medianStep = if (sortedSteps.size % 2 == 1) {
      sortedSteps[sortedSteps.size / 2]
    } else {
      (sortedSteps[sortedSteps.size / 2 - 1] + sortedSteps[sortedSteps.size / 2]) / 2f
    }
    if (!medianStep.isFinite() || medianStep <= 2f) return false

    val widths = detections.map { it.box[2] - it.box[0] }.sorted()
    val heights = detections.map { it.box[3] - it.box[1] }.sorted()
    val cys = detections.map { it.cy }.sorted()
    val medianWidth = widths[widths.size / 2]
    val medianHeight = heights[heights.size / 2]
    val medianCy = cys[cys.size / 2]

    if (steps.any { it < medianWidth * 0.40f || it > medianStep * 1.60f }) return false
    for (det in detections) {
      val h = det.box[3] - det.box[1]
      if (h < medianHeight * 0.45f || h > medianHeight * 1.85f) return false
      if (kotlin.math.abs(det.cy - medianCy) > maxOf(medianHeight * 0.65f, cropHeight * 0.35f)) return false
    }
    if (isCameraFrame && detections.size == 5) {
      val leftMargin = detections.first().box[0]
      val rightMargin = cropWidth.toFloat() - detections.last().box[2]
      if (leftMargin >= medianStep * 0.90f || rightMargin >= medianStep * 0.90f) return false
    }
    return true
  }

  private fun inside(value: Float, limit: Int): Boolean = value.isFinite() &&
    value >= -PIXEL_TOLERANCE && value <= limit + PIXEL_TOLERANCE

  private fun convexPolygon(points: Array<FloatArray>): Boolean {
    var sign = 0f
    for (i in points.indices) {
      val a = points[i]; val b = points[(i + 1) % 4]; val c = points[(i + 2) % 4]
      val cross = (b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0])
      if (!cross.isFinite() || kotlin.math.abs(cross) < 1f || (sign != 0f && sign * cross <= 0f)) return false
      sign = cross
    }
    return true
  }
}
