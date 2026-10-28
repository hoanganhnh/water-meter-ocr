package com.watermeterocr.ocr

/** Retain a clean HIGH frame immediately; uncertain or conflicting readings need consensus. */
internal class ReadingStability {
  private data class Sample(
    val digits: String,
    val quality: ReadingQualityLevel,
    val captureNs: Long,
  )

  private val history = ArrayDeque<Sample>()
  private var anchor: FloatArray? = null
  private var lastBox: FloatArray? = null
  private var seenSixDigits = false
  private var lastQualifiedMs = 0L
  private var lastCaptureNs = 0L
  private var lastNowMs = -1L
  var firstCredibleAtMs: Long? = null
    private set
  var pendingDigits: String? = null
    private set

  @Synchronized fun reset() {
    history.clear()
    anchor = null
    lastBox = null
    seenSixDigits = false
    lastQualifiedMs = 0L
    lastCaptureNs = 0L
    lastNowMs = -1L
    firstCredibleAtMs = null
    pendingDigits = null
  }

  @Synchronized fun observe(
    observation: ReadingObservation,
    capturedAtNs: Long,
    nowMs: Long,
    pairWindowMs: Long = 3000,
  ): Boolean {
    if (capturedAtNs <= 0 || nowMs < 0 || nowMs < lastNowMs || capturedAtNs < lastCaptureNs) {
      reset(); return false
    }
    if (capturedAtNs == lastCaptureNs) return false
    lastCaptureNs = capturedAtNs; lastNowMs = nowMs
    val box = observation.box
    if (observation.quality == ReadingQualityLevel.INVALID || !validBox(box)) {
      reset(); return false
    }
    box!!
    val windowMs = pairWindowMs.coerceIn(3000, 10000)
    // Require both anchor overlap and frame-to-frame steadiness so moving hands cannot accumulate votes.
    if (anchor?.let { overlap(it, box) < 0.55f } == true ||
      lastBox?.let { overlap(it, box) < 0.65f } == true ||
      (history.isNotEmpty() && (capturedAtNs - history.last().captureNs) / 1_000_000 > windowMs)) {
      clearCandidate()
    }
    anchor = anchor ?: box.copyOf()
    lastBox = box.copyOf()
    if (observation.quality == ReadingQualityLevel.LOW) {
      // Only an observed, geometrically valid same row can bridge a brief OCR miss.
      if (history.isNotEmpty() && nowMs - lastQualifiedMs > 1200) clearCandidate()
      return false
    }
    val digits = observation.digits
    if (digits == null || !digits.matches(DIGITS)) { reset(); return false }

    if (digits.length == 6) {
      seenSixDigits = true
      history.removeAll { it.digits.length < 6 }
    } else if (seenSixDigits) {
      // Never allow a 5-digit partial reading to overwrite or freeze a known 6-digit meter window.
      return false
    }

    while (history.isNotEmpty() && (capturedAtNs - history.first().captureNs) / 1_000_000 > windowMs) {
      history.removeFirst()
    }
    if (firstCredibleAtMs == null) firstCredibleAtMs = nowMs
    history.addLast(Sample(digits, observation.quality, capturedAtNs))
    while (history.size > MAX_HISTORY) history.removeFirst()
    lastQualifiedMs = nowMs
    pendingDigits = digits

    val exactMatches = history.count { it.digits == digits }
    val hadConflict = history.any { it.digits != digits }
    val consecutiveMatch = history.size >= 2 &&
      history[history.size - 1].digits == digits &&
      history[history.size - 2].digits == digits

    // HIGH already passed the per-digit and geometry gates. Only uncertain or conflicting
    // evidence needs another inference; do not impose a second-frame delay on a clean reading.
    return if (!hadConflict) {
      observation.quality == ReadingQualityLevel.HIGH || exactMatches >= 2
    } else {
      (consecutiveMatch && (observation.quality == ReadingQualityLevel.HIGH || exactMatches >= 3)) ||
        (exactMatches >= 3 && exactMatches * 2 > history.size)
    }
  }

  private fun clearCandidate() {
    history.clear()
    anchor = null
    lastBox = null
    seenSixDigits = false
    lastQualifiedMs = 0L
    firstCredibleAtMs = null
    pendingDigits = null
  }

  private fun validBox(box: FloatArray?): Boolean = box != null && box.size == 4 &&
    box.all { it.isFinite() } && box[0] >= 0 && box[1] >= 0 && box[2] <= 1 && box[3] <= 1 &&
    box[2] > box[0] && box[3] > box[1]

  private fun overlap(a: FloatArray, b: FloatArray): Float {
    val area = maxOf(0f, minOf(a[2], b[2]) - maxOf(a[0], b[0])) *
      maxOf(0f, minOf(a[3], b[3]) - maxOf(a[1], b[1]))
    return area / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area)
  }

  private companion object {
    const val MAX_HISTORY = 5
    val DIGITS = Regex("[0-9]{5,6}")
  }
}
