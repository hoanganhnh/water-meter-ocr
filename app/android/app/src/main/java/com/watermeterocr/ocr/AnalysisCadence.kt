package com.watermeterocr.ocr

/** Worker-owned start cadence; CameraX still keeps only its latest waiting frame. */
internal class AnalysisCadence {
  private var averageMs: Double? = null
  private var nextAtMs = 0L
  val intervalMs: Long get() = averageMs?.let { (it * 1.25).toLong().coerceIn(150, 400) } ?: 200
  val pairWindowMs: Long get() = averageMs?.let { (it * 3).toLong().coerceIn(3000, 10000) } ?: 3000

  fun reset() { averageMs = null; nextAtMs = 0L }
  fun shouldAnalyze(nowMs: Long): Boolean {
    if (nowMs < nextAtMs) return false
    nextAtMs = nowMs + intervalMs
    return true
  }
  fun recordDuration(durationMs: Long) {
    if (durationMs < 0) return
    averageMs = averageMs?.let { it * 0.75 + durationMs * 0.25 } ?: durationMs.toDouble()
  }
}
