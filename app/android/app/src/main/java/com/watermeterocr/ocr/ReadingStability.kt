package com.watermeterocr.ocr

/** Bounded voting for the same visible row. Time is monotonic milliseconds. */
internal class ReadingStability {
  private data class Sample(val digits: String, val time: Long)
  private val samples = ArrayDeque<Sample>()
  private var anchor: FloatArray? = null
  private var lastTime = 0L

  @Synchronized fun reset() { samples.clear(); anchor = null; lastTime = 0L }

  @Synchronized fun observe(digits: String?, box: FloatArray?, now: Long): Boolean {
    if (digits == null || !digits.matches(Regex("[0-9]{5,6}")) || box == null ||
      box.size != 4 || box.any { !it.isFinite() } || box[0] < 0 || box[1] < 0 ||
      box[2] > 1 || box[3] > 1 || box[2] <= box[0] || box[3] <= box[1]) {
      reset(); return false
    }
    if (now < lastTime || now - lastTime > 1800 || anchor?.let { overlap(it, box) < 0.5f } == true) reset()
    anchor = anchor ?: box.copyOf()
    lastTime = now
    while (samples.isNotEmpty() && now - samples.first().time > 3000) samples.removeFirst()
    samples.addLast(Sample(digits, now))
    while (samples.size > 5) samples.removeFirst()
    val matches = samples.filter { it.digits == digits }
    return matches.size >= 3 && now - matches.first().time >= 800 &&
      matches.size * 2 > samples.size
  }

  private fun overlap(a: FloatArray, b: FloatArray): Float {
    val area = maxOf(0f, minOf(a[2], b[2]) - maxOf(a[0], b[0])) *
      maxOf(0f, minOf(a[3], b[3]) - maxOf(a[1], b[1]))
    return area / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area)
  }
}
