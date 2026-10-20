package com.watermeterocr.ocr

/** Pillow 12 RGB bilinear convention: normalized 22-bit coefficients, byte rounding per pass.
 * Reference: https://github.com/python-pillow/Pillow/blob/12.0.0/src/libImaging/Resample.c
 */
object RgbBilinear {
  private const val PRECISION = 22
  private const val ONE = 1 shl PRECISION
  private data class Filter(val first: Int, val weights: IntArray)

  private fun filters(sourceSize: Int, targetSize: Int): List<Filter> {
    val scale = sourceSize.toDouble() / targetSize
    val support = maxOf(1.0, scale)
    return List(targetSize) { index ->
      val center = (index + 0.5) * scale
      val first = (center - support + 0.5).toInt().coerceAtLeast(0)
      val end = (center + support + 0.5).toInt().coerceAtMost(sourceSize)
      val raw = DoubleArray(end - first) { offset ->
        (1.0 - kotlin.math.abs((offset + first - center + 0.5) / support)).coerceAtLeast(0.0)
      }
      val total = raw.sum()
      Filter(first, IntArray(raw.size) { (0.5 + raw[it] / total * ONE).toInt() })
    }
  }

  fun resize(width: Int, height: Int, targetWidth: Int, targetHeight: Int, readRow: (Int, IntArray) -> Unit): ByteArray {
    require(width > 0 && height > 0 && targetWidth > 0 && targetHeight > 0)
    val xFilters = filters(width, targetWidth)
    val yFilters = filters(height, targetHeight)
    val horizontal = ByteArray(Math.multiplyExact(Math.multiplyExact(targetWidth, height), 3))
    val row = IntArray(width)
    for (y in 0 until height) {
      readRow(y, row)
      for (x in 0 until targetWidth) {
        val filter = xFilters[x]
        for (channel in 0..2) {
          var sum = ONE / 2L
          filter.weights.forEachIndexed { offset, weight ->
            sum += ((row[filter.first + offset] shr (16 - channel * 8)) and 255).toLong() * weight
          }
          horizontal[(y * targetWidth + x) * 3 + channel] = (sum shr PRECISION).coerceIn(0, 255).toByte()
        }
      }
    }
    val result = ByteArray(targetWidth * targetHeight * 3)
    for (y in 0 until targetHeight) {
      val filter = yFilters[y]
      for (x in 0 until targetWidth) for (channel in 0..2) {
        var sum = ONE / 2L
        filter.weights.forEachIndexed { offset, weight ->
          sum += (horizontal[((filter.first + offset) * targetWidth + x) * 3 + channel].toInt() and 255).toLong() * weight
        }
        result[(y * targetWidth + x) * 3 + channel] = (sum shr PRECISION).coerceIn(0, 255).toByte()
      }
    }
    return result
  }
}
