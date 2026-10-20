package com.watermeterocr.ocr

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class RgbBilinearTest {
  @Test fun matchesIndependentPillowFixturesForUpDownIdentityAndRounding() {
    val source = javaClass.getResourceAsStream("/resize/source.rgb")!!.use { it.readBytes() }
    for ((w, h) in listOf(31 to 19, 7 to 5, 23 to 13, 192 to 109, 12 to 6)) {
      val expected = javaClass.getResourceAsStream("/resize/${w}x${h}.rgb")!!.use { it.readBytes() }
      val result = RgbBilinear.resize(23, 13, w, h) { y, row ->
        for (x in row.indices) {
          val i = (y * 23 + x) * 3
          row[x] = ((source[i].toInt() and 255) shl 16) or ((source[i + 1].toInt() and 255) shl 8) or (source[i + 2].toInt() and 255)
        }
      }
      assertArrayEquals("Pillow RGB ${w}x${h}", expected, result)
    }
  }
}
