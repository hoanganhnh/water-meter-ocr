package com.watermeterocr.ocr

import org.junit.Assert.*
import org.junit.Test

class ReadingStabilityTest {
  private val row = floatArrayOf(0.1f, 0.2f, 0.8f, 0.4f)

  @Test fun preservesLeadingZeroAndRequiresElapsedEvidence() {
    val gate = ReadingStability()
    assertFalse(gate.observe("085918", row, 100))
    assertFalse(gate.observe("085918", row, 300))
    assertFalse(gate.observe("085918", row, 500))
    assertTrue(gate.observe("085918", row, 1000))
  }

  @Test fun acceptsFiveDigitsAndToleratesOneDisagreement() {
    val gate = ReadingStability()
    assertFalse(gate.observe("00237", row, 100))
    assertFalse(gate.observe("00237", row, 500))
    assertFalse(gate.observe("00287", row, 900))
    assertTrue(gate.observe("00237", row, 1300))
    assertFalse(gate.observe("00287", row, 1700))
  }

  @Test fun invalidOrLostRowClearsHistory() {
    for (digits in listOf(null, "1234", "abcdef", "1234567")) {
      val gate = ReadingStability()
      gate.observe("085918", row, 100); gate.observe("085918", row, 600)
      assertFalse(gate.observe(digits, row, 1000))
      assertFalse(gate.observe("085918", row, 1400))
    }
  }

  @Test fun movedTargetCannotInheritVotes() {
    val gate = ReadingStability()
    gate.observe("085918", row, 100); gate.observe("085918", row, 600)
    assertFalse(gate.observe("085918", floatArrayOf(0.1f, 0.6f, 0.8f, 0.8f), 1100))
    assertFalse(gate.observe("085918", row, 1500))
  }

  @Test fun longGapAndRestartDiscardOldSamples() {
    val gate = ReadingStability()
    gate.observe("085918", row, 100); gate.observe("085918", row, 600)
    assertFalse(gate.observe("085918", row, 3000))
    gate.reset()
    assertFalse(gate.observe("085918", row, 3500))
    assertFalse(gate.observe("085918", row, 100))
  }

  @Test fun malformedGeometryIsRejected() {
    val gate = ReadingStability()
    for (box in listOf(null, floatArrayOf(0f), floatArrayOf(0f,0f,Float.NaN,1f),
      floatArrayOf(0.8f,0.2f,0.1f,0.4f), floatArrayOf(0f,0f,2f,1f))) {
      assertFalse(gate.observe("085918", box, 1000))
    }
  }
}
