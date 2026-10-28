package com.watermeterocr.ocr

import org.junit.Assert.*
import org.junit.Test

class ReadingStabilityTest {
  private val row = floatArrayOf(0.1f, 0.2f, 0.8f, 0.4f)
  private fun observation(
    digits: String? = "085918",
    quality: ReadingQualityLevel = ReadingQualityLevel.MODERATE,
    box: FloatArray? = row,
  ) = ReadingObservation(box, digits, quality)

  private fun ReadingStability.at(
    milliseconds: Long,
    sample: ReadingObservation = observation(),
    completedAt: Long = milliseconds,
    pairWindow: Long = 3000,
  ) = observe(sample, milliseconds * 1_000_000, completedAt, pairWindow)

  @Test fun firstHighQualityReadingFreezesWithoutWaitingForAnotherFrame() {
    val gate = ReadingStability()
    assertTrue("A fresh, fully qualified high-quality frame must freeze immediately",
      gate.at(100, observation(quality = ReadingQualityLevel.HIGH)))
    assertEquals(100L, gate.firstCredibleAtMs)
  }

  @Test fun firstHighQualityFiveDigitReadingPreservesLeadingZeros() {
    assertTrue(ReadingStability().at(100, observation("00237", ReadingQualityLevel.HIGH)))
  }

  @Test fun unreadableFramesDoNotDelayFirstCredibleHighQualityReading() {
    val gate = ReadingStability()
    assertFalse(gate.at(100, observation(null, ReadingQualityLevel.LOW)))
    assertTrue(gate.at(200, observation(quality = ReadingQualityLevel.HIGH)))
  }

  @Test fun sixDigitTargetBlocksSubsequentFiveDigitPartialReadings() {
    val gate = ReadingStability()
    assertFalse(gate.at(100, observation("085918", ReadingQualityLevel.MODERATE)))
    assertFalse(gate.at(200, observation("08591", ReadingQualityLevel.HIGH)))
    assertFalse(gate.at(300, observation("08591", ReadingQualityLevel.HIGH)))
    assertTrue(gate.at(400, observation("085918", ReadingQualityLevel.MODERATE)))
  }

  @Test fun moderateReadingNeedsTwoDistinctMatchingFramesWithoutMinimumWait() {
    val gate = ReadingStability()
    assertFalse(gate.at(100))
    assertTrue(gate.at(101))
    assertEquals(100L, gate.firstCredibleAtMs)
  }

  @Test fun acceptsFiveDigitsAndPreservesLeadingZeroString() {
    val gate = ReadingStability()
    assertFalse(gate.at(100, observation("00237")))
    assertTrue(gate.at(200, observation("00237")))
  }

  @Test fun lowQualityNeverFreezesEvenWhenRepeated() {
    val gate = ReadingStability()
    for (time in 100L..500L step 100L) {
      assertFalse(gate.at(time, observation(quality = ReadingQualityLevel.LOW)))
    }
    assertNull(gate.firstCredibleAtMs)
  }

  @Test fun briefUnreadableSameRowPreservesCandidateButAddsNoVote() {
    val gate = ReadingStability()
    assertFalse(gate.at(100))
    assertFalse(gate.at(500, observation(null, ReadingQualityLevel.LOW)))
    assertTrue(gate.at(700))
  }

  @Test fun prolongedLowQualityCannotKeepCandidateAlive() {
    val gate = ReadingStability()
    assertFalse(gate.at(100))
    assertFalse(gate.at(900, observation(null, ReadingQualityLevel.LOW)))
    assertFalse(gate.at(1400, observation(null, ReadingQualityLevel.LOW)))
    assertFalse(gate.at(1500))
    assertTrue(gate.at(1600))
  }

  @Test fun invalidOrMissingRowClearsHistory() {
    for (invalid in listOf(
      observation(null, ReadingQualityLevel.INVALID, null),
      observation(null, ReadingQualityLevel.INVALID),
    )) {
      val gate = ReadingStability()
      assertFalse(gate.at(100))
      assertFalse(gate.at(200, invalid))
      assertFalse(gate.at(300))
    }
  }

  @Test fun movedTargetCannotInheritVotes() {
    val gate = ReadingStability()
    val moved = observation(box = floatArrayOf(0.1f, 0.6f, 0.8f, 0.8f))
    assertFalse(gate.at(100))
    assertFalse(gate.at(200, moved))
    assertFalse(gate.at(300))
    assertTrue(gate.at(400))
  }

  @Test fun credibleConflictRequiresAnotherMatchingFrameEvenIfHighQuality() {
    val gate = ReadingStability()
    assertFalse(gate.at(100))
    val different = observation("085913", ReadingQualityLevel.HIGH)
    assertFalse(gate.at(200, different))
    assertTrue(gate.at(300, different))
  }

  @Test fun alternatingStringsCannotBorrowOldVotes() {
    val gate = ReadingStability()
    assertFalse(gate.at(100))
    assertFalse(gate.at(200, observation("085913")))
    assertFalse(gate.at(300))
    assertFalse(gate.at(400, observation("085913")))
  }

  @Test fun slowInferencePairOver1800msCanConfirmWithinAdaptiveWindow() {
    val gate = ReadingStability()
    assertFalse(gate.at(100, completedAt = 2200, pairWindow = 6300))
    assertTrue(gate.at(2300, completedAt = 4400, pairWindow = 6300))
  }

  @Test fun evidenceOlderThanMaximumPairWindowExpires() {
    val gate = ReadingStability()
    assertFalse(gate.at(100, pairWindow = 10000))
    assertFalse(gate.at(10200, pairWindow = 10000))
    assertTrue(gate.at(10300, pairWindow = 10000))
  }

  @Test fun duplicateCaptureCannotBecomeSecondVote() {
    val gate = ReadingStability()
    assertFalse(gate.at(100))
    assertFalse(gate.at(100, completedAt = 200))
    assertTrue(gate.at(300))
  }

  @Test fun outOfOrderCaptureClearsEvidenceAndRejectsFrame() {
    val gate = ReadingStability()
    assertFalse(gate.at(200))
    assertFalse(gate.at(100, completedAt = 300))
    assertFalse(gate.at(400))
    assertTrue(gate.at(500))
  }

  @Test fun backwardsCompletionClockClearsEvidence() {
    val gate = ReadingStability()
    assertFalse(gate.at(100, completedAt = 300))
    assertFalse(gate.at(200, completedAt = 200))
    assertFalse(gate.at(400))
  }

  @Test fun missingCaptureIdentityCannotCountAsEvidence() {
    val gate = ReadingStability()
    assertFalse(gate.observe(observation(), 0, 100))
    assertFalse(gate.observe(observation(quality = ReadingQualityLevel.HIGH), -1, 200))
    assertFalse(gate.at(300))
  }

  @Test fun resetDiscardsCandidateAndTiming() {
    val gate = ReadingStability()
    assertFalse(gate.at(100))
    gate.reset()
    assertNull(gate.firstCredibleAtMs)
    assertFalse(gate.at(200))
  }

  @Test fun malformedNormalizedObservationCannotFreezeOrPreserveVotes() {
    for (box in listOf(null, floatArrayOf(0f), floatArrayOf(0f, 0f, Float.NaN, 1f),
      floatArrayOf(-0.1f, 0.2f, 0.8f, 0.4f), floatArrayOf(0.8f, 0.2f, 0.1f, 0.4f),
      floatArrayOf(0f, 0f, 2f, 1f))) {
      val gate = ReadingStability()
      assertFalse(gate.at(100))
      assertFalse(gate.at(200, observation(quality = ReadingQualityLevel.HIGH, box = box)))
      assertFalse(gate.at(300))
    }
  }
}
