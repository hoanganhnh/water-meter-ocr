package com.watermeterocr.ocr

import org.junit.Assert.*
import org.junit.Test

class AnalysisCadenceTest {
  @Test fun initialCadenceReservesStartButFirstFrameIsImmediate() {
    val cadence = AnalysisCadence()
    assertEquals(200L, cadence.intervalMs)
    assertTrue(cadence.shouldAnalyze(100))
    assertFalse(cadence.shouldAnalyze(101))
    assertTrue(cadence.shouldAnalyze(300))
  }

  @Test fun resetDoesNotDelayFirstFrameWithPreviousReservation() {
    val cadence = AnalysisCadence()
    assertTrue(cadence.shouldAnalyze(100))
    cadence.recordDuration(3000)
    cadence.reset()
    assertTrue(cadence.shouldAnalyze(101))
    assertEquals(200L, cadence.intervalMs)
  }

  @Test fun fastFramesDoNotExceedMinimumInterval() {
    val cadence = AnalysisCadence()
    repeat(20) { cadence.recordDuration(1) }
    assertEquals(150L, cadence.intervalMs)
    assertEquals(3000L, cadence.pairWindowMs)
  }

  @Test fun slowInferenceDominatesCadenceAndExpandsPairWindow() {
    val cadence = AnalysisCadence()
    repeat(20) { cadence.recordDuration(2500) }
    assertEquals(400L, cadence.intervalMs)
    assertTrue(cadence.pairWindowMs > 6000)
    assertTrue(cadence.shouldAnalyze(100))
    assertTrue(cadence.shouldAnalyze(2600))
  }

  @Test fun arbitrarilySlowFramesCannotGrowPairWindowBeyondBound() {
    val cadence = AnalysisCadence()
    repeat(20) { cadence.recordDuration(30000) }
    assertEquals(400L, cadence.intervalMs)
    assertEquals(10000L, cadence.pairWindowMs)
  }

  @Test fun recoveryFromSlowRuntimeAdaptsBackToFastCadence() {
    val cadence = AnalysisCadence()
    repeat(10) { cadence.recordDuration(3000) }
    assertEquals(400L, cadence.intervalMs)
    repeat(40) { cadence.recordDuration(20) }
    assertEquals(150L, cadence.intervalMs)
    assertEquals(3000L, cadence.pairWindowMs)
  }
}
