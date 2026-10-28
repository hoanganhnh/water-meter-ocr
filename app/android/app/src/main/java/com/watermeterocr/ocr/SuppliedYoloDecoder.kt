package com.watermeterocr.ocr

import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pure-math decoder, NMS, corner ordering, reverse orientation policy, and 30-class digit
 * mapping for the approved 3-stage supplied YOLOv8 pipeline (`supplied-yolo-3stage-v1`).
 *
 * Kept free of Android/JNI/OpenCV session state so all geometry and contract semantics are
 * deterministically unit-testable on the host JVM.
 */
object SuppliedYoloDecoder {
  const val PIPELINE_ID = "supplied-yolo-3stage-v1"
  const val SEMANTICS_VERSION = "supplied-yolo-3stage-v1"
  const val PREPROCESS_VERSION = "opencv-linear-byte-letterbox-rgb-div255-pad114-obb-warp-v1"

  const val FRAME_IMG_SIZE = 640
  const val FRAME_CHANNELS = 6
  const val FRAME_ANCHORS = 8400

  const val DETECT_IMG_SIZE = 320
  const val DETECT_ANCHORS = 2100
  const val REVERSE_CLASSES = 6
  const val DIGITS_CLASSES = 30

  val DEFAULT_FRAME_NAMES: Map<Int, String> = mapOf(0 to "khung")
  val DEFAULT_REVERSE_NAMES: Map<Int, String> = mapOf(
    0 to "1n",
    1 to "2n",
    2 to "3n",
    3 to "4n",
    4 to "5n",
    5 to "7n",
  )
  val DEFAULT_DIGITS_NAMES: Map<Int, String> = (0 until DIGITS_CLASSES).associateWith { it.toString() }
  val DEFAULT_CLASS_TO_DIGIT: Map<String, String> = (0 until DIGITS_CLASSES).associate {
    it.toString() to (it % 10).toString()
  }
  val DEFAULT_ASYMMETRIC_REVERSE_LABELS: Set<String> = setOf("2n", "3n", "4n", "5n", "7n")
  val DEFAULT_SYMMETRIC_REVERSE_LABELS: Set<String> = setOf("1n")
  val DEFAULT_ACCEPTED_LENGTHS: Set<Int> = setOf(5, 6)

  data class LetterboxMeta(
    val inputWidth: Int,
    val inputHeight: Int,
    val imgsz: Int,
    val gain: Double,
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
    val resizedWidth: Int,
    val resizedHeight: Int,
  )

  data class GeometryBounds(
    val minWidth: Int = 16,
    val minHeight: Int = 8,
    val minAspect: Double = 1.2,
    val maxAspect: Double = 25.0,
  )

  data class FrameModelConfig(
    val file: String,
    val sha256: String,
    val sizeBytes: Long,
    val imgsz: Int,
    val inputShape: List<Long>,
    val outputShape: List<Long>,
    val confThres: Float,
    val iouThres: Float,
    val maxDet: Int,
    val names: Map<Int, String>,
  )

  data class ReverseModelConfig(
    val file: String,
    val sha256: String,
    val sizeBytes: Long,
    val imgsz: Int,
    val inputShape: List<Long>,
    val outputShape: List<Long>,
    val confThres: Float,
    val iouThres: Float,
    val maxDet: Int,
    val agnosticNms: Boolean,
    val names: Map<Int, String>,
    val asymmetricLabels: Set<String>,
    val symmetricLabels: Set<String>,
    val minTotalBoxesForSymmetricTrigger: Int,
  )

  data class DigitsModelConfig(
    val file: String,
    val sha256: String,
    val sizeBytes: Long,
    val imgsz: Int,
    val inputShape: List<Long>,
    val outputShape: List<Long>,
    val confThres: Float,
    val iouThres: Float,
    val maxDet: Int,
    val agnosticNms: Boolean,
    val names: Map<Int, String>,
    val classToDigit: Map<String, String>,
  )

  data class BundleConfig(
    val schemaVersion: Int,
    val pipelineId: String,
    val semanticsVersion: String,
    val contractStatus: String,
    val runtime: String,
    val preprocessVersion: String,
    val colorOrder: String,
    val padValue: Int,
    val frame: FrameModelConfig,
    val reverse: ReverseModelConfig,
    val digits: DigitsModelConfig,
    val geometryBounds: GeometryBounds,
    val acceptedLengths: Set<Int>,
    val preserveLeadingZeros: Boolean,
  )

  data class FrameDetection(
    val anchor: Int,
    val classId: Int,
    val label: String,
    val score: Float,
    val netObb: FloatArray,
    val polygon: Array<FloatArray>,
  )

  data class CropPlan(
    val reason: String?,
    val rowBox: IntArray?,
    val orderedPolygon: Array<FloatArray>?,
    val cropWidth: Int,
    val cropHeight: Int,
  )

  data class BoxDetection(
    val anchor: Int,
    val classId: Int,
    val label: String,
    val score: Float,
    val box: FloatArray,
    val cx: Float,
    val cy: Float,
    val digit: String? = null,
  )

  data class ReverseDecision(
    val isReversed: Boolean,
    val reason: String,
    val classes: List<String>,
  )

  data class DigitReading(
    val status: String,
    val rawDigits: String?,
    val reason: String?,
    val confidence: Double,
    val rawDiagnostic: String,
    val digitClassesLeftToRight: List<String>,
    val sortedDetections: List<BoxDetection>,
  )

  fun validateConfig(config: BundleConfig) {
    require(config.schemaVersion == 1) { "Unsupported manifest schemaVersion: ${config.schemaVersion}" }
    require(config.pipelineId == PIPELINE_ID) { "Unexpected pipelineId: ${config.pipelineId}" }
    require(config.semanticsVersion == SEMANTICS_VERSION) { "Unexpected semanticsVersion: ${config.semanticsVersion}" }
    require(config.contractStatus == "approved") { "Manifest contractStatus is not approved: ${config.contractStatus}" }
    require(config.preprocessVersion == PREPROCESS_VERSION) { "Unexpected preprocessVersion: ${config.preprocessVersion}" }
    require(config.colorOrder == "RGB") { "Expected RGB colorOrder, got ${config.colorOrder}" }
    require(config.padValue == 114) { "Expected padValue 114, got ${config.padValue}" }
    require(config.preserveLeadingZeros) { "Manifest must preserve leading zeros" }
    require(config.acceptedLengths == DEFAULT_ACCEPTED_LENGTHS) { "Unexpected acceptedLengths: ${config.acceptedLengths}" }

    val frame = config.frame
    require(frame.file == "frame.onnx" && frame.sha256.length == 64 && frame.sizeBytes > 0) { "Invalid frame model file metadata" }
    require(frame.imgsz == FRAME_IMG_SIZE) { "Expected frame imgsz=$FRAME_IMG_SIZE, got ${frame.imgsz}" }
    require(frame.inputShape == listOf(1L, 3L, FRAME_IMG_SIZE.toLong(), FRAME_IMG_SIZE.toLong())) {
      "Unexpected frame inputShape: ${frame.inputShape}"
    }
    require(frame.outputShape == listOf(1L, FRAME_CHANNELS.toLong(), FRAME_ANCHORS.toLong())) {
      "Unexpected frame outputShape: ${frame.outputShape}"
    }
    require(frame.names == DEFAULT_FRAME_NAMES) { "Unsupported frame class map: ${frame.names}" }

    val rev = config.reverse
    require(rev.file == "reverse.onnx" && rev.sha256.length == 64 && rev.sizeBytes > 0) { "Invalid reverse model file metadata" }
    require(rev.imgsz == DETECT_IMG_SIZE) { "Expected reverse imgsz=$DETECT_IMG_SIZE, got ${rev.imgsz}" }
    require(rev.inputShape == listOf(1L, 3L, DETECT_IMG_SIZE.toLong(), DETECT_IMG_SIZE.toLong())) {
      "Unexpected reverse inputShape: ${rev.inputShape}"
    }
    require(rev.outputShape == listOf(1L, (4 + REVERSE_CLASSES).toLong(), DETECT_ANCHORS.toLong())) {
      "Unexpected reverse outputShape: ${rev.outputShape}"
    }
    require(rev.names == DEFAULT_REVERSE_NAMES) { "Unsupported reverse class map: ${rev.names}" }
    require(rev.asymmetricLabels == DEFAULT_ASYMMETRIC_REVERSE_LABELS) { "Unsupported reverse asymmetricLabels: ${rev.asymmetricLabels}" }
    require(rev.symmetricLabels == DEFAULT_SYMMETRIC_REVERSE_LABELS) { "Unsupported reverse symmetricLabels: ${rev.symmetricLabels}" }
    require(rev.minTotalBoxesForSymmetricTrigger == 2) {
      "Unexpected minTotalBoxesForSymmetricTrigger: ${rev.minTotalBoxesForSymmetricTrigger}"
    }

    val digits = config.digits
    require(digits.file == "digits.onnx" && digits.sha256.length == 64 && digits.sizeBytes > 0) { "Invalid digits model file metadata" }
    require(digits.imgsz == DETECT_IMG_SIZE) { "Expected digits imgsz=$DETECT_IMG_SIZE, got ${digits.imgsz}" }
    require(digits.inputShape == listOf(1L, 3L, DETECT_IMG_SIZE.toLong(), DETECT_IMG_SIZE.toLong())) {
      "Unexpected digits inputShape: ${digits.inputShape}"
    }
    require(digits.outputShape == listOf(1L, (4 + DIGITS_CLASSES).toLong(), DETECT_ANCHORS.toLong())) {
      "Unexpected digits outputShape: ${digits.outputShape}"
    }
    require(digits.names == DEFAULT_DIGITS_NAMES) { "Unsupported digits class map: ${digits.names}" }
    require(digits.classToDigit == DEFAULT_CLASS_TO_DIGIT) { "Unsupported digits classToDigit mapping: ${digits.classToDigit}" }
  }

  fun letterboxMeta(width: Int, height: Int, imgsz: Int): LetterboxMeta {
    require(width > 0 && height > 0 && imgsz > 0) { "Invalid letterbox dimensions: ${width}x${height} -> $imgsz" }
    val gain = min(imgsz.toDouble() / width.toDouble(), imgsz.toDouble() / height.toDouble())
    val rw = round(width * gain).toInt()
    val rh = round(height * gain).toInt()
    val padW = (imgsz - rw) / 2.0
    val padH = (imgsz - rh) / 2.0
    val left = round(padW - 0.1).toInt()
    val top = round(padH - 0.1).toInt()
    val right = imgsz - rw - left
    val bottom = imgsz - rh - top
    return LetterboxMeta(width, height, imgsz, gain, left, top, right, bottom, rw, rh)
  }

  fun packLetterboxTensor(resizedRgb: ByteArray, meta: LetterboxMeta, padValue: Int = 114): FloatArray {
    val rw = meta.resizedWidth
    val rh = meta.resizedHeight
    val imgsz = meta.imgsz
    require(resizedRgb.size == rw * rh * 3) {
      "Unexpected resized RGB byte length: ${resizedRgb.size}, expected ${rw * rh * 3}"
    }
    val area = imgsz * imgsz
    val padFloat = (padValue and 0xFF) / 255f
    val tensor = FloatArray(3 * area) { padFloat }
    val top = meta.top
    val left = meta.left
    for (y in 0 until rh) {
      val dstRowOffset = (y + top) * imgsz + left
      val srcRowOffset = y * rw * 3
      for (x in 0 until rw) {
        val srcIdx = srcRowOffset + x * 3
        val dstIdx = dstRowOffset + x
        tensor[dstIdx] = (resizedRgb[srcIdx].toInt() and 0xFF) / 255f
        tensor[area + dstIdx] = (resizedRgb[srcIdx + 1].toInt() and 0xFF) / 255f
        tensor[2 * area + dstIdx] = (resizedRgb[srcIdx + 2].toInt() and 0xFF) / 255f
      }
    }
    return tensor
  }

  fun probiou(obb1: FloatArray, obb2: FloatArray, eps: Double = 1e-7): Double {
    require(obb1.size >= 5 && obb2.size >= 5) { "Expected 5-element OBB [cx, cy, w, h, angleRad]" }
    val x1 = obb1[0].toDouble()
    val y1 = obb1[1].toDouble()
    val w1 = obb1[2].toDouble()
    val h1 = obb1[3].toDouble()
    val r1 = obb1[4].toDouble()

    val x2 = obb2[0].toDouble()
    val y2 = obb2[1].toDouble()
    val w2 = obb2[2].toDouble()
    val h2 = obb2[3].toDouble()
    val r2 = obb2[4].toDouble()

    val aa1 = (w1 * w1) / 12.0
    val bb1 = (h1 * h1) / 12.0
    val c1 = cos(r1)
    val s1 = sin(r1)
    val a1 = aa1 * c1 * c1 + bb1 * s1 * s1
    val b1 = aa1 * s1 * s1 + bb1 * c1 * c1
    val cov1 = (aa1 - bb1) * c1 * s1

    val aa2 = (w2 * w2) / 12.0
    val bb2 = (h2 * h2) / 12.0
    val c2 = cos(r2)
    val s2 = sin(r2)
    val a2 = aa2 * c2 * c2 + bb2 * s2 * s2
    val b2 = aa2 * s2 * s2 + bb2 * c2 * c2
    val cov2 = (aa2 - bb2) * c2 * s2

    val covSum = cov1 + cov2
    val denom = (a1 + a2) * (b1 + b2) - covSum * covSum + eps
    val dx = x1 - x2
    val dy = y1 - y2
    val t1 = (((a1 + a2) * dy * dy + (b1 + b2) * dx * dx) / denom) * 0.25
    val t2 = ((covSum * (x2 - x1) * dy) / denom) * 0.5
    val det1 = max(0.0, a1 * b1 - cov1 * cov1)
    val det2 = max(0.0, a2 * b2 - cov2 * cov2)
    val t3 = 0.5 * ln(denom / (4.0 * sqrt(det1 * det2) + eps) + eps)

    val bd = (t1 + t2 + t3).coerceIn(eps, 100.0)
    val hd = sqrt(1.0 - exp(-bd) + eps)
    return 1.0 - hd
  }

  fun obbToCorners(cx: Float, cy: Float, w: Float, h: Float, angleRad: Float): Array<FloatArray> {
    val cosA = cos(angleRad.toDouble())
    val sinA = sin(angleRad.toDouble())
    val vx1 = (w.toDouble() / 2.0) * cosA
    val vy1 = (w.toDouble() / 2.0) * sinA
    val vx2 = -(h.toDouble() / 2.0) * sinA
    val vy2 = (h.toDouble() / 2.0) * cosA
    val cxD = cx.toDouble()
    val cyD = cy.toDouble()
    return arrayOf(
      floatArrayOf((cxD + vx1 + vx2).toFloat(), (cyD + vy1 + vy2).toFloat()),
      floatArrayOf((cxD + vx1 - vx2).toFloat(), (cyD + vy1 - vy2).toFloat()),
      floatArrayOf((cxD - vx1 - vx2).toFloat(), (cyD - vy1 - vy2).toFloat()),
      floatArrayOf((cxD - vx1 + vx2).toFloat(), (cyD - vy1 + vy2).toFloat()),
    )
  }

  fun decodeFrameObb(
    rawOutput: FloatArray,
    meta: LetterboxMeta,
    names: Map<Int, String> = DEFAULT_FRAME_NAMES,
    confThres: Float = 0.25f,
    iouThres: Float = 0.50f,
    maxDet: Int = 300,
  ): List<FrameDetection> {
    require(rawOutput.size == FRAME_CHANNELS * FRAME_ANCHORS && rawOutput.all { it.isFinite() }) {
      "Invalid frame OBB output tensor length ${rawOutput.size}"
    }
    require(names.size == 1 && names[0] == "khung") {
      "Unsupported frame class dictionary: $names"
    }

    data class Candidate(val anchor: Int, val score: Float, val obb: FloatArray)

    val candidates = ArrayList<Candidate>()
    for (anchor in 0 until FRAME_ANCHORS) {
      val score = rawOutput[4 * FRAME_ANCHORS + anchor]
      if (score < confThres) continue
      val w = rawOutput[2 * FRAME_ANCHORS + anchor]
      val h = rawOutput[3 * FRAME_ANCHORS + anchor]
      if (w <= 0f || h <= 0f) continue
      val cx = rawOutput[anchor]
      val cy = rawOutput[FRAME_ANCHORS + anchor]
      val angle = rawOutput[5 * FRAME_ANCHORS + anchor]
      candidates.add(Candidate(anchor, score, floatArrayOf(cx, cy, w, h, angle)))
    }

    candidates.sortWith(compareByDescending<Candidate> { it.score }.thenBy { it.anchor })

    val kept = ArrayList<Candidate>()
    val iouLimit = iouThres.toDouble()
    for (cand in candidates) {
      if (kept.all { probiou(cand.obb, it.obb) < iouLimit }) {
        kept.add(cand)
        if (kept.size >= maxDet) break
      }
    }

    val gain = meta.gain
    val left = meta.left.toDouble()
    val top = meta.top.toDouble()
    val label = names.getValue(0)

    return kept.map { cand ->
      val obb = cand.obb
      val cornersNet = obbToCorners(obb[0], obb[1], obb[2], obb[3], obb[4])
      val cornersImg = Array(4) { i ->
        floatArrayOf(
          ((cornersNet[i][0].toDouble() - left) / gain).toFloat(),
          ((cornersNet[i][1].toDouble() - top) / gain).toFloat(),
        )
      }
      FrameDetection(
        anchor = cand.anchor,
        classId = 0,
        label = label,
        score = cand.score,
        netObb = obb,
        polygon = cornersImg,
      )
    }
  }

  fun orderObbCorners(pts: Array<FloatArray>): Array<FloatArray> {
    require(pts.size == 4 && pts.all { it.size == 2 && it[0].isFinite() && it[1].isFinite() }) {
      "Expected (4, 2) finite polygon vertices"
    }
    fun edgeDist(a: FloatArray, b: FloatArray): Double {
      val dx = (a[0] - b[0]).toDouble()
      val dy = (a[1] - b[1]).toDouble()
      return sqrt(dx * dx + dy * dy)
    }
    val d01 = edgeDist(pts[1], pts[0])
    val d12 = edgeDist(pts[2], pts[1])

    val raw1 = if (d01 >= d12) arrayOf(pts[0], pts[1], pts[2], pts[3]) else arrayOf(pts[1], pts[2], pts[3], pts[0])
    val raw2 = if (d01 >= d12) arrayOf(pts[2], pts[3], pts[0], pts[1]) else arrayOf(pts[3], pts[0], pts[1], pts[2])

    fun ensurePositiveChirality(c: Array<FloatArray>): Array<FloatArray> {
      val tl = c[0]
      val tr = c[1]
      val br = c[2]
      val bl = c[3]
      val cross = (tr[0] - tl[0]).toDouble() * (bl[1] - tl[1]).toDouble() -
        (tr[1] - tl[1]).toDouble() * (bl[0] - tl[0]).toDouble()
      return if (cross < 0.0) {
        arrayOf(bl.copyOf(), br.copyOf(), tr.copyOf(), tl.copyOf())
      } else {
        arrayOf(tl.copyOf(), tr.copyOf(), br.copyOf(), bl.copyOf())
      }
    }

    val c1 = ensurePositiveChirality(raw1)
    val c2 = ensurePositiveChirality(raw2)
    val dx1 = (c1[1][0] - c1[0][0]).toDouble()
    val dy1 = (c1[1][1] - c1[0][1]).toDouble()
    return if (dx1 > 1e-6 || (abs(dx1) <= 1e-6 && dy1 >= 0.0)) c1 else c2
  }

  fun planFrameCrop(
    width: Int,
    height: Int,
    detections: List<FrameDetection>,
    bounds: GeometryBounds = GeometryBounds(),
  ): CropPlan {
    require(width > 0 && height > 0) { "Invalid image dimensions: ${width}x${height}" }
    if (detections.isEmpty()) return CropPlan("no_row", null, null, 0, 0)
    if (detections.size > 1) return CropPlan("multiple_rows", null, null, 0, 0)

    val ordered = orderObbCorners(detections[0].polygon)
    val tl = ordered[0]
    val tr = ordered[1]
    val br = ordered[2]
    val bl = ordered[3]

    val minX = min(min(tl[0], tr[0]), min(br[0], bl[0])).toDouble()
    val maxX = max(max(tl[0], tr[0]), max(br[0], bl[0])).toDouble()
    val minY = min(min(tl[1], tr[1]), min(br[1], bl[1])).toDouble()
    val maxY = max(max(tl[1], tr[1]), max(br[1], bl[1])).toDouble()

    val leftBox = floor(minX).toInt().coerceIn(0, width - 1)
    val topBox = floor(minY).toInt().coerceIn(0, height - 1)
    val rightBox = ceil(maxX).toInt().coerceIn(leftBox + 1, width)
    val bottomBox = ceil(maxY).toInt().coerceIn(topBox + 1, height)
    val rowBox = intArrayOf(leftBox, topBox, rightBox, bottomBox)

    fun edgeDist(a: FloatArray, b: FloatArray): Double {
      val dx = (a[0] - b[0]).toDouble()
      val dy = (a[1] - b[1]).toDouble()
      return sqrt(dx * dx + dy * dy)
    }

    val wTop = edgeDist(tr, tl)
    val wBot = edgeDist(br, bl)
    val hLeft = edgeDist(bl, tl)
    val hRight = edgeDist(br, tr)
    val cropW = max(round(wTop).toInt(), round(wBot).toInt())
    val cropH = max(round(hLeft).toInt(), round(hRight).toInt())
    val aspect = if (cropH > 0) cropW.toDouble() / cropH.toDouble() else 0.0

    if (
      cropW < bounds.minWidth ||
      cropH < bounds.minHeight ||
      aspect < bounds.minAspect ||
      aspect > bounds.maxAspect
    ) {
      return CropPlan("unusable_row_geometry", null, ordered, cropW, cropH)
    }

    return CropPlan(null, rowBox, ordered, cropW, cropH)
  }

  fun rotateRgb180(rgb: ByteArray, width: Int, height: Int): ByteArray {
    require(width > 0 && height > 0 && rgb.size == width * height * 3) {
      "Invalid RGB buffer for 180-degree rotation: ${rgb.size} bytes for ${width}x${height}"
    }
    val totalPixels = width * height
    val out = ByteArray(rgb.size)
    for (i in 0 until totalPixels) {
      val srcOffset = i * 3
      val dstOffset = (totalPixels - 1 - i) * 3
      out[dstOffset] = rgb[srcOffset]
      out[dstOffset + 1] = rgb[srcOffset + 1]
      out[dstOffset + 2] = rgb[srcOffset + 2]
    }
    return out
  }

  fun axisIou(box1: FloatArray, box2: FloatArray): Float {
    val x1 = max(box1[0], box2[0])
    val y1 = max(box1[1], box2[1])
    val x2 = min(box1[2], box2[2])
    val y2 = min(box1[3], box2[3])
    val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
    val area1 = max(0f, box1[2] - box1[0]) * max(0f, box1[3] - box1[1])
    val area2 = max(0f, box2[2] - box2[0]) * max(0f, box2[3] - box2[1])
    val denom = area1 + area2 - inter
    return if (denom > 0f) inter / denom else 0f
  }

  fun decodeAxisAlignedBoxes(
    rawOutput: FloatArray,
    anchors: Int,
    meta: LetterboxMeta,
    names: Map<Int, String>,
    confThres: Float = 0.25f,
    iouThres: Float = 0.50f,
    agnosticNms: Boolean = true,
    maxDet: Int = 300,
  ): List<BoxDetection> {
    val nc = names.size
    require(nc > 0 && anchors > 0) { "Invalid detection dimensions nc=$nc anchors=$anchors" }
    require(rawOutput.size == (4 + nc) * anchors && rawOutput.all { it.isFinite() }) {
      "Invalid detection output tensor length ${rawOutput.size}, expected ${(4 + nc) * anchors}"
    }

    data class Candidate(
      val anchor: Int,
      val classId: Int,
      val label: String,
      val score: Float,
      val netBox: FloatArray,
    )

    val candidates = ArrayList<Candidate>()
    for (anchor in 0 until anchors) {
      var bestCls = 0
      var bestScore = rawOutput[4 * anchors + anchor]
      for (c in 1 until nc) {
        val score = rawOutput[(4 + c) * anchors + anchor]
        if (score > bestScore) {
          bestScore = score
          bestCls = c
        }
      }
      if (bestScore < confThres) continue
      val w = rawOutput[2 * anchors + anchor]
      val h = rawOutput[3 * anchors + anchor]
      if (w <= 0f || h <= 0f) continue
      val cx = rawOutput[anchor]
      val cy = rawOutput[anchors + anchor]
      val label = names[bestCls] ?: throw IllegalArgumentException("Unsupported class index $bestCls")
      val netBox = floatArrayOf(cx - w / 2f, cy - h / 2f, cx + w / 2f, cy + h / 2f)
      candidates.add(Candidate(anchor, bestCls, label, bestScore, netBox))
    }

    candidates.sortWith(compareByDescending<Candidate> { it.score }.thenBy { it.anchor })

    val kept = ArrayList<Candidate>()
    for (cand in candidates) {
      val conflict = if (agnosticNms) {
        kept.any { axisIou(cand.netBox, it.netBox) >= iouThres }
      } else {
        kept.any { it.classId == cand.classId && axisIou(cand.netBox, it.netBox) >= iouThres }
      }
      if (!conflict) {
        kept.add(cand)
        if (kept.size >= maxDet) break
      }
    }

    val gain = meta.gain
    val left = meta.left.toDouble()
    val top = meta.top.toDouble()
    val width = meta.inputWidth.toDouble()
    val height = meta.inputHeight.toDouble()

    return kept.map { cand ->
      val nb = cand.netBox
      val x1 = (((nb[0].toDouble() - left) / gain).coerceIn(0.0, width)).toFloat()
      val y1 = (((nb[1].toDouble() - top) / gain).coerceIn(0.0, height)).toFloat()
      val x2 = (((nb[2].toDouble() - left) / gain).coerceIn(0.0, width)).toFloat()
      val y2 = (((nb[3].toDouble() - top) / gain).coerceIn(0.0, height)).toFloat()
      BoxDetection(
        anchor = cand.anchor,
        classId = cand.classId,
        label = cand.label,
        score = cand.score,
        box = floatArrayOf(x1, y1, x2, y2),
        cx = (x1 + x2) / 2f,
        cy = (y1 + y2) / 2f,
      )
    }
  }

  fun evaluateReverseTrigger(
    detections: List<BoxDetection>,
    asymmetricLabels: Set<String> = DEFAULT_ASYMMETRIC_REVERSE_LABELS,
    symmetricLabels: Set<String> = DEFAULT_SYMMETRIC_REVERSE_LABELS,
    minTotalBoxes: Int = 2,
  ): ReverseDecision {
    val allowed = asymmetricLabels union symmetricLabels
    val labels = detections.map { det ->
      require(det.label in allowed) { "Unsupported reverse check class label: ${det.label}" }
      det.label
    }
    return when {
      labels.any { it in asymmetricLabels } -> ReverseDecision(true, "asymmetric_reverse_label", labels)
      labels.size >= minTotalBoxes -> ReverseDecision(true, "multiple_reverse_boxes", labels)
      labels.size == 1 -> ReverseDecision(false, "single_symmetric_1n_ignored", labels)
      else -> ReverseDecision(false, "no_reverse_boxes", labels)
    }
  }

  fun mapDigitLabel(
    label: String,
    classToDigit: Map<String, String> = DEFAULT_CLASS_TO_DIGIT,
  ): String {
    val mapped = classToDigit[label]
      ?: throw IllegalArgumentException("Unsupported digit class label: $label")
    val numeric = label.toIntOrNull()
      ?: throw IllegalArgumentException("Non-numeric digit class label: $label")
    require(numeric in 0 until DIGITS_CLASSES) { "Out-of-range digit class label: $label" }
    val expected = (numeric % 10).toString()
    require(mapped == expected && mapped.length == 1 && mapped[0] in '0'..'9') {
      "Invalid digit mapping for label $label: $mapped"
    }
    return mapped
  }

  fun decodeDigitsReading(
    detections: List<BoxDetection>,
    classToDigit: Map<String, String> = DEFAULT_CLASS_TO_DIGIT,
    acceptedLengths: Set<Int> = DEFAULT_ACCEPTED_LENGTHS,
  ): DigitReading {
    val sorted = detections.sortedWith(
      compareBy<BoxDetection> { it.box[0] }
        .thenBy { it.cx }
        .thenBy { it.anchor }
    )

    val digitsBuilder = StringBuilder(sorted.size)
    val classesLtr = ArrayList<String>(sorted.size)
    val enriched = ArrayList<BoxDetection>(sorted.size)
    var scoreSum = 0.0

    for (det in sorted) {
      val digit = mapDigitLabel(det.label, classToDigit)
      digitsBuilder.append(digit)
      classesLtr.add(det.label)
      enriched.add(det.copy(digit = digit))
      scoreSum += det.score.toDouble()
    }

    val raw = digitsBuilder.toString()
    val confidence = if (sorted.isEmpty()) 0.0 else scoreSum / sorted.size.toDouble()
    require(confidence.isFinite() && confidence in 0.0..1.0) { "Invalid digit reading confidence: $confidence" }

    val reason = when {
      raw.isEmpty() -> "no_characters"
      raw.any { it !in '0'..'9' } -> "non_digit_output"
      raw.length !in acceptedLengths -> "unexpected_length"
      else -> null
    }
    val status = if (reason == null) "ok" else "uncertain"
    val rawDigits = if (status == "ok") raw else null

    return DigitReading(
      status = status,
      rawDigits = rawDigits,
      reason = reason,
      confidence = confidence,
      rawDiagnostic = raw,
      digitClassesLeftToRight = classesLtr,
      sortedDetections = enriched,
    )
  }

  /**
   * When a 6-digit water meter has one roller halfway between two digits (e.g. 4-5), its activation
   * in the 30-class model (`d`, `d+10`, `d+20`) can split across sub-classes so each individual
   * head falls just below `confThres = 0.25f`, leaving 5 high-confidence digits and one empty slot.
   *
   * If exactly 5 digits are detected and their horizontal pitch reveals an empty 6th slot (internal
   * gap or left/right edge slot), scan the raw output tensor within that slot and combine same-digit
   * sub-class probabilities (`d`, `d+10`, `d+20`) via noisy-OR to recover the transitioning roller.
   */
  fun recoverMissingRollerDigit(
    rawOutput: FloatArray,
    anchors: Int,
    meta: LetterboxMeta,
    names: Map<Int, String>,
    primaryDetections: List<BoxDetection>,
    cropWidth: Int,
    cropHeight: Int,
  ): List<BoxDetection> {
    if (primaryDetections.size != 5 || cropWidth <= 0 || cropHeight <= 0) return primaryDetections
    val nc = names.size
    if (nc != DIGITS_CLASSES || rawOutput.size != (4 + nc) * anchors) return primaryDetections

    val sorted = primaryDetections.sortedWith(
      compareBy<BoxDetection> { it.box[0] }.thenBy { it.cx }.thenBy { it.anchor }
    )
    val steps = FloatArray(4) { i -> sorted[i + 1].cx - sorted[i].cx }
    val sortedSteps = steps.sortedArray()
    val medianStep = (sortedSteps[1] + sortedSteps[2]) / 2f
    if (!medianStep.isFinite() || medianStep <= 4f) return primaryDetections

    val widths = sorted.map { it.box[2] - it.box[0] }.sorted()
    val heights = sorted.map { it.box[3] - it.box[1] }.sorted()
    val cys = sorted.map { it.cy }.sorted()
    val medianWidth = widths[2]
    val medianHeight = heights[2]
    val medianCy = cys[2]
    if (medianWidth <= 2f || medianHeight <= 4f) return primaryDetections
    // Verify the 5 existing digits have consistent spacing (at most one missing-digit gap).
    var internalGapIndex = -1
    for (i in steps.indices) {
      val s = steps[i]
      if (s < medianWidth * 0.45f) return primaryDetections
      if (s > medianStep * 1.60f) {
        if (internalGapIndex != -1 || s > medianStep * 2.55f) return primaryDetections
        internalGapIndex = i
      }
    }

    val expectedSlots = ArrayList<Float>(2)
    if (internalGapIndex >= 0) {
      expectedSlots.add((sorted[internalGapIndex].cx + sorted[internalGapIndex + 1].cx) / 2f)
    } else {
      val rightMargin = cropWidth.toFloat() - sorted.last().box[2]
      val leftMargin = sorted.first().box[0]
      if (rightMargin >= medianStep * 0.82f) {
        val rightCx = sorted.last().cx + medianStep
        if (rightCx <= cropWidth - medianWidth * 0.20f) expectedSlots.add(rightCx)
      }
      if (leftMargin >= medianStep * 0.82f) {
        val leftCx = sorted.first().cx - medianStep
        if (leftCx >= medianWidth * 0.20f) expectedSlots.add(leftCx)
      }
    }
    if (expectedSlots.isEmpty()) return primaryDetections

    val gain = meta.gain
    val leftPad = meta.left.toDouble()
    val topPad = meta.top.toDouble()
    val widthD = meta.inputWidth.toDouble()
    val heightD = meta.inputHeight.toDouble()

    var bestCandidate: BoxDetection? = null
    var bestCombinedScore = 0f

    for (slotCx in expectedSlots) {
      for (anchor in 0 until anchors) {
        val wNet = rawOutput[2 * anchors + anchor]
        val hNet = rawOutput[3 * anchors + anchor]
        if (wNet <= 0f || hNet <= 0f) continue
        val cxNet = rawOutput[anchor]
        val cyNet = rawOutput[anchors + anchor]
        val x1 = (((cxNet - wNet / 2f).toDouble() - leftPad) / gain).coerceIn(0.0, widthD).toFloat()
        val y1 = (((cyNet - hNet / 2f).toDouble() - topPad) / gain).coerceIn(0.0, heightD).toFloat()
        val x2 = (((cxNet + wNet / 2f).toDouble() - leftPad) / gain).coerceIn(0.0, widthD).toFloat()
        val y2 = (((cyNet + hNet / 2f).toDouble() - topPad) / gain).coerceIn(0.0, heightD).toFloat()
        val bw = x2 - x1
        val bh = y2 - y1
        if (bw < medianWidth * 0.45f || bw > medianWidth * 1.85f) continue
        if (bh < medianHeight * 0.50f || bh > medianHeight * 1.65f) continue
        val cx = (x1 + x2) / 2f
        val cy = (y1 + y2) / 2f
        if (abs(cx - slotCx) > medianStep * 0.42f) continue
        if (abs(cy - medianCy) > medianHeight * 0.50f) continue
        val candBox = floatArrayOf(x1, y1, x2, y2)
        if (sorted.any { axisIou(candBox, it.box) >= 0.22f }) continue

        for (digit in 0 until 10) {
          val s0 = rawOutput[(4 + digit) * anchors + anchor]
          val s1 = rawOutput[(4 + digit + 10) * anchors + anchor]
          val s2 = rawOutput[(4 + digit + 20) * anchors + anchor]
          val maxSub = max(s0, max(s1, s2))
          if (maxSub < 0.12f) continue
          val combined = 1f - (1f - s0.coerceIn(0f, 1f)) * (1f - s1.coerceIn(0f, 1f)) * (1f - s2.coerceIn(0f, 1f))
          if (combined < 0.22f) continue
          val bestCls = when {
            s1 >= s0 && s1 >= s2 -> digit + 10
            s2 >= s0 && s2 >= s1 -> digit + 20
            else -> digit
          }
          val label = names[bestCls] ?: continue
          if (combined > bestCombinedScore) {
            bestCombinedScore = combined
            bestCandidate = BoxDetection(
              anchor = anchor,
              classId = bestCls,
              label = label,
              score = max(maxSub, combined.coerceAtMost(0.75f)),
              box = candBox,
              cx = cx,
              cy = cy,
            )
          }
        }
      }
    }

    return if (bestCandidate != null) primaryDetections + bestCandidate else primaryDetections
  }
}
