package com.watermeterocr.ocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.security.MessageDigest
import java.util.UUID

/**
 * Owned exclusively by MeterOcrModule's serial executor. Manages the three ONNX Runtime sessions
 * (`frame.onnx`, `reverse.onnx`, `digits.onnx`) and OpenCV image operations offline on device.
 */
class SuppliedYoloEngine(private val context: Context) : AutoCloseable {
  private val environment = OrtEnvironment.getEnvironment()
  private val manifestBytes = context.assets.open("ocr/supplied-yolo-manifest.json").use { it.readBytes() }
  private val manifestSha256 = hash(manifestBytes)
  private val manifest = JSONObject(String(manifestBytes, Charsets.UTF_8))
  private val config = parseAndValidateManifest(manifest)

  private val frameSession: OrtSession
  private val reverseSession: OrtSession
  private val digitsSession: OrtSession

  init {
    check(OpenCVLoader.initLocal()) { "Cannot initialize bundled OpenCV" }
    val options = OrtSession.SessionOptions().apply {
      setIntraOpNumThreads(2)
      setInterOpNumThreads(1)
    }
    var openedFrame: OrtSession? = null
    var openedReverse: OrtSession? = null
    var openedDigits: OrtSession? = null
    try {
      val frameFile = stageModelFile(config.frame.file, config.frame.sha256, config.frame.sizeBytes)
      openedFrame = environment.createSession(frameFile.absolutePath, options)

      val reverseFile = stageModelFile(config.reverse.file, config.reverse.sha256, config.reverse.sizeBytes)
      openedReverse = environment.createSession(reverseFile.absolutePath, options)

      val digitsFile = stageModelFile(config.digits.file, config.digits.sha256, config.digits.sizeBytes)
      openedDigits = environment.createSession(digitsFile.absolutePath, options)

      frameSession = openedFrame
      reverseSession = openedReverse
      digitsSession = openedDigits
    } catch (error: Throwable) {
      try { openedDigits?.close() } catch (_: Throwable) {}
      try { openedReverse?.close() } catch (_: Throwable) {}
      try { openedFrame?.close() } catch (_: Throwable) {}
      throw error
    } finally {
      options.close()
    }
  }

  private fun parseAndValidateManifest(root: JSONObject): SuppliedYoloDecoder.BundleConfig {
    val preprocess = root.getJSONObject("preprocessing")
    require(preprocess.getString("interpolation") == "INTER_LINEAR") { "Unsupported resize interpolation" }
    require(preprocess.getString("obbWarpInterpolation") == "INTER_LINEAR") { "Unsupported OBB warp interpolation" }
    require(preprocess.getString("obbWarpBorderMode") == "BORDER_REPLICATE") { "Unsupported OBB warp border mode" }

    val models = root.getJSONObject("models")
    val frameJson = models.getJSONObject("frame")
    val revJson = models.getJSONObject("reverse")
    val digitsJson = models.getJSONObject("digits")
    val policyJson = root.getJSONObject("policy")
    val boundsJson = policyJson.getJSONObject("geometryBounds")

    fun parseLongList(arr: JSONArray): List<Long> = List(arr.length()) { arr.getLong(it) }
    fun parseIntStringMap(obj: JSONObject): Map<Int, String> {
      val map = LinkedHashMap<Int, String>()
      val keys = obj.keys()
      while (keys.hasNext()) {
        val k = keys.next()
        map[k.toInt()] = obj.getString(k)
      }
      return map
    }
    fun parseStringMap(obj: JSONObject): Map<String, String> {
      val map = LinkedHashMap<String, String>()
      val keys = obj.keys()
      while (keys.hasNext()) {
        val k = keys.next()
        map[k] = obj.getString(k)
      }
      return map
    }
    fun parseStringSet(arr: JSONArray): Set<String> = (0 until arr.length()).mapTo(LinkedHashSet()) { arr.getString(it) }
    fun parseIntSet(arr: JSONArray): Set<Int> = (0 until arr.length()).mapTo(LinkedHashSet()) { arr.getInt(it) }

    val bundle = SuppliedYoloDecoder.BundleConfig(
      schemaVersion = root.getInt("schemaVersion"),
      pipelineId = root.getString("pipelineId"),
      semanticsVersion = root.getString("semanticsVersion"),
      contractStatus = root.getString("contractStatus"),
      runtime = root.getString("runtime"),
      preprocessVersion = preprocess.getString("version"),
      colorOrder = preprocess.getString("colorOrder"),
      padValue = preprocess.getInt("padValue"),
      frame = SuppliedYoloDecoder.FrameModelConfig(
        file = frameJson.getString("file"),
        sha256 = frameJson.getString("sha256"),
        sizeBytes = frameJson.getLong("sizeBytes"),
        imgsz = frameJson.getInt("imgsz"),
        inputShape = parseLongList(frameJson.getJSONArray("inputShape")),
        outputShape = parseLongList(frameJson.getJSONArray("outputShape")),
        confThres = frameJson.getDouble("confThres").toFloat(),
        iouThres = frameJson.getDouble("iouThres").toFloat(),
        maxDet = frameJson.getInt("maxDet"),
        names = parseIntStringMap(frameJson.getJSONObject("names")),
      ),
      reverse = SuppliedYoloDecoder.ReverseModelConfig(
        file = revJson.getString("file"),
        sha256 = revJson.getString("sha256"),
        sizeBytes = revJson.getLong("sizeBytes"),
        imgsz = revJson.getInt("imgsz"),
        inputShape = parseLongList(revJson.getJSONArray("inputShape")),
        outputShape = parseLongList(revJson.getJSONArray("outputShape")),
        confThres = revJson.getDouble("confThres").toFloat(),
        iouThres = revJson.getDouble("iouThres").toFloat(),
        maxDet = revJson.getInt("maxDet"),
        agnosticNms = revJson.optBoolean("agnosticNms", true),
        names = parseIntStringMap(revJson.getJSONObject("names")),
        asymmetricLabels = parseStringSet(revJson.getJSONArray("asymmetricLabels")),
        symmetricLabels = parseStringSet(revJson.getJSONArray("symmetricLabels")),
        minTotalBoxesForSymmetricTrigger = revJson.getInt("minTotalBoxesForSymmetricTrigger"),
      ),
      digits = SuppliedYoloDecoder.DigitsModelConfig(
        file = digitsJson.getString("file"),
        sha256 = digitsJson.getString("sha256"),
        sizeBytes = digitsJson.getLong("sizeBytes"),
        imgsz = digitsJson.getInt("imgsz"),
        inputShape = parseLongList(digitsJson.getJSONArray("inputShape")),
        outputShape = parseLongList(digitsJson.getJSONArray("outputShape")),
        confThres = digitsJson.getDouble("confThres").toFloat(),
        iouThres = digitsJson.getDouble("iouThres").toFloat(),
        maxDet = digitsJson.getInt("maxDet"),
        agnosticNms = digitsJson.optBoolean("agnosticNms", true),
        names = parseIntStringMap(digitsJson.getJSONObject("names")),
        classToDigit = parseStringMap(digitsJson.getJSONObject("classToDigit")),
      ),
      geometryBounds = SuppliedYoloDecoder.GeometryBounds(
        minWidth = boundsJson.getInt("minWidth"),
        minHeight = boundsJson.getInt("minHeight"),
        minAspect = boundsJson.getDouble("minAspect"),
        maxAspect = boundsJson.getDouble("maxAspect"),
      ),
      acceptedLengths = parseIntSet(policyJson.getJSONArray("acceptedLengths")),
      preserveLeadingZeros = policyJson.getBoolean("preserveLeadingZeros"),
    )
    SuppliedYoloDecoder.validateConfig(bundle)
    return bundle
  }

  private fun stageModelFile(name: String, expectedSha256: String, expectedSize: Long): File {
    require(name in setOf("frame.onnx", "reverse.onnx", "digits.onnx")) { "Unexpected bundled model filename: $name" }
    val target = File(context.cacheDir, name)
    if (!target.exists() || target.length() != expectedSize || hashFile(target) != expectedSha256) {
      context.assets.open("ocr/$name").use { input ->
        target.outputStream().use { output -> input.copyTo(output) }
      }
    }
    check(target.length() == expectedSize && hashFile(target) == expectedSha256) {
      "Bundled ONNX model checksum or size mismatch for $name"
    }
    return target
  }

  private data class Output(val data: FloatArray, val shape: LongArray)

  private fun invoke(session: OrtSession, input: FloatArray, shape: LongArray): Output {
    OnnxTensor.createTensor(environment, FloatBuffer.wrap(input), shape).use { tensor ->
      session.run(mapOf(session.inputNames.single() to tensor)).use { result ->
        val output = result[0] as OnnxTensor
        val buffer = output.floatBuffer
        require(buffer.remaining() <= 8_000_000) { "Model output exceeds tensor budget" }
        val values = FloatArray(buffer.remaining())
        buffer.get(values)
        check(values.all { it.isFinite() }) { "Model output contains non-finite values" }
        return Output(values, output.info.shape)
      }
    }
  }

  private fun resizeOpenCv(rgb: ByteArray, width: Int, height: Int, targetWidth: Int, targetHeight: Int): ByteArray {
    if (width == targetWidth && height == targetHeight) return rgb
    val source = Mat(height, width, CvType.CV_8UC3)
    val target = Mat()
    try {
      source.put(0, 0, rgb)
      Imgproc.resize(
        source,
        target,
        Size(targetWidth.toDouble(), targetHeight.toDouble()),
        0.0,
        0.0,
        Imgproc.INTER_LINEAR,
      )
      return ByteArray(targetWidth * targetHeight * 3).also { target.get(0, 0, it) }
    } finally {
      source.release()
      target.release()
    }
  }

  private fun letterbox(rgb: ByteArray, width: Int, height: Int, imgsz: Int): Pair<FloatArray, SuppliedYoloDecoder.LetterboxMeta> {
    val meta = SuppliedYoloDecoder.letterboxMeta(width, height, imgsz)
    val resized = resizeOpenCv(rgb, width, height, meta.resizedWidth, meta.resizedHeight)
    val tensor = SuppliedYoloDecoder.packLetterboxTensor(resized, meta, config.padValue)
    return tensor to meta
  }

  private fun warpObbCrop(
    rgb: ByteArray,
    width: Int,
    height: Int,
    orderedPolygon: Array<FloatArray>,
    cropWidth: Int,
    cropHeight: Int,
  ): Pair<ByteArray, DoubleArray> {
    val srcMat = Mat(height, width, CvType.CV_8UC3)
    val dstMat = Mat()
    val srcPts = Mat(4, 1, CvType.CV_32FC2)
    val dstPts = Mat(4, 1, CvType.CV_32FC2)
    var matrix: Mat? = null
    try {
      srcMat.put(0, 0, rgb)
      srcPts.put(
        0,
        0,
        floatArrayOf(
          orderedPolygon[0][0], orderedPolygon[0][1],
          orderedPolygon[1][0], orderedPolygon[1][1],
          orderedPolygon[2][0], orderedPolygon[2][1],
          orderedPolygon[3][0], orderedPolygon[3][1],
        ),
      )
      dstPts.put(
        0,
        0,
        floatArrayOf(
          0f, 0f,
          (cropWidth - 1).toFloat(), 0f,
          (cropWidth - 1).toFloat(), (cropHeight - 1).toFloat(),
          0f, (cropHeight - 1).toFloat(),
        ),
      )
      matrix = Imgproc.getPerspectiveTransform(srcPts, dstPts)
      Imgproc.warpPerspective(
        srcMat,
        dstMat,
        matrix,
        Size(cropWidth.toDouble(), cropHeight.toDouble()),
        Imgproc.INTER_LINEAR,
        Core.BORDER_REPLICATE,
      )
      val cropRgb = ByteArray(cropWidth * cropHeight * 3).also { dstMat.get(0, 0, it) }
      val matrixValues = DoubleArray(9).also { matrix.get(0, 0, it) }
      return cropRgb to matrixValues
    } finally {
      matrix?.release()
      srcPts.release()
      dstPts.release()
      srcMat.release()
      dstMat.release()
    }
  }

  fun run(bitmap: Bitmap, synthetic: Boolean, saveParity: Boolean): String {
    val width = bitmap.width
    val height = bitmap.height
    val rgb = ByteArray(width * height * 3)
    val row = IntArray(width)
    for (y in 0 until height) {
      bitmap.getPixels(row, 0, width, 0, y, width, 1)
      for (x in 0 until width) {
        val i = (y * width + x) * 3
        val pixel = row[x]
        rgb[i] = (pixel shr 16).toByte()
        rgb[i + 1] = (pixel shr 8).toByte()
        rgb[i + 2] = pixel.toByte()
      }
    }

    val start = System.nanoTime()
    val frameShape = longArrayOf(1, 3, config.frame.imgsz.toLong(), config.frame.imgsz.toLong())
    val expectedFrameOut = longArrayOf(1, SuppliedYoloDecoder.FRAME_CHANNELS.toLong(), SuppliedYoloDecoder.FRAME_ANCHORS.toLong())
    val detectShape = longArrayOf(1, 3, SuppliedYoloDecoder.DETECT_IMG_SIZE.toLong(), SuppliedYoloDecoder.DETECT_IMG_SIZE.toLong())
    val expectedRevOut = longArrayOf(1, (4 + SuppliedYoloDecoder.REVERSE_CLASSES).toLong(), SuppliedYoloDecoder.DETECT_ANCHORS.toLong())
    val expectedDigitsOut = longArrayOf(1, (4 + SuppliedYoloDecoder.DIGITS_CLASSES).toLong(), SuppliedYoloDecoder.DETECT_ANCHORS.toLong())

    val (frameInput, frameMeta) = letterbox(rgb, width, height, config.frame.imgsz)
    val frameOutput = invoke(frameSession, frameInput, frameShape)
    require(frameOutput.shape.contentEquals(expectedFrameOut)) { "Unexpected frame OBB output shape" }

    if (synthetic) {
      val dummy320 = FloatArray(3 * SuppliedYoloDecoder.DETECT_IMG_SIZE * SuppliedYoloDecoder.DETECT_IMG_SIZE) {
        (config.padValue and 0xFF) / 255f
      }
      val revOut = invoke(reverseSession, dummy320, detectShape)
      require(revOut.shape.contentEquals(expectedRevOut)) { "Unexpected reverse detector output shape" }
      val digOut = invoke(digitsSession, dummy320, detectShape)
      require(digOut.shape.contentEquals(expectedDigitsOut)) { "Unexpected digits detector output shape" }

      return JSONObject()
        .put("runtime", config.runtime)
        .put("pipelineId", config.pipelineId)
        .put("semanticsVersion", config.semanticsVersion)
        .put("decoderVersion", SuppliedYoloDecoder.SEMANTICS_VERSION)
        .put("preprocessVersion", config.preprocessVersion)
        .put("modelManifestSha256", manifestSha256)
        .put("frameModelSha256", config.frame.sha256)
        .put("reverseModelSha256", config.reverse.sha256)
        .put("digitsModelSha256", config.digits.sha256)
        .put("inputShape", JSONArray(frameShape))
        .put("outputShape", JSONArray(frameOutput.shape))
        .put("reverseInputShape", JSONArray(detectShape))
        .put("reverseOutputShape", JSONArray(revOut.shape))
        .put("digitsInputShape", JSONArray(detectShape))
        .put("digitsOutputShape", JSONArray(digOut.shape))
        .put("finiteOutputs", frameOutput.data.size + revOut.data.size + digOut.data.size)
        .put("invokeMs", (System.nanoTime() - start) / 1_000_000.0)
        .put("inputSource", "synthetic")
        .put("source", "diagnostic")
        .put("authenticity", "unverified")
        .put("status", "diagnostic")
        .put("rawDigits", JSONObject.NULL)
        .put("reason", JSONObject.NULL)
        .put("confidence", 0.0)
        .put("rowBox", JSONObject.NULL)
        .put("imageSize", JSONArray(intArrayOf(width, height)))
        .toString()
    }

    val frameDetections = SuppliedYoloDecoder.decodeFrameObb(
      rawOutput = frameOutput.data,
      meta = frameMeta,
      names = config.frame.names,
      confThres = config.frame.confThres,
      iouThres = config.frame.iouThres,
      maxDet = config.frame.maxDet,
    )
    val cropPlan = SuppliedYoloDecoder.planFrameCrop(width, height, frameDetections, config.geometryBounds)

    var cropRgb: ByteArray? = null
    var cropUsedRgb: ByteArray? = null
    var revInput: FloatArray? = null
    var revOutput: Output? = null
    var digitsInput: FloatArray? = null
    var digitsOutput: Output? = null
    var revDetections: List<SuppliedYoloDecoder.BoxDetection> = emptyList()
    var revDecision = SuppliedYoloDecoder.ReverseDecision(false, "not_run", emptyList())
    var reading = SuppliedYoloDecoder.DigitReading(
      status = "uncertain",
      rawDigits = null,
      reason = cropPlan.reason,
      confidence = 0.0,
      rawDiagnostic = "",
      digitClassesLeftToRight = emptyList(),
      sortedDetections = emptyList(),
    )

    if (cropPlan.reason == null && cropPlan.orderedPolygon != null) {
      val (warpedRgb, _) = warpObbCrop(
        rgb = rgb,
        width = width,
        height = height,
        orderedPolygon = cropPlan.orderedPolygon,
        cropWidth = cropPlan.cropWidth,
        cropHeight = cropPlan.cropHeight,
      )
      cropRgb = warpedRgb

      val (preparedRev, revMeta) = letterbox(warpedRgb, cropPlan.cropWidth, cropPlan.cropHeight, config.reverse.imgsz)
      revInput = preparedRev
      val rawRev = invoke(reverseSession, preparedRev, detectShape)
      require(rawRev.shape.contentEquals(expectedRevOut)) { "Unexpected reverse detector output shape" }
      revOutput = rawRev

      revDetections = SuppliedYoloDecoder.decodeAxisAlignedBoxes(
        rawOutput = rawRev.data,
        anchors = SuppliedYoloDecoder.DETECT_ANCHORS,
        meta = revMeta,
        names = config.reverse.names,
        confThres = config.reverse.confThres,
        iouThres = config.reverse.iouThres,
        agnosticNms = config.reverse.agnosticNms,
        maxDet = config.reverse.maxDet,
      )
      revDecision = SuppliedYoloDecoder.evaluateReverseTrigger(
        detections = revDetections,
        asymmetricLabels = config.reverse.asymmetricLabels,
        symmetricLabels = config.reverse.symmetricLabels,
        minTotalBoxes = config.reverse.minTotalBoxesForSymmetricTrigger,
      )

      val activeCrop = if (revDecision.isReversed) {
        SuppliedYoloDecoder.rotateRgb180(warpedRgb, cropPlan.cropWidth, cropPlan.cropHeight)
      } else {
        warpedRgb
      }
      cropUsedRgb = activeCrop

      val (preparedDigits, digitsMeta) = letterbox(activeCrop, cropPlan.cropWidth, cropPlan.cropHeight, config.digits.imgsz)
      digitsInput = preparedDigits
      val rawDigits = invoke(digitsSession, preparedDigits, detectShape)
      require(rawDigits.shape.contentEquals(expectedDigitsOut)) { "Unexpected digits detector output shape" }
      digitsOutput = rawDigits

      val digitDetections = SuppliedYoloDecoder.decodeAxisAlignedBoxes(
        rawOutput = rawDigits.data,
        anchors = SuppliedYoloDecoder.DETECT_ANCHORS,
        meta = digitsMeta,
        names = config.digits.names,
        confThres = config.digits.confThres,
        iouThres = config.digits.iouThres,
        agnosticNms = config.digits.agnosticNms,
        maxDet = config.digits.maxDet,
      )
      reading = SuppliedYoloDecoder.decodeDigitsReading(
        detections = digitDetections,
        classToDigit = config.digits.classToDigit,
        acceptedLengths = config.acceptedLengths,
      )
    }

    val frameDetectionsJson = JSONArray()
    frameDetections.forEach { d ->
      val polyJson = JSONArray()
      d.polygon.forEach { pt -> polyJson.put(JSONArray(pt)) }
      frameDetectionsJson.put(
        JSONObject()
          .put("anchor", d.anchor)
          .put("classId", d.classId)
          .put("class", d.label)
          .put("score", d.score.toDouble())
          .put("netObb", JSONArray(d.netObb))
          .put("polygon", polyJson)
      )
    }

    val revDetectionsJson = JSONArray()
    revDetections.forEach { d ->
      revDetectionsJson.put(
        JSONObject()
          .put("anchor", d.anchor)
          .put("classId", d.classId)
          .put("class", d.label)
          .put("score", d.score.toDouble())
          .put("box", JSONArray(d.box))
          .put("cx", d.cx.toDouble())
          .put("cy", d.cy.toDouble())
      )
    }

    val digitDetectionsJson = JSONArray()
    reading.sortedDetections.forEach { d ->
      digitDetectionsJson.put(
        JSONObject()
          .put("anchor", d.anchor)
          .put("classId", d.classId)
          .put("class", d.label)
          .put("digit", d.digit ?: JSONObject.NULL)
          .put("score", d.score.toDouble())
          .put("box", JSONArray(d.box))
          .put("cx", d.cx.toDouble())
          .put("cy", d.cy.toDouble())
      )
    }

    val orderedPolygonJson = cropPlan.orderedPolygon?.let { poly ->
      JSONArray().also { arr -> poly.forEach { pt -> arr.put(JSONArray(pt)) } }
    } ?: JSONObject.NULL

    val cropSizeJson = if (cropPlan.cropWidth > 0 && cropPlan.cropHeight > 0) {
      JSONArray(intArrayOf(cropPlan.cropWidth, cropPlan.cropHeight))
    } else {
      JSONObject.NULL
    }

    val finiteCount = frameOutput.data.size + (revOutput?.data?.size ?: 0) + (digitsOutput?.data?.size ?: 0)
    val report = JSONObject()
      .put("runtime", config.runtime)
      .put("pipelineId", config.pipelineId)
      .put("semanticsVersion", config.semanticsVersion)
      .put("decoderVersion", SuppliedYoloDecoder.SEMANTICS_VERSION)
      .put("preprocessVersion", config.preprocessVersion)
      .put("modelManifestSha256", manifestSha256)
      .put("frameModelSha256", config.frame.sha256)
      .put("reverseModelSha256", config.reverse.sha256)
      .put("digitsModelSha256", config.digits.sha256)
      .put("inputShape", JSONArray(frameShape))
      .put("outputShape", JSONArray(frameOutput.shape))
      .put("reverseInputShape", if (revInput == null) JSONObject.NULL else JSONArray(detectShape))
      .put("reverseOutputShape", revOutput?.let { JSONArray(it.shape) } ?: JSONObject.NULL)
      .put("digitsInputShape", if (digitsInput == null) JSONObject.NULL else JSONArray(detectShape))
      .put("digitsOutputShape", digitsOutput?.let { JSONArray(it.shape) } ?: JSONObject.NULL)
      .put("finiteOutputs", finiteCount)
      .put("invokeMs", (System.nanoTime() - start) / 1_000_000.0)
      .put("inputSource", if (saveParity) "parity-image" else "full-image")
      .put("source", "gallery")
      .put("authenticity", "unverified")
      .put("imagePixelSha256", hash(rgb))
      .put("cropPixelSha256", cropRgb?.let { hash(it) } ?: JSONObject.NULL)
      .put("cropUsedPixelSha256", cropUsedRgb?.let { hash(it) } ?: JSONObject.NULL)
      .put("status", reading.status)
      .put("rawDigits", reading.rawDigits ?: JSONObject.NULL)
      .put("reason", reading.reason ?: JSONObject.NULL)
      .put("confidence", reading.confidence)
      .put("rawDiagnostic", reading.rawDiagnostic)
      .put("isReversed", revDecision.isReversed)
      .put("reverseTriggerReason", revDecision.reason)
      .put("rowBox", cropPlan.rowBox?.let { JSONArray(it) } ?: JSONObject.NULL)
      .put("orderedPolygon", orderedPolygonJson)
      .put("cropSize", cropSizeJson)
      .put("imageSize", JSONArray(intArrayOf(width, height)))
      .put("frameCount", frameDetections.size)
      .put("frameDetections", frameDetectionsJson)
      .put("reverseDetections", revDetectionsJson)
      .put("revClasses", JSONArray(revDecision.classes))
      .put("digitDetections", digitDetectionsJson)
      .put("digitClassesLeftToRight", JSONArray(reading.digitClassesLeftToRight))
      .put("detections", digitDetectionsJson)

    if (saveParity) {
      val directory = File(context.filesDir, "ocr-parity/latest").apply { check(exists() || mkdirs()) }
      saveTensor(directory, "frame_input", frameInput)
      saveTensor(directory, "frame_output", frameOutput.data)
      if (revInput != null && revOutput != null && digitsInput != null && digitsOutput != null) {
        saveTensor(directory, "reverse_input", revInput)
        saveTensor(directory, "reverse_output", revOutput.data)
        saveTensor(directory, "digits_input", digitsInput)
        saveTensor(directory, "digits_output", digitsOutput.data)
      } else {
        for (stale in listOf("reverse_input.f32", "reverse_output.f32", "digits_input.f32", "digits_output.f32")) {
          File(directory, stale).delete()
        }
      }
      report.put("runId", UUID.randomUUID().toString()).put("tensorEncoding", "float32-le")
      File(directory, "metadata.json").writeText(report.toString(2))
      report.put("parityFixture", "files/ocr-parity/latest")
    }

    return report.toString()
  }

  private fun saveTensor(directory: File, name: String, values: FloatArray) {
    val bytes = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    values.forEach { bytes.putFloat(it) }
    File(directory, "$name.f32").writeBytes(bytes.array())
  }

  private fun hash(bytes: ByteArray): String =
    MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

  private fun hashFile(file: File): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
      val buffer = ByteArray(64 * 1024)
      while (true) {
        val count = input.read(buffer)
        if (count < 0) break
        digest.update(buffer, 0, count)
      }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
  }

  override fun close() {
    var failure: Throwable? = null
    for (session in listOf(digitsSession, reverseSession, frameSession)) {
      try {
        session.close()
      } catch (error: Throwable) {
        if (failure == null) failure = error
      }
    }
    if (failure != null) throw failure
  }
}
