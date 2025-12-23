package com.amigoinovasi.fishgo.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Fish detection using YOLOv8 TFLite model
 * Uses core TFLite Interpreter (not Task Vision)
 */
class FishDetector(context: Context) {

    private val interpreter: Interpreter
    private val labels: List<String>
    private val indonesianNames: Map<String, String>

    // YOLOv8 model expects 640x640 input
    private val inputSize = INPUT_SIZE
    private val inputBuffer: ByteBuffer

    init {
        // Load labels
        labels = loadLabels(context)

        // Map scientific names to Indonesian names
        indonesianNames = mapOf(
            "Alepes Djedaba -Ikan Selar Bulat-" to "Selar Bulat",
            "Atropus Atropos -Ikan Cipa-Cipa-" to "Cipa-Cipa",
            "Caranx Ignobilis -Ikan Kuwe-" to "Kuwe",
            "Chanos Chanos -Ikan Bandeng-" to "Bandeng",
            "Decapterus Macarellus -Ikan Malalugis-" to "Malalugis",
            "Euthynnus Affinis -Ikan Bonito Tuna-" to "Tongkol",
            "Katsuwonus Pelamis -Ikan Cakalang-" to "Cakalang",
            "Lutjanus Malabaricus -Ikan Kakap Merah-" to "Kakap Merah",
            "Parastromateus Niger -Ikan Bawal Hitam-" to "Bawal Hitam",
            "Rastrelliger Kanagurta -Ikan Kembung-" to "Kembung",
            "Rastrelliger sp -Ikan Kembung Banjar-" to "Kembung Banjar",
            "Scaridae -Ikan Kakatua-" to "Kakatua",
            "Scomber Japonicus -Ikan Salem-" to "Salem",
            "Scomberomorus Guttatus -Tenggiri Papan-" to "Tenggiri Papan",
            "Thunnus Alalunga -Ikan Albacore Tuna-" to "Albacore Tuna",
            "Thunnus Obesus -Ikan Bigeye Tuna-" to "Tuna Mata Besar",
            "Thunnus Tonggol -Ikan Tuna-" to "Tuna",
            "Tribus Sardini -Ikan Kenyar-" to "Kenyar",
            "Upeneus Moluccensis -Ikan Kuniran-" to "Kuniran"
        )

        // Allocate input buffer (Float32: 1 x 640 x 640 x 3)
        inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Load model and create interpreter
        val model = loadModelFile(context)
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(model, options)

        Log.d(TAG, "FishDetector initialized with ${labels.size} classes")
        logModelInfo()
    }

    private fun logModelInfo() {
        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)
        Log.d(TAG, "Input shape: ${inputTensor.shape().contentToString()}")
        Log.d(TAG, "Output shape: ${outputTensor.shape().contentToString()}")
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadLabels(context: Context): List<String> {
        return try {
            context.assets.open(LABELS_FILE).bufferedReader().readLines()
                .filter { it.isNotBlank() }
        } catch (e: Exception) {
            Log.w(TAG, "Could not load labels file, using default")
            listOf(
                "Alepes Djedaba -Ikan Selar Bulat-",
                "Atropus Atropos -Ikan Cipa-Cipa-",
                "Caranx Ignobilis -Ikan Kuwe-",
                "Chanos Chanos -Ikan Bandeng-",
                "Decapterus Macarellus -Ikan Malalugis-",
                "Euthynnus Affinis -Ikan Bonito Tuna-",
                "Katsuwonus Pelamis -Ikan Cakalang-",
                "Lutjanus Malabaricus -Ikan Kakap Merah-",
                "Parastromateus Niger -Ikan Bawal Hitam-",
                "Rastrelliger Kanagurta -Ikan Kembung-",
                "Rastrelliger sp -Ikan Kembung Banjar-",
                "Scaridae -Ikan Kakatua-",
                "Scomber Japonicus -Ikan Salem-",
                "Scomberomorus Guttatus -Tenggiri Papan-",
                "Thunnus Alalunga -Ikan Albacore Tuna-",
                "Thunnus Obesus -Ikan Bigeye Tuna-",
                "Thunnus Tonggol -Ikan Tuna-",
                "Tribus Sardini -Ikan Kenyar-",
                "Upeneus Moluccensis -Ikan Kuniran-"
            )
        }
    }

    /**
     * Detect fish in a Bitmap
     */
    fun detect(bitmap: Bitmap): List<DetectionResult> {
        // Resize to model input size
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // Prepare input buffer
        inputBuffer.rewind()
        val pixels = IntArray(inputSize * inputSize)
        resizedBitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            // Normalize to 0-1 range (YOLOv8 expects normalized input)
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
            inputBuffer.putFloat((pixel and 0xFF) / 255.0f)          // B
        }

        // YOLOv8 output shape: [1, 23, 8400] (4 box coords + 19 classes = 23)
        // Or transposed: [1, 8400, 23]
        val outputShape = interpreter.getOutputTensor(0).shape()
        val numDetections = if (outputShape[1] > outputShape[2]) outputShape[1] else outputShape[2]
        val numClasses = labels.size
        val outputSize = outputShape[1] * outputShape[2]

        val outputBuffer = ByteBuffer.allocateDirect(outputSize * 4)
        outputBuffer.order(ByteOrder.nativeOrder())

        // Run inference
        interpreter.run(inputBuffer, outputBuffer)

        // Parse YOLOv8 output
        outputBuffer.rewind()
        val output = FloatArray(outputSize)
        outputBuffer.asFloatBuffer().get(output)

        return parseYoloOutput(output, outputShape, bitmap.width, bitmap.height)
    }

    private fun parseYoloOutput(
        output: FloatArray,
        shape: IntArray,
        imageWidth: Int,
        imageHeight: Int
    ): List<DetectionResult> {
        val results = mutableListOf<DetectionResult>()
        val numClasses = labels.size

        // YOLOv8 output: [1, 23, 8400]
        // Row 0-3: x_center, y_center, width, height (normalized)
        // Row 4-22: class scores
        val numPredictions = shape[2] // 8400
        val stride = shape[1] // 23

        for (i in 0 until numPredictions) {
            // Get box coordinates
            val xCenter = output[0 * numPredictions + i]
            val yCenter = output[1 * numPredictions + i]
            val width = output[2 * numPredictions + i]
            val height = output[3 * numPredictions + i]

            // Find best class
            var maxScore = 0f
            var maxClassIdx = 0
            for (c in 0 until numClasses) {
                val score = output[(4 + c) * numPredictions + i]
                if (score > maxScore) {
                    maxScore = score
                    maxClassIdx = c
                }
            }

            // Filter by confidence
            if (maxScore >= CONFIDENCE_THRESHOLD) {
                // Convert from center-based to corner-based coordinates
                val left = (xCenter - width / 2) / inputSize
                val top = (yCenter - height / 2) / inputSize
                val right = (xCenter + width / 2) / inputSize
                val bottom = (yCenter + height / 2) / inputSize

                // Clamp to [0, 1]
                val boundingBox = RectF(
                    left.coerceIn(0f, 1f),
                    top.coerceIn(0f, 1f),
                    right.coerceIn(0f, 1f),
                    bottom.coerceIn(0f, 1f)
                )

                val label = if (maxClassIdx < labels.size) labels[maxClassIdx] else "Unknown"
                val indonesianName = indonesianNames[label] ?: extractIndonesianName(label)

                results.add(
                    DetectionResult(
                        label = label,
                        indonesianName = indonesianName,
                        confidence = maxScore,
                        boundingBox = boundingBox
                    )
                )
            }
        }

        // Non-maximum suppression
        return nms(results, NMS_THRESHOLD)
    }

    private fun nms(detections: List<DetectionResult>, iouThreshold: Float): List<DetectionResult> {
        if (detections.isEmpty()) return emptyList()

        val sorted = detections.sortedByDescending { it.confidence }.toMutableList()
        val selected = mutableListOf<DetectionResult>()

        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            selected.add(best)

            sorted.removeAll { detection ->
                iou(best.boundingBox, detection.boundingBox) > iouThreshold
            }
        }

        return selected.take(MAX_RESULTS)
    }

    private fun iou(box1: RectF, box2: RectF): Float {
        val intersectLeft = maxOf(box1.left, box2.left)
        val intersectTop = maxOf(box1.top, box2.top)
        val intersectRight = minOf(box1.right, box2.right)
        val intersectBottom = minOf(box1.bottom, box2.bottom)

        val intersectArea = maxOf(0f, intersectRight - intersectLeft) *
                maxOf(0f, intersectBottom - intersectTop)

        val box1Area = (box1.right - box1.left) * (box1.bottom - box1.top)
        val box2Area = (box2.right - box2.left) * (box2.bottom - box2.top)

        val unionArea = box1Area + box2Area - intersectArea

        return if (unionArea > 0f) intersectArea / unionArea else 0f
    }

    private fun extractIndonesianName(label: String): String {
        val regex = "-Ikan ([^-]+)-".toRegex()
        val match = regex.find(label)
        return match?.groupValues?.get(1) ?: label.substringAfterLast("-Ikan ")
            .removeSuffix("-")
            .trim()
            .ifEmpty { label }
    }

    fun close() {
        interpreter.close()
    }

    companion object {
        private const val TAG = "FishDetector"
        private const val MODEL_FILE = "model/fish_detector.tflite"
        private const val LABELS_FILE = "model/labels.txt"
        private const val INPUT_SIZE = 640
        private const val MAX_RESULTS = 10
        private const val CONFIDENCE_THRESHOLD = 0.5f
        private const val NMS_THRESHOLD = 0.45f
    }
}
