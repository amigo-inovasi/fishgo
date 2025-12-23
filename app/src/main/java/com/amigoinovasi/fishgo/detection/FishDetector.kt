package com.amigoinovasi.fishgo.detection

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector

/**
 * Fish detection using TensorFlow Lite
 * Designed for integration into Navigo app
 */
class FishDetector(context: Context) {

    private val detector: ObjectDetector
    private val labels: List<String>
    private val indonesianNames: Map<String, String>

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

        // Initialize TFLite Object Detector
        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setMaxResults(MAX_RESULTS)
            .setScoreThreshold(CONFIDENCE_THRESHOLD)
            .build()

        detector = ObjectDetector.createFromFileAndOptions(
            context,
            MODEL_FILE,
            options
        )

        Log.d(TAG, "FishDetector initialized with ${labels.size} classes")
    }

    private fun loadLabels(context: Context): List<String> {
        return try {
            context.assets.open(LABELS_FILE).bufferedReader().readLines()
        } catch (e: Exception) {
            Log.w(TAG, "Could not load labels file, using default")
            // Return default labels from dataset
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
     * Detect fish in an ImageProxy from CameraX
     */
    fun detect(imageProxy: ImageProxy): List<DetectionResult> {
        val bitmap = imageProxy.toBitmap()
        return detect(bitmap)
    }

    /**
     * Detect fish in a Bitmap
     */
    fun detect(bitmap: Bitmap): List<DetectionResult> {
        val tensorImage = TensorImage.fromBitmap(bitmap)
        val detections = detector.detect(tensorImage)

        return detections.mapNotNull { detection ->
            convertToResult(detection, bitmap.width, bitmap.height)
        }
    }

    private fun convertToResult(
        detection: Detection,
        imageWidth: Int,
        imageHeight: Int
    ): DetectionResult? {
        val category = detection.categories.firstOrNull() ?: return null
        val label = category.label ?: return null
        val confidence = category.score

        if (confidence < CONFIDENCE_THRESHOLD) return null

        val boundingBox = detection.boundingBox

        // Normalize bounding box to 0-1 range for overlay drawing
        val normalizedBox = RectF(
            boundingBox.left / imageWidth,
            boundingBox.top / imageHeight,
            boundingBox.right / imageWidth,
            boundingBox.bottom / imageHeight
        )

        val indonesianName = indonesianNames[label] ?: extractIndonesianName(label)

        return DetectionResult(
            label = label,
            indonesianName = indonesianName,
            confidence = confidence,
            boundingBox = normalizedBox
        )
    }

    /**
     * Extract Indonesian name from label like "Scientific Name -Ikan XYZ-"
     */
    private fun extractIndonesianName(label: String): String {
        val regex = "-Ikan ([^-]+)-".toRegex()
        val match = regex.find(label)
        return match?.groupValues?.get(1) ?: label.substringAfterLast("-Ikan ")
            .removeSuffix("-")
            .trim()
            .ifEmpty { label }
    }

    fun close() {
        detector.close()
    }

    companion object {
        private const val TAG = "FishDetector"
        private const val MODEL_FILE = "model/fish_detector.tflite"
        private const val LABELS_FILE = "model/labels.txt"
        private const val MAX_RESULTS = 10
        private const val CONFIDENCE_THRESHOLD = 0.5f
    }
}
