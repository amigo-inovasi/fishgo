package com.amigoinovasi.fishgo.classifier

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * TFLite 기반 물고기 분류기
 * - 정확도 우선 설정
 * - CPU 사용 (호환성)
 * - 1~2초 소요 허용
 *
 * 모델: EfficientNet-B0 (FP16)
 * 입력: 224x224x3 RGB (ImageNet normalized)
 * 출력: 19 클래스 확률 (softmax)
 */
class FishClassifier(private val context: Context) {

    private val interpreter: Interpreter
    private val labels: List<FishLabel>

    private val inputSize = INPUT_SIZE
    private val numClasses = NUM_CLASSES

    // 입력 버퍼 (재사용)
    private val inputBuffer: ByteBuffer

    // ImageNet normalization 값
    private val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val std = floatArrayOf(0.229f, 0.224f, 0.225f)

    init {
        // 입력 버퍼 할당
        inputBuffer = ByteBuffer.allocateDirect(1 * inputSize * inputSize * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        // 모델 로드
        val modelBuffer = loadModelFile(MODEL_FILE)

        // 정확도 우선 옵션
        val options = Interpreter.Options().apply {
            setNumThreads(4)  // 멀티스레드
            // GPU delegate 미사용 (호환성 + 정확도)
        }

        interpreter = Interpreter(modelBuffer, options)

        // 레이블 로드
        labels = loadLabels(LABELS_FILE)

        Log.d(TAG, "FishClassifier initialized with ${labels.size} classes")
        logModelInfo()
    }

    private fun logModelInfo() {
        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)
        Log.d(TAG, "Input shape: ${inputTensor.shape().contentToString()}")
        Log.d(TAG, "Output shape: ${outputTensor.shape().contentToString()}")
    }

    /**
     * 이미지 분류 수행
     *
     * @param bitmap 입력 이미지 (모든 크기 가능, 내부에서 리사이즈)
     * @return 분류 결과 (클래스명, 인도네시아명, 신뢰도)
     */
    fun classify(bitmap: Bitmap): ClassificationResult {
        // 1. 이미지 전처리
        val inputBuffer = preprocessImage(bitmap)

        // 2. 출력 버퍼 준비
        val outputBuffer = Array(1) { FloatArray(numClasses) }

        // 3. 추론 실행
        interpreter.run(inputBuffer, outputBuffer)

        // 4. 결과 파싱 (softmax 적용됨)
        val probabilities = outputBuffer[0]
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: 0
        val confidence = probabilities[maxIndex]

        // 레이블이 충분하지 않은 경우 처리
        val label = if (maxIndex < labels.size) {
            labels[maxIndex]
        } else {
            FishLabel("Unknown", "Tidak diketahui")
        }

        Log.d(TAG, "Classification: ${label.scientificName} (${(confidence * 100).toInt()}%)")

        return ClassificationResult(
            scientificName = label.scientificName,
            indonesianName = label.indonesianName,
            confidence = confidence
        )
    }

    /**
     * 이미지 전처리
     * - 224x224 리사이즈
     * - RGB 정규화: (pixel / 255.0 - mean) / std
     * - ImageNet 기준 mean/std 사용
     */
    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        inputBuffer.rewind()

        // 224x224로 리사이즈
        val resized = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        val pixels = IntArray(inputSize * inputSize)
        resized.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // 픽셀 정규화
        for (pixel in pixels) {
            val r = ((pixel shr 16 and 0xFF) / 255.0f - mean[0]) / std[0]
            val g = ((pixel shr 8 and 0xFF) / 255.0f - mean[1]) / std[1]
            val b = ((pixel and 0xFF) / 255.0f - mean[2]) / std[2]

            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    /**
     * 모델 파일 로드
     */
    private fun loadModelFile(filename: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * 레이블 파일 로드
     * 형식: "scientific_name,indonesian_name"
     */
    private fun loadLabels(filename: String): List<FishLabel> {
        return try {
            context.assets.open(filename).bufferedReader().readLines()
                .filter { it.isNotBlank() }
                .map { line ->
                    val parts = line.split(",")
                    if (parts.size >= 2) {
                        FishLabel(parts[0].trim(), parts[1].trim())
                    } else {
                        FishLabel(parts[0].trim(), parts[0].trim())
                    }
                }
        } catch (e: Exception) {
            Log.w(TAG, "Could not load labels file, using default")
            getDefaultLabels()
        }
    }

    /**
     * 기본 레이블 (labels.txt 로드 실패 시)
     */
    private fun getDefaultLabels(): List<FishLabel> {
        return listOf(
            FishLabel("Alepes Djedaba", "Selar Bulat"),
            FishLabel("Atropus Atropos", "Cipa-Cipa"),
            FishLabel("Caranx Ignobilis", "Kuwe"),
            FishLabel("Chanos Chanos", "Bandeng"),
            FishLabel("Decapterus Macarellus", "Malalugis"),
            FishLabel("Euthynnus Affinis", "Tongkol"),
            FishLabel("Katsuwonus Pelamis", "Cakalang"),
            FishLabel("Lutjanus Malabaricus", "Kakap Merah"),
            FishLabel("Parastromateus Niger", "Bawal Hitam"),
            FishLabel("Rastrelliger Kanagurta", "Kembung"),
            FishLabel("Rastrelliger sp", "Kembung Banjar"),
            FishLabel("Scaridae", "Kakatua"),
            FishLabel("Scomber Japonicus", "Salem"),
            FishLabel("Scomberomorus Guttatus", "Tenggiri Papan"),
            FishLabel("Thunnus Alalunga", "Albacore Tuna"),
            FishLabel("Thunnus Obesus", "Tuna Mata Besar"),
            FishLabel("Thunnus Tonggol", "Tuna"),
            FishLabel("Tribus Sardini", "Kenyar"),
            FishLabel("Upeneus Moluccensis", "Kuniran")
        )
    }

    /**
     * 리소스 해제
     */
    fun close() {
        interpreter.close()
    }

    companion object {
        private const val TAG = "FishClassifier"
        private const val MODEL_FILE = "model/fish_classifier.tflite"
        private const val LABELS_FILE = "model/labels.txt"
        private const val INPUT_SIZE = 224
        private const val NUM_CLASSES = 19
    }
}
