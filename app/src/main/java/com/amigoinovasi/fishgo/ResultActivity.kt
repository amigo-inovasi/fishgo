package com.amigoinovasi.fishgo

import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.amigoinovasi.fishgo.classifier.ClassificationResult
import com.amigoinovasi.fishgo.classifier.ConfidenceLevel
import com.amigoinovasi.fishgo.classifier.FishClassifier
import com.amigoinovasi.fishgo.databinding.ActivityResultBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

/**
 * 물고기 분류 결과 화면
 *
 * - 촬영된 이미지 표시
 * - 분류 결과 (인도네시아어 이름 + 학명)
 * - 신뢰도 표시 (프로그레스 바 + 퍼센트)
 * - 다시 촬영 / 저장 버튼
 */
class ResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultBinding
    private var classifier: FishClassifier? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 이미지 경로 받기
        val imagePath = intent.getStringExtra(EXTRA_IMAGE_PATH)

        if (imagePath != null) {
            initializeClassifier()
            analyzeImage(imagePath)
        } else {
            showError("이미지를 불러올 수 없습니다")
        }

        setupButtons()
    }

    private fun initializeClassifier() {
        try {
            classifier = FishClassifier(this)
        } catch (e: Exception) {
            showError("모델 로드 실패: ${e.message}")
        }
    }

    private fun analyzeImage(imagePath: String) {
        // 로딩 표시
        showLoading(true)

        lifecycleScope.launch(Dispatchers.Default) {
            try {
                // 이미지 로드
                val bitmap = BitmapFactory.decodeFile(imagePath)

                if (bitmap == null) {
                    withContext(Dispatchers.Main) {
                        showError("이미지를 불러올 수 없습니다")
                    }
                    return@launch
                }

                // 분류 수행 (1~2초 소요 가능)
                val result = classifier?.classify(bitmap)

                withContext(Dispatchers.Main) {
                    if (result != null) {
                        displayResult(bitmap, result)
                    } else {
                        showError("분류 실패")
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    showError("분석 오류: ${e.message}")
                }
            }
        }
    }

    private fun displayResult(bitmap: android.graphics.Bitmap, result: ClassificationResult) {
        showLoading(false)

        // 촬영된 이미지 표시
        binding.ivCapturedFish.setImageBitmap(bitmap)

        // 결과 표시
        binding.tvIndonesianName.text = result.indonesianName
        binding.tvScientificName.text = result.scientificName
        binding.tvConfidence.text = result.confidencePercent
        binding.progressConfidence.progress = (result.confidence * 100).toInt()

        // 신뢰도에 따른 색상
        val confidenceColor = when (result.confidenceLevel) {
            ConfidenceLevel.HIGH -> Color.parseColor("#4CAF50")    // 초록
            ConfidenceLevel.MEDIUM -> Color.parseColor("#FF9800") // 주황
            ConfidenceLevel.LOW -> Color.parseColor("#F44336")    // 빨강
        }
        binding.progressConfidence.setIndicatorColor(confidenceColor)

        // 낮은 신뢰도 경고
        if (result.confidenceLevel == ConfidenceLevel.LOW) {
            binding.tvWarning.visibility = View.VISIBLE
            binding.tvWarning.text = getString(R.string.result_low_confidence_warning)
        } else {
            binding.tvWarning.visibility = View.GONE
        }

        // 결과 카드 표시
        binding.resultCard.visibility = View.VISIBLE
    }

    private fun showLoading(show: Boolean) {
        binding.progressBar.visibility = if (show) View.VISIBLE else View.GONE
        binding.resultCard.visibility = if (show) View.GONE else View.VISIBLE
    }

    private fun showError(message: String) {
        showLoading(false)
        binding.resultCard.visibility = View.GONE
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        // 에러 시 카메라 화면으로 돌아가기
        finish()
    }

    private fun setupButtons() {
        // 다시 촬영 버튼
        binding.btnRetake.setOnClickListener {
            // 임시 파일 삭제
            intent.getStringExtra(EXTRA_IMAGE_PATH)?.let { path ->
                File(path).delete()
            }
            finish()
        }

        // 저장 버튼
        binding.btnSave.setOnClickListener {
            // TODO: 결과 저장 기능 구현
            Toast.makeText(this, "Tersimpan!", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        classifier?.close()

        // 임시 파일 삭제
        intent.getStringExtra(EXTRA_IMAGE_PATH)?.let { path ->
            File(path).delete()
        }
    }

    companion object {
        const val EXTRA_IMAGE_PATH = "image_path"
    }
}
