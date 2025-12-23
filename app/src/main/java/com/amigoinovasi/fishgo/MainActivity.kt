package com.amigoinovasi.fishgo

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.amigoinovasi.fishgo.databinding.ActivityMainBinding
import com.amigoinovasi.fishgo.ui.FishGuideOverlay
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * FishGo 메인 카메라 화면
 *
 * Classification 방식:
 * 1. 사용자가 물고기를 가이드 프레임 안에 맞춤
 * 2. "촬영하기" 버튼 클릭
 * 3. 가이드 영역만 crop하여 ResultActivity로 전달
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService

    private var imageCapture: ImageCapture? = null
    private var preview: Preview? = null

    // 가이드 프레임 비율 (FishGuideOverlay와 동일)
    private val guideWidthPercent = 0.85f
    private val guideAspectRatio = 3f / 1f

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, getString(R.string.permission_denied), Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        setupCaptureButton()

        // 카메라 권한 확인
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setupCaptureButton() {
        binding.btnCapture.setOnClickListener {
            captureAndAnalyze()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview
            preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also {
                    it.setSurfaceProvider(binding.previewView.surfaceProvider)
                }

            // ImageCapture
            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            // 후면 카메라 선택
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageCapture
                )
                Log.d(TAG, "Camera started successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun captureAndAnalyze() {
        val imageCapture = imageCapture ?: return

        // 로딩 표시
        showLoading(true)

        // 프리뷰에서 비트맵 가져오기 (더 빠름)
        val previewBitmap = binding.previewView.bitmap

        if (previewBitmap != null) {
            processAndNavigate(previewBitmap)
        } else {
            // 폴백: ImageCapture 사용
            imageCapture.takePicture(
                ContextCompat.getMainExecutor(this),
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(image: ImageProxy) {
                        val bitmap = image.toBitmap()
                        image.close()
                        processAndNavigate(bitmap)
                    }

                    override fun onError(exception: ImageCaptureException) {
                        Log.e(TAG, "Image capture failed", exception)
                        showLoading(false)
                        Toast.makeText(
                            this@MainActivity,
                            getString(R.string.error_image_load),
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                }
            )
        }
    }

    private fun processAndNavigate(bitmap: Bitmap) {
        cameraExecutor.execute {
            try {
                // 가이드 영역만 crop
                val croppedBitmap = cropGuideRegion(bitmap)

                // 직접 224x224로 리사이즈 (패딩 없이 - 학습 데이터와 일치)
                val resizedBitmap = Bitmap.createScaledBitmap(croppedBitmap, 224, 224, true)

                // 임시 파일로 저장
                val imagePath = saveTempImage(resizedBitmap)

                runOnUiThread {
                    showLoading(false)

                    // ResultActivity로 이동
                    val intent = Intent(this@MainActivity, ResultActivity::class.java)
                    intent.putExtra(ResultActivity.EXTRA_IMAGE_PATH, imagePath)
                    startActivity(intent)
                }
            } catch (e: Exception) {
                Log.e(TAG, "Image processing failed", e)
                runOnUiThread {
                    showLoading(false)
                    Toast.makeText(
                        this@MainActivity,
                        getString(R.string.error_image_load),
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }
    }

    /**
     * 카메라 이미지에서 가이드 프레임 영역만 crop
     */
    private fun cropGuideRegion(bitmap: Bitmap): Bitmap {
        // 가이드 영역 계산 (화면 중앙, 3:1 비율)
        val guideWidth = (bitmap.width * guideWidthPercent).toInt()
        val guideHeight = (guideWidth / guideAspectRatio).toInt()
        val left = (bitmap.width - guideWidth) / 2
        val top = (bitmap.height - guideHeight) / 2

        // 경계 검사
        val safeLeft = left.coerceAtLeast(0)
        val safeTop = top.coerceAtLeast(0)
        val safeWidth = guideWidth.coerceAtMost(bitmap.width - safeLeft)
        val safeHeight = guideHeight.coerceAtMost(bitmap.height - safeTop)

        return Bitmap.createBitmap(bitmap, safeLeft, safeTop, safeWidth, safeHeight)
    }


    /**
     * 비트맵을 임시 파일로 저장
     */
    private fun saveTempImage(bitmap: Bitmap): String {
        val tempFile = File(cacheDir, "temp_fish_${System.currentTimeMillis()}.jpg")
        FileOutputStream(tempFile).use { out ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
        }
        return tempFile.absolutePath
    }

    private fun showLoading(show: Boolean) {
        binding.progressLoading.visibility = if (show) View.VISIBLE else View.GONE
        binding.btnCapture.isEnabled = !show
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    companion object {
        private const val TAG = "FishGo"
    }
}
