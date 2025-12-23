package com.amigoinovasi.fishgo

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.amigoinovasi.fishgo.databinding.ActivityMainBinding
import com.amigoinovasi.fishgo.detection.FishDetector
import com.amigoinovasi.fishgo.detection.DetectionResult
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var fishDetector: FishDetector? = null
    private var imageAnalyzer: ImageAnalysis? = null

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

        // Initialize detector
        binding.statusText.text = getString(R.string.model_loading)
        initializeDetector()

        // Check camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initializeDetector() {
        try {
            fishDetector = FishDetector(this)
            binding.statusText.text = getString(R.string.detecting)
            Log.d(TAG, "Fish detector initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize detector", e)
            binding.statusText.text = "Model load failed: ${e.message}"
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .build()
                .also {
                    it.setSurfaceProvider(binding.previewView.surfaceProvider)
                }

            // Image Analysis for detection
            imageAnalyzer = ImageAnalysis.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy)
                    }
                }

            // Select back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalyzer
                )
                Log.d(TAG, "Camera started successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        val detector = fishDetector
        if (detector == null) {
            imageProxy.close()
            return
        }

        try {
            val results = detector.detect(imageProxy)

            runOnUiThread {
                // Update overlay with detection results
                binding.overlayView.setResults(
                    results,
                    imageProxy.width,
                    imageProxy.height
                )

                // Update detection count
                val fishCount = results.size
                if (fishCount > 0) {
                    val fishNames = results.take(3).joinToString(", ") { it.label }
                    binding.detectionCountText.text = "Detected: $fishCount ($fishNames)"
                } else {
                    binding.detectionCountText.text = "No fish detected"
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Detection failed", e)
        } finally {
            imageProxy.close()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        fishDetector?.close()
    }

    companion object {
        private const val TAG = "FishGo"
    }
}
