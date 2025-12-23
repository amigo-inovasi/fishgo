package com.amigoinovasi.fishgo

import android.Manifest
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
import com.amigoinovasi.fishgo.detection.FishDetector
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var fishDetector: FishDetector? = null
    private var imageCapture: ImageCapture? = null
    private var preview: Preview? = null

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

        // Setup capture button
        binding.captureButton.setOnClickListener {
            captureAndAnalyze()
        }

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
        cameraExecutor.execute {
            try {
                fishDetector = FishDetector(this)
                runOnUiThread {
                    binding.statusText.text = getString(R.string.ready_to_scan)
                    binding.captureButton.isEnabled = true
                    Log.d(TAG, "Fish detector initialized successfully")
                }
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize detector", e)
                runOnUiThread {
                    binding.statusText.text = "Model load failed: ${e.message}"
                    binding.captureButton.isEnabled = false
                }
            }
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

            // ImageCapture for taking photos
            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_4_3)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            // Select back camera
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
        val detector = fishDetector ?: return

        // Show loading state
        binding.captureButton.isEnabled = false
        binding.progressBar.visibility = View.VISIBLE
        binding.statusText.text = getString(R.string.analyzing)
        binding.detectionCountText.visibility = View.GONE
        binding.overlayView.clearResults()

        // Capture image from preview
        val bitmap = binding.previewView.bitmap
        if (bitmap != null) {
            analyzeImage(bitmap)
        } else {
            // Fallback: use ImageCapture
            imageCapture.takePicture(
                ContextCompat.getMainExecutor(this),
                object : ImageCapture.OnImageCapturedCallback() {
                    override fun onCaptureSuccess(imageProxy: ImageProxy) {
                        val capturedBitmap = imageProxyToBitmap(imageProxy)
                        imageProxy.close()
                        if (capturedBitmap != null) {
                            analyzeImage(capturedBitmap)
                        } else {
                            showError("Failed to capture image")
                        }
                    }

                    override fun onError(exception: ImageCaptureException) {
                        Log.e(TAG, "Image capture failed", exception)
                        showError("Capture failed: ${exception.message}")
                    }
                }
            )
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val buffer = imageProxy.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            android.graphics.BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to convert ImageProxy to Bitmap", e)
            null
        }
    }

    private fun analyzeImage(bitmap: Bitmap) {
        cameraExecutor.execute {
            try {
                val detector = fishDetector ?: return@execute
                val results = detector.detect(bitmap)

                runOnUiThread {
                    binding.progressBar.visibility = View.GONE
                    binding.captureButton.isEnabled = true

                    if (results.isNotEmpty()) {
                        // Show detection results
                        binding.overlayView.setResults(results, bitmap.width, bitmap.height)

                        val fishCount = results.size
                        val topResult = results.first()
                        val confidence = (topResult.confidence * 100).toInt()

                        binding.detectionCountText.text = if (fishCount == 1) {
                            "${topResult.indonesianName} ($confidence%)"
                        } else {
                            "$fishCount fish detected\n${topResult.indonesianName} ($confidence%)"
                        }
                        binding.detectionCountText.visibility = View.VISIBLE
                        binding.statusText.text = "Detection complete"
                    } else {
                        binding.statusText.text = getString(R.string.no_fish_detected)
                        binding.detectionCountText.visibility = View.GONE
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Detection failed", e)
                runOnUiThread {
                    showError("Detection failed: ${e.message}")
                }
            }
        }
    }

    private fun showError(message: String) {
        binding.progressBar.visibility = View.GONE
        binding.captureButton.isEnabled = true
        binding.statusText.text = message
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
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
