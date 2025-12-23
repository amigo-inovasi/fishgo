package com.amigoinovasi.fishgo.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.amigoinovasi.fishgo.R
import com.amigoinovasi.fishgo.detection.DetectionResult

/**
 * Custom view for drawing detection bounding boxes over camera preview
 * Designed for easy integration into Navigo app
 */
class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var results: List<DetectionResult> = emptyList()
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1

    // Box paint
    private val boxPaint = Paint().apply {
        color = context.getColor(R.color.detection_box)
        style = Paint.Style.STROKE
        strokeWidth = 6f
        isAntiAlias = true
    }

    // Text background paint
    private val textBackgroundPaint = Paint().apply {
        color = Color.argb(200, 0, 0, 0)
        style = Paint.Style.FILL
    }

    // Text paint
    private val textPaint = Paint().apply {
        color = context.getColor(R.color.detection_text)
        textSize = 42f
        typeface = Typeface.DEFAULT_BOLD
        isAntiAlias = true
    }

    // Confidence text paint
    private val confidencePaint = Paint().apply {
        color = Color.WHITE
        textSize = 36f
        typeface = Typeface.DEFAULT
        isAntiAlias = true
    }

    private val bounds = Rect()

    /**
     * Update detection results to be drawn
     */
    fun setResults(
        detectionResults: List<DetectionResult>,
        imageWidth: Int,
        imageHeight: Int
    ) {
        this.results = detectionResults
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        invalidate()
    }

    fun clear() {
        results = emptyList()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val viewWidth = width.toFloat()
        val viewHeight = height.toFloat()

        for (result in results) {
            drawDetection(canvas, result, viewWidth, viewHeight)
        }
    }

    private fun drawDetection(
        canvas: Canvas,
        result: DetectionResult,
        viewWidth: Float,
        viewHeight: Float
    ) {
        // Scale normalized bounding box to view coordinates
        // Note: For front camera, you might need to mirror the coordinates
        val box = result.boundingBox

        val left = box.left * viewWidth
        val top = box.top * viewHeight
        val right = box.right * viewWidth
        val bottom = box.bottom * viewHeight

        // Draw bounding box
        canvas.drawRect(left, top, right, bottom, boxPaint)

        // Prepare label text
        val labelText = result.displayName
        val confidenceText = result.confidencePercent

        // Measure text
        textPaint.getTextBounds(labelText, 0, labelText.length, bounds)
        val textWidth = bounds.width()
        val textHeight = bounds.height()

        // Draw text background
        val padding = 12f
        val bgLeft = left
        val bgTop = top - textHeight - padding * 3
        val bgRight = left + textWidth + padding * 2
        val bgBottom = top

        // Ensure background is within view bounds
        val adjustedBgTop = if (bgTop < 0) bottom else bgTop
        val adjustedBgBottom = if (bgTop < 0) bottom + textHeight + padding * 3 else bgBottom

        canvas.drawRoundRect(
            bgLeft,
            adjustedBgTop,
            bgRight + 80, // Extra space for confidence
            adjustedBgBottom,
            8f,
            8f,
            textBackgroundPaint
        )

        // Draw label text
        val textY = if (bgTop < 0) {
            bottom + textHeight + padding
        } else {
            top - padding
        }

        canvas.drawText(labelText, left + padding, textY, textPaint)

        // Draw confidence next to label
        canvas.drawText(
            confidenceText,
            left + padding + textWidth + 16,
            textY,
            confidencePaint
        )
    }
}
