package com.amigoinovasi.fishgo.detection

import android.graphics.RectF

/**
 * Represents a single fish detection result
 */
data class DetectionResult(
    val label: String,
    val indonesianName: String,
    val confidence: Float,
    val boundingBox: RectF
) {
    /**
     * Returns the display name (Indonesian name preferred)
     */
    val displayName: String
        get() = indonesianName.ifEmpty { label }

    /**
     * Returns confidence as percentage string
     */
    val confidencePercent: String
        get() = "${(confidence * 100).toInt()}%"
}
