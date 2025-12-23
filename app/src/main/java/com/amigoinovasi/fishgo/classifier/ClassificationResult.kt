package com.amigoinovasi.fishgo.classifier

/**
 * 물고기 분류 결과 데이터 클래스
 */
data class ClassificationResult(
    val scientificName: String,
    val indonesianName: String,
    val confidence: Float
) {
    /**
     * 신뢰도를 퍼센트 문자열로 반환
     */
    val confidencePercent: String
        get() = "${(confidence * 100).toInt()}%"

    /**
     * 신뢰도 레벨 (UI 표시용)
     */
    val confidenceLevel: ConfidenceLevel
        get() = when {
            confidence >= 0.8f -> ConfidenceLevel.HIGH
            confidence >= 0.6f -> ConfidenceLevel.MEDIUM
            else -> ConfidenceLevel.LOW
        }
}

/**
 * 신뢰도 레벨 enum
 */
enum class ConfidenceLevel {
    HIGH,    // >= 80%
    MEDIUM,  // >= 60%
    LOW      // < 60%
}

/**
 * 물고기 레이블 데이터
 */
data class FishLabel(
    val scientificName: String,
    val indonesianName: String
)
