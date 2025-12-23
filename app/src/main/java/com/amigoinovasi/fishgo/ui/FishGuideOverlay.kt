package com.amigoinovasi.fishgo.ui

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import com.amigoinovasi.fishgo.R

/**
 * 물고기 촬영 가이드 오버레이
 * - 정사각형 프레임 (1:1 비율) - 모델 입력과 일치
 * - 모서리가 둥근 테두리
 * - 프레임 외부는 반투명 어둡게
 * - 가이드 텍스트 표시
 */
class FishGuideOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    // 가이드 프레임 비율
    private val aspectRatio = 1f  // 가로:세로 = 1:1 (정사각형)
    private val widthPercent = 0.85f
    private val cornerRadius = 24f

    // 가이드 영역 (외부에서 참조 가능)
    val guideRect = RectF()

    // 프레임 테두리 페인트 (점선)
    private val framePaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 4f
        pathEffect = DashPathEffect(floatArrayOf(20f, 10f), 0f)
        isAntiAlias = true
    }

    // 어두운 영역 페인트
    private val dimPaint = Paint().apply {
        color = Color.parseColor("#80000000")  // 반투명 검정
        style = Paint.Style.FILL
    }

    // 투명 영역 페인트 (구멍 뚫기용)
    private val clearPaint = Paint().apply {
        color = Color.TRANSPARENT
        style = Paint.Style.FILL
        xfermode = PorterDuffXfermode(PorterDuff.Mode.CLEAR)
    }

    // 텍스트 페인트
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 42f
        textAlign = Paint.Align.CENTER
        isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
    }

    // 작은 텍스트 페인트
    private val smallTextPaint = Paint().apply {
        color = Color.parseColor("#CCFFFFFF")  // 약간 투명한 흰색
        textSize = 36f
        textAlign = Paint.Align.CENTER
        isAntiAlias = true
    }

    // 물고기 아이콘 페인트
    private val iconPaint = Paint().apply {
        textSize = 64f
        textAlign = Paint.Align.CENTER
        isAntiAlias = true
    }

    // 코너 가이드 페인트 (L자 모양)
    private val cornerPaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 6f
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }

    private val cornerLength = 40f

    init {
        // 하드웨어 가속 비활성화 (PorterDuff.Mode.CLEAR 사용 위해)
        setLayerType(LAYER_TYPE_SOFTWARE, null)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        // 가이드 영역 계산
        val guideWidth = width * widthPercent
        val guideHeight = guideWidth / aspectRatio
        val left = (width - guideWidth) / 2
        val top = (height - guideHeight) / 2
        guideRect.set(left, top, left + guideWidth, top + guideHeight)

        // 1. 전체를 어둡게
        canvas.drawRect(0f, 0f, width.toFloat(), height.toFloat(), dimPaint)

        // 2. 가이드 영역을 투명하게 (구멍 뚫기)
        canvas.drawRoundRect(guideRect, cornerRadius, cornerRadius, clearPaint)

        // 3. 가이드 프레임 테두리 (점선)
        canvas.drawRoundRect(guideRect, cornerRadius, cornerRadius, framePaint)

        // 4. 코너 가이드 (L자 모양) - 더 눈에 띄게
        drawCornerGuides(canvas)

        // 5. 상단 아이콘 + 텍스트
        val fishIcon = "🐟"
        canvas.drawText(fishIcon, width / 2f, guideRect.top - 80f, iconPaint)

        // 6. 하단 가이드 텍스트 (인도네시아어)
        val guideText = "Letakkan ikan di dalam kotak"  // "물고기를 박스 안에 놓으세요"
        canvas.drawText(guideText, width / 2f, guideRect.bottom + 70f, textPaint)

        // 7. 추가 힌트 텍스트
        val hintText = "Pastikan ikan terlihat jelas"  // "물고기가 선명하게 보이도록 하세요"
        canvas.drawText(hintText, width / 2f, guideRect.bottom + 120f, smallTextPaint)
    }

    /**
     * 코너에 L자 가이드 그리기
     */
    private fun drawCornerGuides(canvas: Canvas) {
        val rect = guideRect
        val r = cornerRadius
        val len = cornerLength

        // 좌상단
        canvas.drawLine(rect.left + r, rect.top, rect.left + r + len, rect.top, cornerPaint)
        canvas.drawLine(rect.left, rect.top + r, rect.left, rect.top + r + len, cornerPaint)

        // 우상단
        canvas.drawLine(rect.right - r - len, rect.top, rect.right - r, rect.top, cornerPaint)
        canvas.drawLine(rect.right, rect.top + r, rect.right, rect.top + r + len, cornerPaint)

        // 좌하단
        canvas.drawLine(rect.left + r, rect.bottom, rect.left + r + len, rect.bottom, cornerPaint)
        canvas.drawLine(rect.left, rect.bottom - r - len, rect.left, rect.bottom - r, cornerPaint)

        // 우하단
        canvas.drawLine(rect.right - r - len, rect.bottom, rect.right - r, rect.bottom, cornerPaint)
        canvas.drawLine(rect.right, rect.bottom - r - len, rect.right, rect.bottom - r, cornerPaint)
    }

    /**
     * 가이드 영역의 비율 정보 반환
     * MainActivity에서 이미지 crop 시 사용
     */
    fun getGuideRatios(): GuideRatios {
        return GuideRatios(
            widthPercent = widthPercent,
            aspectRatio = aspectRatio
        )
    }

    data class GuideRatios(
        val widthPercent: Float,
        val aspectRatio: Float
    )
}
