# FishGo Development Log

## 2025-12-23 (Classification 전환)

### Detection -> Classification 아키텍처 전환

**배경**
- YOLOv8 Detection 방식의 한계: 실시간 처리 부담, Task Vision 호환성 문제
- 사용자 요구사항: 정확도 우선, 1-2초 추론 허용 가능

**전환 이유**
1. **정확도 향상**: 단일 객체 분류에 집중
2. **사용자 경험**: 가이드 프레임으로 일관된 촬영 유도
3. **안정성**: 복잡한 bbox NMS 후처리 제거
4. **유지보수**: 더 단순한 코드베이스

---

### Phase 1: 모델 학습 파이프라인 작성

**스크립트 생성**:
- `scripts/prepare_classification_data.py` - YOLO bbox crop -> Classification format
- `scripts/train_classifier.py` - EfficientNet-B0 학습
- `scripts/convert_to_tflite.py` - PyTorch -> ONNX -> TFLite FP16

**모델 스펙**:
- Base: EfficientNet-B0 (ImageNet pretrained)
- Input: 224x224x3 RGB
- Output: 19 class probabilities
- Target: <5MB, Top-1 Accuracy >85%

---

### Phase 2: Android 앱 리팩토링

**새로운 파일**:
| 파일 | 설명 |
|------|------|
| `FishGuideOverlay.kt` | 가이드 프레임 오버레이 (3:1 비율, 대시 테두리) |
| `FishClassifier.kt` | EfficientNet TFLite 분류기 |
| `ClassificationResult.kt` | 분류 결과 데이터 클래스 |
| `ResultActivity.kt` | 결과 화면 (인도네시아어 이름, 신뢰도) |
| `activity_result.xml` | 결과 화면 레이아웃 |
| `bg_guide_text.xml` | 가이드 텍스트 배경 drawable |

**수정된 파일**:
| 파일 | 변경 내용 |
|------|----------|
| `MainActivity.kt` | Classification 워크플로우 (crop, pad, navigate) |
| `activity_main.xml` | FishGuideOverlay + MaterialButton |
| `AndroidManifest.xml` | ResultActivity 등록 |
| `strings.xml` | 인도네시아어 UI 문자열 |
| `colors.xml` | 신뢰도 색상 추가 |

**앱 플로우**:
```
MainActivity (카메라 + 가이드)
    ↓ 촬영 버튼
가이드 영역 crop
    ↓
정사각형 패딩 (224x224)
    ↓
임시 파일 저장
    ↓
ResultActivity (분류 결과)
```

**UI 변경**:
- Detection bounding box -> Guide frame overlay
- 실시간 인식 -> 촬영 후 분석
- FloatingActionButton -> MaterialButton with text

---

### Phase 3: 문서 업데이트

- `docs/README.md` - Classification 기반으로 전면 개편
- `docs/model.md` - EfficientNet-B0 스펙으로 업데이트
- `docs/tasks20251223.md` - 전체 태스크 명세 (참조용)

---

### 남은 작업

**Phase 1 실행 필요**:
- [ ] 데이터셋 준비 스크립트 실행
- [ ] 모델 학습
- [ ] TFLite 변환
- [ ] 앱에 모델 배치

**테스트**:
- [ ] GitHub Actions 빌드 확인
- [ ] 실제 기기 테스트
- [ ] 분류 정확도 검증

---

## 2025-12-23 (이전: Detection 버전)

### Initial Development

#### 1. 모델 학습 및 변환

**YOLOv8 학습**
- Base model: yolov8n.pt (nano)
- Dataset: Commercial Marine Fish Species (Kaggle)
- Device: Apple Silicon (MPS)
- Epochs: 6 (early stopped)
- Results: mAP50=59%, mAP50-95=45%

**TFLite 변환 이슈**
- Python 3.14: TensorFlow 미지원 -> Python 3.10으로 변경
- TensorFlow 네트워크 이슈: 여러 번 재시도 후 성공
- NumPy 버전 충돌: TensorFlow 2.19.1로 해결

**변환 결과**
- Float16 모델: 6.2MB
- 출력 형식: `[1, 23, 8400]`

---

#### 2. Android 앱 구현

**초기 문제: TFLite Task Vision 호환성**
```
Model load failed: Error getting native address of native library: task_vision_jni
```

**원인**: YOLOv8 TFLite 모델은 TFLite Task Vision의 ObjectDetector와 호환되지 않음.

**해결**: Core TFLite Interpreter 직접 사용
- `FishDetector.kt` 재작성
- YOLOv8 출력 형식 파싱 구현
- NMS (Non-Maximum Suppression) 구현

---

#### 3. GPU Delegate 버전 충돌

**빌드 에러**
```
Cannot access class 'org.tensorflow.lite.gpu.GpuDelegateFactory.Options'
```

**원인**: tensorflow-lite와 tensorflow-lite-gpu 버전 간 API 불일치

**해결**: GPU delegate 제거, CPU 전용으로 변경
- `tensorflow-lite-gpu` 의존성 제거
- `GpuDelegate` 관련 코드 제거

---

#### 4. UI 변경

**요청사항**
- 실시간 인식 -> 캡쳐 버튼 클릭 시 인식
- 앱 아이콘: fishgo02.png 적용

**구현**
- `FloatingActionButton` 추가 (캡쳐 버튼)
- `ImageCapture` 대신 `previewView.bitmap` 사용 (더 빠름)
- 로딩 인디케이터 추가
- 앱 아이콘: fishgo02.png에서 각 density별 생성

**색상 변경**
- Primary color: `#1976D2` (파랑) -> `#B71C1C` (빨강)

---

#### 5. GitHub Actions CI/CD

**초기 문제: gradle-wrapper.jar 손상**
- 파일이 HTML로 저장됨 (GitHub 404 페이지)
- Gradle 8.4 공식 릴리스에서 재다운로드

**최종 워크플로우**
- Push to main -> 자동 빌드
- Artifacts: fishgo-debug, fishgo-release-unsigned
- 빌드 시간: ~3-4분

---

### 파일 변경 이력

| 파일 | 변경 내용 |
|------|----------|
| `FishDetector.kt` | YOLOv8 TFLite Interpreter 구현 |
| `MainActivity.kt` | 캡쳐 버튼 방식으로 변경 |
| `DetectionOverlayView.kt` | `clearResults()` 메서드 추가 |
| `activity_main.xml` | FloatingActionButton, ProgressBar 추가 |
| `colors.xml` | Primary color 빨간색으로 변경 |
| `strings.xml` | 새 문자열 추가 |
| `labels.txt` | metadata.yaml 형식과 일치하도록 수정 |
| `libs.versions.toml` | TFLite 의존성 변경 |
| `build.gradle.kts` | GPU 의존성 제거 |

---

### Commits

1. `feat: FishGo - AI Fish Species Detection Android App`
   - 초기 프로젝트 구조

2. `fix: replace corrupted gradle-wrapper.jar with valid JAR`
   - Gradle wrapper 수정

3. `fix: YOLOv8 TFLite compatibility + capture button UI`
   - TFLite Interpreter 전환
   - 캡쳐 버튼 UI

4. `fix: remove GPU delegate to resolve version conflict`
   - GPU 의존성 제거

---

### 테스트 결과

**빌드**: 성공 (GitHub Actions)
**앱 실행**: 카메라 프리뷰 정상
**모델 로딩**: 성공 (에러 없음)
**물고기 인식**: 테스트 필요 (실제 물고기 이미지)

---

## TODO

- [ ] 실제 물고기 이미지로 인식 테스트
- [ ] 인식 속도 측정 및 최적화
- [ ] 갤러리에서 이미지 선택 기능 추가
- [ ] 인식 결과 저장/공유 기능
- [ ] Navigo 앱 통합 준비
