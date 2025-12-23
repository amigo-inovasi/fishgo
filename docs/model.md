# FishGo Model Documentation

## Model Overview

| 항목 | 값 |
|------|------|
| Base Model | EfficientNet-B0 (ImageNet pretrained) |
| Framework | PyTorch -> TFLite |
| Task | Image Classification |
| Export Format | TFLite FP16 |
| Target Size | < 5 MB |
| Input Size | 224 x 224 x 3 (RGB) |
| Output | 19 class probabilities |
| Target Metrics | Top-1 Accuracy > 85%, Inference < 2s |

## Training Dataset

**Commercial Marine Fish Species v6** (Roboflow)
- 원본: YOLO Detection format
- 변환: Classification format (bbox crop)
- 19종 인도네시아 상업용 해양 어류

### Data Augmentation (학습 시)

```python
# 권장 설정 (Roboflow 기준)
- RandomCrop(224)
- RandomHorizontalFlip (50%)
- RandomRotation(15)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
```

### Classes (19종)

| ID | Scientific Name | Indonesian Name | 한국어 |
|----|-----------------|-----------------|--------|
| 0 | Alepes Djedaba | Selar Bulat | 전갱이류 |
| 1 | Atropus Atropos | Cipa-Cipa | 시파시파 |
| 2 | Caranx Ignobilis | Kuwe | 쿠웨 |
| 3 | Chanos Chanos | Bandeng | 밀크피시 |
| 4 | Decapterus Macarellus | Malalugis | 말라루기스 |
| 5 | Euthynnus Affinis | Tongkol | 가다랑어류 |
| 6 | Katsuwonus Pelamis | Cakalang | 가다랑어 |
| 7 | Lutjanus Malabaricus | Kakap Merah | 적도미 |
| 8 | Parastromateus Niger | Bawal Hitam | 검은병어 |
| 9 | Rastrelliger Kanagurta | Kembung | 고등어류 |
| 10 | Rastrelliger sp | Kembung Banjar | 반자르 고등어 |
| 11 | Scaridae | Kakatua | 앵무고기 |
| 12 | Scomber Japonicus | Salem | 고등어 |
| 13 | Scomberomorus Guttatus | Tenggiri Papan | 삼치류 |
| 14 | Thunnus Alalunga | Albacore Tuna | 날개다랑어 |
| 15 | Thunnus Obesus | Tuna Mata Besar | 눈다랑어 |
| 16 | Thunnus Tonggol | Tuna | 참다랑어 |
| 17 | Tribus Sardini | Kenyar | 정어리류 |
| 18 | Upeneus Moluccensis | Kuniran | 쿠니란 |

## Training Pipeline

### 1. 데이터셋 준비

YOLO Detection bbox를 crop하여 Classification 포맷으로 변환:

```bash
python scripts/prepare_classification_data.py
```

```
# 입력 구조 (YOLO Detection)
dataset/
├── train/
│   ├── images/
│   └── labels/    # YOLO format (class x y w h)
└── valid/
    ├── images/
    └── labels/

# 출력 구조 (Classification)
classification_data/
├── train/
│   ├── Alepes Djedaba/
│   ├── Atropus Atropos/
│   └── ...
└── valid/
    ├── Alepes Djedaba/
    └── ...
```

### 2. 모델 학습

```bash
python scripts/train_classifier.py
```

**학습 설정:**
```python
model = EfficientNet-B0 (pretrained=ImageNet)
optimizer = Adam(lr=1e-4)
scheduler = ReduceLROnPlateau(patience=3)
epochs = 50 (early stopping patience=10)
batch_size = 32
```

**데이터 전처리:**
```python
# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
```

### 3. TFLite 변환

```bash
python scripts/convert_to_tflite.py
```

**변환 경로:**
```
PyTorch (.pth) -> ONNX (.onnx) -> TFLite FP16 (.tflite)
```

**변환 라이브러리 (우선순위):**
1. ai-edge-torch (권장)
2. onnx2tf
3. onnx-tf

## Model Input/Output

### Input
- Shape: `[1, 224, 224, 3]`
- Type: Float32
- Normalization: ImageNet (mean/std)

### Output
- Shape: `[1, 19]`
- Type: Float32
- Values: Class probabilities (softmax)

## Android Integration

### 위치
```
app/src/main/assets/model/
├── fish_classifier.tflite   # < 5 MB (FP16)
└── labels.txt               # 19 class labels
```

### Usage
```kotlin
val classifier = FishClassifier(context)
val result = classifier.classify(bitmap)
// result: ClassificationResult(indonesianName, scientificName, confidence)
classifier.close()
```

### Preprocessing (Android)
```kotlin
// 1. 가이드 영역 crop (3:1 비율)
// 2. 정사각형으로 패딩 (검정색)
// 3. 224x224로 리사이즈
// 4. ImageNet 정규화
val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
val std = floatArrayOf(0.229f, 0.224f, 0.225f)
val normalized = (pixel / 255.0f - mean) / std
```

## Confidence Levels

| Level | Range | Color | Action |
|-------|-------|-------|--------|
| HIGH | >= 0.7 | Green | 결과 신뢰 |
| MEDIUM | 0.4 ~ 0.7 | Orange | 주의 필요 |
| LOW | < 0.4 | Red | 경고 표시, 재촬영 권장 |

## Known Issues

### GPU Delegate 제거됨
TensorFlow Lite GPU delegate 버전 충돌로 CPU 전용으로 변경.

### Classification vs Detection
Detection 방식에서 Classification으로 전환한 이유:
- **정확도 향상**: 단일 객체 분류에 집중
- **단순화**: bbox 후처리 불필요
- **안정성**: 가이드 프레임으로 일관된 입력

## Future Improvements

1. **INT8 Quantization**: 더 작은 모델 크기 (1-2MB 예상)
2. **NNAPI Delegate**: Android NNAPI 가속기 지원
3. **More Classes**: 더 많은 어종 추가
4. **Data Augmentation**: Cutout, Mixup 등 추가
5. **Ensemble**: 여러 모델 앙상블로 정확도 향상

## Previous Version (YOLOv8 Detection)

이전에 YOLOv8 Detection 방식을 사용했으나 Classification으로 전환됨.
이전 버전 정보는 git history 참조.

### 이전 모델 스펙
- Model: YOLOv8n (nano)
- Input: 640x640
- Output: [1, 23, 8400] (4 box + 19 class)
- File Size: 6.2MB
- mAP50: 59%
