# FishGo Model Documentation

## Model Overview

| 항목 | 값 |
|------|------|
| Base Model | EfficientNet-B0 (ImageNet pretrained) |
| Framework | PyTorch -> TFLite |
| Task | Image Classification |
| Export Format | TFLite FP16 |
| Model Size | **7.7 MB** |
| Input Size | 224 x 224 x 3 (RGB) |
| Output | 19 class probabilities |
| **Validation Accuracy** | **96.80%** |
| Best Epoch | 11 |

## Training Results (2024-12-24)

| Epoch | Train Acc | Val Acc | 비고 |
|-------|-----------|---------|------|
| 1 | ~80% | 92.07% | |
| 9 | 95.73% | 95.73% | |
| 10 | 95.73% | 95.73% | |
| **11** | 97.27% | **96.80%** | **Best Model** |
| 12 | - | 95.88% | ↓ |
| 13 | - | 94.97% | Overfitting 시작 |

> Epoch 13에서 중단 (Overfitting 방지)

---

## Dataset Preparation (핵심)

### 문제점: Roboflow 데이터셋의 클래스 불균형

Roboflow에서 다운로드한 데이터셋은 Train/Valid/Test 간에 클래스가 불균등하게 분포됨:

```
V2 Dataset:
- Train: 19개 클래스 ✅
- Valid: 10개 클래스만 ❌ (9개 누락)
- Test: 13개 클래스만 ❌

V4 Dataset:
- Train: 18개 클래스 (class 1 누락) ❌
- Valid: 13개 클래스만 ❌
- Test: 14개 클래스만 ❌
```

### 해결책: 80-10-10 균등 재분할

모든 버전(V1, V2, V4)의 모든 split을 합쳐서 **클래스별로 80/10/10 비율로 재분할**:

```bash
python scripts/prepare_classification_balanced.py \
    --input "/path/to/combined_dataset" \
    --output "/path/to/classification_data" \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

### 재분할 알고리즘

```python
def collect_all_crops(input_paths, class_names):
    """모든 버전, 모든 split에서 crop 수집"""
    crops_by_class = defaultdict(list)

    for version in ["v1", "v2", "v4"]:
        for split in ["train", "valid", "test"]:
            # YOLO bbox를 crop하여 수집
            for image in images:
                for bbox in parse_yolo_label(label_path):
                    cropped = image.crop(bbox)
                    crops_by_class[class_id].append(cropped)

    return crops_by_class

def split_and_save(crops_by_class, train_ratio=0.8, val_ratio=0.1):
    """클래스별로 80/10/10 분할"""
    for class_id, crops in crops_by_class.items():
        random.shuffle(crops)

        n_train = int(len(crops) * train_ratio)
        n_val = int(len(crops) * val_ratio)

        train_crops = crops[:n_train]
        val_crops = crops[n_train:n_train + n_val]
        test_crops = crops[n_train + n_val:]

        # 각 split에 저장
        save_to_folder(train_crops, f"train/{class_name}")
        save_to_folder(val_crops, f"val/{class_name}")
        save_to_folder(test_crops, f"test/{class_name}")
```

### 최종 데이터셋 통계

| Split | Images | Classes |
|-------|--------|---------|
| Train | ~12,471 | 19/19 ✅ |
| Val | ~1,559 | 19/19 ✅ |
| Test | ~1,559 | 19/19 ✅ |

---

## Training Pipeline

### 1. 데이터셋 준비 (균등 분할)

```bash
# 기존 방식 (문제 있음 - 원본 split 유지)
python scripts/prepare_classification_data.py

# 권장 방식 (균등 분할)
python scripts/prepare_classification_balanced.py \
    --input "/path/to/yolo_dataset" \
    --output "./classification_data" \
    --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

### 2. 모델 학습

```bash
python scripts/train_classifier.py \
    --data "./classification_data" \
    --output "./runs/classifier" \
    --epochs 50 \
    --batch-size 32 \
    --patience 5
```

**학습 설정:**
```python
model = EfficientNet-B0 (pretrained=ImageNet)
optimizer = Adam(lr=0.001)
scheduler = ReduceLROnPlateau(patience=5, factor=0.5)
criterion = CrossEntropyLoss()

# 중단 조건
early_stopping_patience = 5  # 5 epoch 무개선시 종료
target_accuracy = 0.95       # 95% 도달시 종료
```

**Data Augmentation (Train):**
```python
transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),  # ±15° (현실적)
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### 3. TFLite 변환

```bash
python scripts/convert_to_tflite.py \
    --model "./runs/classifier/best_model.pth" \
    --output "./app/src/main/assets/model/fish_classifier.tflite" \
    --classes "./runs/classifier/class_mapping.json" \
    --quantization fp16
```

---

## PyTorch to TFLite Conversion (상세)

### 변환 경로

```
PyTorch (.pth)
    ↓ torch.onnx.export()
ONNX (.onnx)
    ↓ onnx2tf
TensorFlow SavedModel
    ↓ tf.lite.TFLiteConverter
TFLite FP16 (.tflite)
```

### 1단계: PyTorch → ONNX

```python
import torch

# 모델 로드
model = create_efficientnet_b0(num_classes=19)
checkpoint = torch.load("best_model.pth", map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ONNX 내보내기
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "fish_classifier.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamo=False  # 레거시 방식 (호환성)
)
```

> **주의**: PyTorch 2.9+ 에서는 `dynamo=False` 옵션으로 레거시 방식 사용 권장

### 2단계: ONNX → TFLite (onnx2tf 사용)

**환경 설정 (Python 3.10 필수):**
```bash
# Python 3.14는 TensorFlow 미지원
python3.10 -m venv venv_tflite
source venv_tflite/bin/activate

pip install tensorflow==2.15.0 onnx==1.15.0 onnx2tf==1.21.0 \
    tf_keras onnx_graphsurgeon sng4onnx psutil flatbuffers
```

**변환 실행:**
```bash
onnx2tf -i fish_classifier.onnx \
    -o tflite_output \
    --output_signaturedefs \
    --copy_onnx_input_output_names_to_tflite
```

**결과:**
```
tflite_output/
├── fish_classifier_float16.tflite  (7.7 MB) ← 사용
├── fish_classifier_float32.tflite  (15.4 MB)
└── saved_model.pb
```

### 변환 방법 비교

| 방법 | 장점 | 단점 | 권장 |
|------|------|------|------|
| **onnx2tf** | 안정적, 호환성 좋음 | 패키지 의존성 많음 | ✅ 권장 |
| ai-edge-torch | 직접 변환 | macOS 미지원 (torch_xla 필요) | ❌ |
| onnx-tf | 공식 지원 | 버전 충돌 많음 | ❌ |

### 패키지 호환성 (중요!)

```
tensorflow==2.15.0
onnx==1.15.0        # 1.16+ 는 ml_dtypes 충돌
onnx2tf==1.21.0
tf_keras            # onnx2tf 의존성
onnx_graphsurgeon   # onnx2tf 의존성
```

> **주의**: `onnx 1.20+`와 `tensorflow 2.15`는 `ml_dtypes` 버전 충돌 발생

---

## Classes (19종)

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

---

## Model Input/Output

### Input
- Shape: `[1, 224, 224, 3]` (NHWC)
- Type: Float32
- Normalization: ImageNet (mean/std)

### Output
- Shape: `[1, 19]`
- Type: Float32
- Values: Class logits (apply softmax for probabilities)

---

## Android Integration

### 파일 위치
```
app/src/main/assets/model/
├── fish_classifier.tflite   (7.7 MB, FP16)
├── fish_detector.tflite     (5.9 MB, YOLOv8 - 미사용)
└── labels.txt               (19 class labels)
```

### labels.txt 형식
```
Alepes Djedaba -Ikan Selar Bulat-,Selar Bulat
Atropus Atropos -Ikan Cipa-Cipa-,Cipa-Cipa
...
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

---

## Confidence Levels

| Level | Range | Color | Action |
|-------|-------|-------|--------|
| HIGH | >= 0.7 | Green | 결과 신뢰 |
| MEDIUM | 0.4 ~ 0.7 | Orange | 주의 필요 |
| LOW | < 0.4 | Red | 경고 표시, 재촬영 권장 |

---

## Troubleshooting

### 1. PyTorch ReduceLROnPlateau 오류
```
TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'
```
**해결**: PyTorch 2.9.1에서 `verbose` 파라미터 제거됨. 삭제하면 해결.

### 2. TensorFlow Python 버전 미지원
```
ERROR: No matching distribution found for tensorflow
```
**해결**: Python 3.14는 미지원. Python 3.10 또는 3.11 사용.

### 3. ONNX ml_dtypes 충돌
```
AttributeError: module 'ml_dtypes' has no attribute 'float4_e2m1fn'
```
**해결**: `onnx==1.15.0` 사용 (1.16+ 버전은 충돌).

### 4. GPU Delegate 충돌
```
java.lang.UnsatisfiedLinkError: GPU delegate
```
**해결**: GPU delegate 제거, CPU 전용으로 실행.

---

## Future Improvements

1. **INT8 Quantization**: 더 작은 모델 크기 (3-4MB 예상)
2. **NNAPI Delegate**: Android NNAPI 가속기 지원
3. **More Classes**: 더 많은 어종 추가
4. **Data Augmentation**: Cutout, Mixup 등 추가
5. **Ensemble**: 여러 모델 앙상블로 정확도 향상

---

## Version History

| 버전 | 날짜 | 변경사항 |
|------|------|----------|
| 2.0 | 2024-12-24 | EfficientNet-B0 Classification (Val Acc: 96.80%) |
| 1.0 | 2024-12-23 | YOLOv8 Detection (mAP50: 59%) |
