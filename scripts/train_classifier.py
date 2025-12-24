#!/usr/bin/env python3
"""
FishGo EfficientNet-B0 Classification 모델 학습 스크립트

정확도 우선 Classification 모델 학습
- Base: EfficientNet-B0 (MobileNetV3보다 정확)
- Input: 224x224 (정사각형으로 패딩)
- 50+ epochs, early stopping

Usage:
    python train_classifier.py \
        --data /path/to/classification_dataset \
        --output ./runs/classifier \
        --epochs 50 \
        --batch-size 32

Requirements:
    pip install torch torchvision tqdm pillow
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from tqdm import tqdm


# 디바이스 설정
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# EfficientNet-B0 모델 생성
def create_model(num_classes: int = 19, pretrained: bool = True):
    """
    EfficientNet-B0 모델 생성

    정확도 우선이므로 EfficientNet-B0 사용 (MobileNet보다 정확)
    모델 크기: ~5MB (FP16 양자화 후)

    Args:
        num_classes: 출력 클래스 수
        pretrained: ImageNet pretrained weights 사용 여부

    Returns:
        EfficientNet-B0 모델
    """
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)

    # Classifier 수정 (1000 -> num_classes)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


# 강화된 Data Augmentation (도메인 일반화 향상)
def get_transforms(strong_augment: bool = True):
    """
    학습/검증 transforms 반환

    강화된 augmentation (도메인 갭 해소):
    - 다양한 조명/색상 변화 (다른 카메라 시뮬레이션)
    - GaussianBlur (카메라 품질 차이)
    - RandomGrayscale (색상 의존도 감소)
    - RandomErasing (부분 가림 대응)
    - RandomAffine (다양한 각도/스케일)
    """
    if strong_augment:
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            # 기하학적 변환
            transforms.RandomAffine(
                degrees=20,           # ±20° 회전
                translate=(0.1, 0.1), # 10% 이동
                scale=(0.9, 1.1),     # 90~110% 스케일
                shear=10              # ±10° 전단
            ),
            # 색상 변환 (강화)
            transforms.ColorJitter(
                brightness=0.4,       # 더 강한 밝기 변화
                contrast=0.4,
                saturation=0.3,
                hue=0.15
            ),
            # 블러 (다양한 카메라 품질 시뮬레이션)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.3),
            # 그레이스케일 (색상에만 의존하지 않도록)
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            # Random Erasing (부분 가림 대응)
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
        ])
    else:
        # 기본 augmentation (이전 버전)
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
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

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transforms, val_transforms


class EarlyStopping:
    """Early Stopping 구현"""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> tuple:
    """한 epoch 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Progress bar 업데이트
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple:
    """검증"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    patience: int = 10,
    num_workers: int = 4
):
    """
    모델 학습 메인 함수

    Args:
        data_dir: Classification 데이터셋 경로
        output_dir: 출력 디렉토리
        epochs: 최대 epoch 수
        batch_size: 배치 크기
        learning_rate: 학습률
        patience: Early stopping patience
        num_workers: DataLoader workers

    학습 전략:
    - Adam optimizer
    - ReduceLROnPlateau scheduler
    - Early stopping (patience=10)
    - Best model 저장
    """
    # 경로 설정
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 디바이스
    device = get_device()
    print(f"Using device: {device}")

    # Transforms
    train_transforms, val_transforms = get_transforms()

    # 데이터셋 로드
    train_dataset = datasets.ImageFolder(
        data_path / "train",
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        data_path / "val",
        transform=val_transforms
    )

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # 모델
    model = create_model(num_classes=num_classes)
    model = model.to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # Early Stopping
    early_stopping = EarlyStopping(patience=patience)

    # 학습 기록
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print("\n" + "=" * 50)
    print("Starting Training")
    print("=" * 50)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 30)

        # 학습
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # 검증
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 기록
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        # Best model 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"★ New best model! Val Acc: {val_acc:.4f}")

            # Best model 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'classes': train_dataset.classes
            }, output_path / "best_model.pth")

        # Target Accuracy 도달 체크 (95% 이상이면 종료)
        if val_acc >= 0.95:
            print(f"\n🎯 Target accuracy reached! Val Acc: {val_acc:.4f} >= 95%")
            break

        # Early Stopping 체크
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # 최종 모델 로드
    model.load_state_dict(best_model_wts)

    # 최종 모델 저장 (추론용)
    torch.save(model.state_dict(), output_path / "fish_classifier.pth")

    # History 저장
    with open(output_path / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # 클래스 매핑 저장
    class_mapping = {
        'idx_to_class': {i: c for i, c in enumerate(train_dataset.classes)},
        'class_to_idx': train_dataset.class_to_idx
    }
    with open(output_path / "class_mapping.json", 'w') as f:
        json.dump(class_mapping, f, indent=2)

    print("\n" + "=" * 50)
    print("Training Complete!")
    print("=" * 50)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Models saved to: {output_path}")

    return model, history


def main():
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 fish classifier"
    )
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to classification dataset (ImageFolder format)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./runs/classifier",
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Maximum number of epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers (default: 4)"
    )

    args = parser.parse_args()

    # 출력 디렉토리에 타임스탬프 추가
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp

    train(
        data_dir=args.data,
        output_dir=str(output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
