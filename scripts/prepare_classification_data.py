#!/usr/bin/env python3
"""
FishGo Classification 데이터셋 준비 스크립트

기존 Roboflow Object Detection 데이터셋 (YOLO format)에서
Classification용 데이터셋 생성

- Detection 데이터의 각 BBox를 crop하여 개별 이미지로 저장
- 클래스별 폴더 구조로 정리
- Train/Val/Test 분할 유지

Usage:
    python prepare_classification_data.py \
        --input /path/to/yolo_dataset \
        --output /path/to/classification_dataset

Input 구조 (YOLO format):
    input/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── data.yaml

Output 구조 (ImageFolder format):
    output/
    ├── train/
    │   ├── Alepes_Djedaba/
    │   ├── Atropus_Atropos/
    │   └── ...
    └── val/
        └── ...
"""

import os
import sys
import argparse
from pathlib import Path
from PIL import Image
import yaml
from tqdm import tqdm


# 19종 클래스 매핑 (기존 labels.txt 기반)
CLASS_NAMES = [
    "Alepes Djedaba",
    "Atropus Atropos",
    "Caranx Ignobilis",
    "Chanos Chanos",
    "Decapterus Macarellus",
    "Euthynnus Affinis",
    "Katsuwonus Pelamis",
    "Lutjanus Malabaricus",
    "Parastromateus Niger",
    "Rastrelliger Kanagurta",
    "Rastrelliger sp",
    "Scaridae",
    "Scomber Japonicus",
    "Scomberomorus Guttatus",
    "Thunnus Alalunga",
    "Thunnus Obesus",
    "Thunnus Tonggol",
    "Tribus Sardini",
    "Upeneus Moluccensis"
]

# 인도네시아어 이름 매핑
INDONESIAN_NAMES = {
    "Alepes Djedaba": "Selar Bulat",
    "Atropus Atropos": "Cipa-Cipa",
    "Caranx Ignobilis": "Kuwe",
    "Chanos Chanos": "Bandeng",
    "Decapterus Macarellus": "Malalugis",
    "Euthynnus Affinis": "Tongkol",
    "Katsuwonus Pelamis": "Cakalang",
    "Lutjanus Malabaricus": "Kakap Merah",
    "Parastromateus Niger": "Bawal Hitam",
    "Rastrelliger Kanagurta": "Kembung",
    "Rastrelliger sp": "Kembung Banjar",
    "Scaridae": "Kakatua",
    "Scomber Japonicus": "Salem",
    "Scomberomorus Guttatus": "Tenggiri Papan",
    "Thunnus Alalunga": "Albacore Tuna",
    "Thunnus Obesus": "Tuna Mata Besar",
    "Thunnus Tonggol": "Tuna",
    "Tribus Sardini": "Kenyar",
    "Upeneus Moluccensis": "Kuniran"
}


def sanitize_class_name(name: str) -> str:
    """클래스 이름을 폴더명으로 사용 가능하도록 변환"""
    return name.replace(" ", "_").replace("-", "_")


def parse_yolo_label(label_path: Path, img_width: int, img_height: int) -> list:
    """
    YOLO format 라벨 파일 파싱

    YOLO format: class_id x_center y_center width height (normalized 0-1)

    Returns:
        list of (class_id, x1, y1, x2, y2) in pixel coordinates
    """
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height

            # Convert to corner coordinates
            x1 = int(max(0, x_center - width / 2))
            y1 = int(max(0, y_center - height / 2))
            x2 = int(min(img_width, x_center + width / 2))
            y2 = int(min(img_height, y_center + height / 2))

            # Validate box
            if x2 > x1 and y2 > y1:
                boxes.append((class_id, x1, y1, x2, y2))

    return boxes


def crop_and_save(
    image: Image.Image,
    box: tuple,
    class_name: str,
    output_dir: Path,
    image_name: str,
    box_idx: int,
    min_size: int = 32
) -> bool:
    """
    이미지에서 BBox 영역을 crop하여 저장

    Args:
        image: PIL Image
        box: (class_id, x1, y1, x2, y2)
        class_name: 클래스 이름
        output_dir: 출력 디렉토리
        image_name: 원본 이미지 이름
        box_idx: 같은 이미지 내 박스 인덱스
        min_size: 최소 크기 (너무 작은 박스 필터링)

    Returns:
        성공 여부
    """
    _, x1, y1, x2, y2 = box

    # 크기 검증
    width = x2 - x1
    height = y2 - y1
    if width < min_size or height < min_size:
        return False

    # Crop
    cropped = image.crop((x1, y1, x2, y2))

    # 출력 경로
    safe_class_name = sanitize_class_name(class_name)
    class_dir = output_dir / safe_class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    # 파일명: 원본이름_박스인덱스.jpg
    base_name = Path(image_name).stem
    output_path = class_dir / f"{base_name}_{box_idx}.jpg"

    # RGB로 변환 후 저장
    if cropped.mode != 'RGB':
        cropped = cropped.convert('RGB')
    cropped.save(output_path, 'JPEG', quality=95)

    return True


def process_split(
    input_dir: Path,
    output_dir: Path,
    class_names: list,
    split_name: str = "train"
) -> dict:
    """
    하나의 분할(train/val/test) 처리

    Args:
        input_dir: YOLO 데이터셋 분할 디렉토리
        output_dir: 출력 분할 디렉토리
        class_names: 클래스 이름 리스트 (인덱스로 매핑)
        split_name: 분할 이름 (로깅용)

    Returns:
        통계 딕셔너리
    """
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    if not images_dir.exists():
        print(f"Warning: {images_dir} does not exist")
        return {}

    # 통계
    stats = {name: 0 for name in class_names}
    total_boxes = 0
    skipped_boxes = 0

    # 이미지 파일 목록
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [f for f in images_dir.iterdir()
                   if f.suffix.lower() in image_extensions]

    print(f"\nProcessing {split_name}: {len(image_files)} images")

    for img_path in tqdm(image_files, desc=split_name):
        # 라벨 파일 경로
        label_path = labels_dir / f"{img_path.stem}.txt"

        # 이미지 로드
        try:
            image = Image.open(img_path)
            img_width, img_height = image.size
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        # 라벨 파싱
        boxes = parse_yolo_label(label_path, img_width, img_height)

        # 각 박스 처리
        for idx, box in enumerate(boxes):
            class_id = box[0]

            if class_id >= len(class_names):
                print(f"Warning: Unknown class_id {class_id} in {label_path}")
                skipped_boxes += 1
                continue

            class_name = class_names[class_id]

            success = crop_and_save(
                image=image,
                box=box,
                class_name=class_name,
                output_dir=output_dir,
                image_name=img_path.name,
                box_idx=idx
            )

            if success:
                stats[class_name] += 1
                total_boxes += 1
            else:
                skipped_boxes += 1

        image.close()

    print(f"  Total crops: {total_boxes}, Skipped: {skipped_boxes}")

    return stats


def load_class_names_from_yaml(data_yaml_path: Path) -> list:
    """data.yaml에서 클래스 이름 로드"""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'names' in data:
        if isinstance(data['names'], list):
            return data['names']
        elif isinstance(data['names'], dict):
            # {0: 'class0', 1: 'class1', ...} 형태
            max_idx = max(data['names'].keys())
            return [data['names'].get(i, f"class_{i}") for i in range(max_idx + 1)]

    return CLASS_NAMES  # 기본값


def save_labels_file(output_dir: Path, class_names: list):
    """labels.txt 파일 생성 (Android 앱용)"""
    labels_path = output_dir / "labels.txt"

    with open(labels_path, 'w') as f:
        for name in class_names:
            indo_name = INDONESIAN_NAMES.get(name, name)
            f.write(f"{name},{indo_name}\n")

    print(f"\nSaved labels to {labels_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO detection dataset to classification format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to YOLO format dataset (contains train/valid folders)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output classification dataset"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=32,
        help="Minimum crop size in pixels (default: 32)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 입력 검증
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    # 클래스 이름 로드
    data_yaml = input_path / "data.yaml"
    if data_yaml.exists():
        class_names = load_class_names_from_yaml(data_yaml)
        print(f"Loaded {len(class_names)} classes from data.yaml")
    else:
        class_names = CLASS_NAMES
        print(f"Using default {len(class_names)} classes")

    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)

    # 분할별 처리
    all_stats = {}

    # Train
    train_input = input_path / "train"
    if train_input.exists():
        train_output = output_path / "train"
        stats = process_split(train_input, train_output, class_names, "train")
        all_stats["train"] = stats

    # Valid -> val
    valid_input = input_path / "valid"
    if valid_input.exists():
        val_output = output_path / "val"
        stats = process_split(valid_input, val_output, class_names, "val")
        all_stats["val"] = stats

    # Test (있는 경우)
    test_input = input_path / "test"
    if test_input.exists():
        test_output = output_path / "test"
        stats = process_split(test_input, test_output, class_names, "test")
        all_stats["test"] = stats

    # Labels 파일 생성
    save_labels_file(output_path, class_names)

    # 최종 통계 출력
    print("\n" + "=" * 50)
    print("Classification Dataset Statistics")
    print("=" * 50)

    for split, stats in all_stats.items():
        print(f"\n{split.upper()}:")
        total = sum(stats.values())
        for class_name, count in sorted(stats.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / total * 100 if total > 0 else 0
                print(f"  {class_name}: {count} ({pct:.1f}%)")
        print(f"  Total: {total}")

    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
