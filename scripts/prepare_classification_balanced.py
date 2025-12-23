#!/usr/bin/env python3
"""
FishGo Classification 데이터셋 준비 스크립트 (클래스별 균등 분할)

모든 split(train/valid/test)의 이미지를 합쳐서
클래스별로 80/10/10 비율로 재분할

Usage:
    python prepare_classification_balanced.py \
        --input "/path/to/yolo_dataset" \
        --output "/path/to/classification_dataset" \
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""

import os
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict
from PIL import Image
import yaml
from tqdm import tqdm


def sanitize_class_name(name: str) -> str:
    """클래스 이름을 폴더명으로 사용 가능하도록 변환"""
    # 특수문자 제거, 공백을 언더스코어로
    return name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


def parse_yolo_label(label_path: Path, img_width: int, img_height: int) -> list:
    """YOLO format 라벨 파일 파싱"""
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

            x1 = int(max(0, x_center - width / 2))
            y1 = int(max(0, y_center - height / 2))
            x2 = int(min(img_width, x_center + width / 2))
            y2 = int(min(img_height, y_center + height / 2))

            if x2 > x1 and y2 > y1:
                boxes.append((class_id, x1, y1, x2, y2))

    return boxes


def collect_all_crops(input_path: Path, class_names: list, min_size: int = 32) -> dict:
    """
    모든 split에서 crop된 이미지 데이터 수집

    Returns:
        {class_id: [(cropped_image, source_info), ...]}
    """
    crops_by_class = defaultdict(list)

    for split in ["train", "valid", "test"]:
        split_dir = input_path / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():
            print(f"  Skipping {split}: images folder not found")
            continue

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in images_dir.iterdir()
                       if f.suffix.lower() in image_extensions]

        print(f"  Processing {split}: {len(image_files)} images")

        for img_path in tqdm(image_files, desc=f"  {split}"):
            label_path = labels_dir / f"{img_path.stem}.txt"

            try:
                image = Image.open(img_path)
                img_width, img_height = image.size
            except Exception as e:
                continue

            boxes = parse_yolo_label(label_path, img_width, img_height)

            for idx, box in enumerate(boxes):
                class_id, x1, y1, x2, y2 = box

                # 크기 검증
                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    continue

                if class_id >= len(class_names):
                    continue

                # Crop
                cropped = image.crop((x1, y1, x2, y2))
                if cropped.mode != 'RGB':
                    cropped = cropped.convert('RGB')

                # 저장할 정보
                source_info = f"{split}_{img_path.stem}_{idx}"
                crops_by_class[class_id].append((cropped, source_info))

            image.close()

    return crops_by_class


def split_and_save(
    crops_by_class: dict,
    class_names: list,
    output_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """클래스별로 분할하여 저장"""
    random.seed(seed)

    stats = {"train": defaultdict(int), "val": defaultdict(int), "test": defaultdict(int)}

    for class_id, crops in crops_by_class.items():
        class_name = class_names[class_id]
        safe_name = sanitize_class_name(class_name)

        # 셔플
        random.shuffle(crops)

        n_total = len(crops)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        # 나머지는 test

        train_crops = crops[:n_train]
        val_crops = crops[n_train:n_train + n_val]
        test_crops = crops[n_train + n_val:]

        # 저장
        for split_name, split_crops in [("train", train_crops), ("val", val_crops), ("test", test_crops)]:
            split_dir = output_path / split_name / safe_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for cropped, source_info in split_crops:
                output_file = split_dir / f"{source_info}.jpg"
                cropped.save(output_file, 'JPEG', quality=95)
                stats[split_name][class_id] += 1

    return stats


def load_class_names_from_yaml(data_yaml_path: Path) -> list:
    """data.yaml에서 클래스 이름 로드"""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'names' in data:
        if isinstance(data['names'], list):
            return data['names']
        elif isinstance(data['names'], dict):
            max_idx = max(data['names'].keys())
            return [data['names'].get(i, f"class_{i}") for i in range(max_idx + 1)]

    return []


def save_labels_file(output_path: Path, class_names: list):
    """labels.txt 파일 생성"""
    # 인도네시아어 이름 매핑
    indonesian_names = {
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

    labels_path = output_path / "labels.txt"
    with open(labels_path, 'w') as f:
        for name in class_names:
            # 클래스명에서 인도네시아어 부분 추출 시도
            base_name = name.split(" -")[0] if " -" in name else name
            indo_name = indonesian_names.get(base_name, name)
            f.write(f"{name},{indo_name}\n")

    print(f"\nSaved labels to {labels_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare balanced classification dataset from YOLO format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to YOLO format dataset"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output classification dataset"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=32,
        help="Minimum crop size (default: 32)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 비율 검증
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"Warning: Ratios sum to {total_ratio}, not 1.0")

    print("=" * 60)
    print("Balanced Classification Dataset Preparation")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Split ratio: {args.train_ratio}/{args.val_ratio}/{args.test_ratio}")
    print()

    # 클래스 이름 로드
    data_yaml = input_path / "data.yaml"
    if data_yaml.exists():
        class_names = load_class_names_from_yaml(data_yaml)
        print(f"Loaded {len(class_names)} classes from data.yaml")
    else:
        print("Error: data.yaml not found")
        sys.exit(1)

    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: 모든 crop 수집
    print("\n[Step 1] Collecting all crops from all splits...")
    crops_by_class = collect_all_crops(input_path, class_names, args.min_size)

    total_crops = sum(len(v) for v in crops_by_class.values())
    print(f"\nTotal crops collected: {total_crops}")
    print(f"Classes with data: {len(crops_by_class)}")

    # Step 2: 분할 및 저장
    print("\n[Step 2] Splitting and saving...")
    stats = split_and_save(
        crops_by_class,
        class_names,
        output_path,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed
    )

    # Step 3: Labels 파일 생성
    save_labels_file(output_path, class_names)

    # 최종 통계
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)

    for split_name in ["train", "val", "test"]:
        split_stats = stats[split_name]
        total = sum(split_stats.values())
        n_classes = len([v for v in split_stats.values() if v > 0])
        print(f"\n{split_name.upper()}: {total} images, {n_classes} classes")
        for class_id in sorted(split_stats.keys()):
            count = split_stats[class_id]
            if count > 0:
                class_name = class_names[class_id].split(" -")[0]  # 짧게 표시
                print(f"  {class_id:2d}. {class_name:<25}: {count:4d}")

    # 검증
    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    all_good = True
    for split_name in ["train", "val", "test"]:
        n_classes = len([v for v in stats[split_name].values() if v > 0])
        status = "✅" if n_classes == 19 else "❌"
        print(f"{split_name}: {n_classes}/19 classes {status}")
        if n_classes != 19:
            all_good = False

    if all_good:
        print("\n✅ All 19 classes present in all splits!")
    else:
        print("\n❌ Some classes are missing!")

    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
