#!/usr/bin/env python3
"""
FishGo Classification 데이터셋 준비 스크립트 (V1 + V2 + V4 통합)

모든 버전의 모든 split(train/valid/test)을 합쳐서
클래스별로 80/10/10 비율로 재분할

Usage:
    python prepare_combined_dataset.py --output ./classification_data_combined
"""

import os
import sys
import argparse
import random
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image
import yaml
from tqdm import tqdm


def sanitize_class_name(name: str) -> str:
    """클래스 이름을 폴더명으로 사용 가능하도록 변환"""
    return name.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")


def get_image_hash(image: Image.Image) -> str:
    """이미지의 해시값 계산 (중복 제거용)"""
    return hashlib.md5(image.tobytes()).hexdigest()


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


def collect_crops_from_version(
    input_path: Path,
    class_names: list,
    seen_hashes: set,
    min_size: int = 32
) -> dict:
    """한 버전에서 crop된 이미지 수집 (중복 제거)"""
    crops_by_class = defaultdict(list)
    duplicates = 0

    for split in ["train", "valid", "test"]:
        split_dir = input_path / split
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():
            continue

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in images_dir.iterdir()
                       if f.suffix.lower() in image_extensions]

        for img_path in tqdm(image_files, desc=f"  {input_path.name}/{split}", leave=False):
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

                # 중복 체크
                img_hash = get_image_hash(cropped)
                if img_hash in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes.add(img_hash)

                # 저장할 정보
                source_info = f"{input_path.name}_{split}_{img_path.stem}_{idx}"
                crops_by_class[class_id].append((cropped, source_info))

            image.close()

    return crops_by_class, duplicates


def merge_crops(all_crops: list) -> dict:
    """여러 버전의 crops를 병합"""
    merged = defaultdict(list)
    for crops_dict in all_crops:
        for class_id, crops in crops_dict.items():
            merged[class_id].extend(crops)
    return merged


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

    for class_id, crops in tqdm(crops_by_class.items(), desc="Saving"):
        if class_id >= len(class_names):
            continue

        class_name = class_names[class_id]
        safe_name = sanitize_class_name(class_name)

        # 셔플
        random.shuffle(crops)

        n_total = len(crops)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

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
            base_name = name.split(" -")[0] if " -" in name else name
            indo_name = indonesian_names.get(base_name, name)
            f.write(f"{name},{indo_name}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Combine V1, V2, V4 datasets and prepare balanced classification dataset"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./classification_data_combined",
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

    # 데이터셋 경로
    base_path = Path(__file__).parent.parent
    dataset_versions = [
        base_path / "Commercial Marine Fish Species (v1)",
        base_path / "Commercial Marine Fish Species (v2)",
        base_path / "Commercial Marine Fish Species (v4)",
    ]

    output_path = Path(args.output)
    test_ratio = 1.0 - args.train_ratio - args.val_ratio

    print("=" * 60)
    print("Combined Dataset Preparation (V1 + V2 + V4)")
    print("=" * 60)
    print(f"Output: {output_path}")
    print(f"Split ratio: {args.train_ratio}/{args.val_ratio}/{test_ratio:.1f}")
    print()

    # 클래스 이름 로드 (V1 기준)
    data_yaml = dataset_versions[0] / "data.yaml"
    if data_yaml.exists():
        class_names = load_class_names_from_yaml(data_yaml)
        print(f"Loaded {len(class_names)} classes")
    else:
        print("Error: data.yaml not found")
        sys.exit(1)

    # 출력 디렉토리 생성
    output_path.mkdir(parents=True, exist_ok=True)

    # 모든 버전에서 crop 수집 (중복 제거)
    print("\n[Step 1] Collecting crops from all versions...")
    all_crops = []
    seen_hashes = set()
    total_duplicates = 0

    for version_path in dataset_versions:
        if not version_path.exists():
            print(f"  Skipping {version_path.name}: not found")
            continue

        print(f"\n  Processing {version_path.name}...")
        crops, duplicates = collect_crops_from_version(
            version_path, class_names, seen_hashes, args.min_size
        )
        all_crops.append(crops)
        total_duplicates += duplicates

        version_total = sum(len(v) for v in crops.values())
        print(f"    -> {version_total} unique crops (skipped {duplicates} duplicates)")

    # 병합
    print("\n[Step 2] Merging all crops...")
    merged_crops = merge_crops(all_crops)
    total_crops = sum(len(v) for v in merged_crops.values())
    print(f"Total unique crops: {total_crops}")
    print(f"Total duplicates removed: {total_duplicates}")

    # 클래스별 통계
    print("\nCrops per class:")
    for class_id in sorted(merged_crops.keys()):
        if class_id < len(class_names):
            name = class_names[class_id].split(" -")[0][:25]
            count = len(merged_crops[class_id])
            print(f"  {class_id:2d}. {name:<25}: {count:4d}")

    # 분할 및 저장
    print("\n[Step 3] Splitting and saving...")
    stats = split_and_save(
        merged_crops,
        class_names,
        output_path,
        args.train_ratio,
        args.val_ratio,
        test_ratio,
        args.seed
    )

    # Labels 파일 생성
    save_labels_file(output_path, class_names)

    # 최종 통계
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)

    for split_name in ["train", "val", "test"]:
        split_stats = stats[split_name]
        total = sum(split_stats.values())
        n_classes = len([v for v in split_stats.values() if v > 0])
        print(f"\n{split_name.upper()}: {total} images, {n_classes}/19 classes")

    # 검증
    print("\n" + "=" * 60)
    all_good = True
    for split_name in ["train", "val", "test"]:
        n_classes = len([v for v in stats[split_name].values() if v > 0])
        status = "OK" if n_classes == 19 else "MISSING"
        print(f"{split_name}: {n_classes}/19 classes [{status}]")
        if n_classes != 19:
            all_good = False

    if all_good:
        print("\nAll 19 classes present in all splits!")

    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
