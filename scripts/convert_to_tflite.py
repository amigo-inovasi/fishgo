#!/usr/bin/env python3
"""
FishGo PyTorch → TFLite 변환 스크립트

PyTorch → ONNX → TensorFlow SavedModel → TFLite 변환
정확도 우선이므로 FP16 사용 (INT8보다 정확)

Usage:
    python convert_to_tflite.py \
        --model ./runs/classifier/best_model.pth \
        --output ./app/src/main/assets/model/fish_classifier.tflite \
        --classes ./runs/classifier/class_mapping.json

Requirements:
    pip install torch torchvision onnx onnx-tf tensorflow
    # 또는
    pip install torch torchvision onnx onnx2tf tensorflow
"""

import os
import sys
import argparse
import json
from pathlib import Path
import shutil

import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes: int = 19):
    """EfficientNet-B0 모델 생성"""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_pytorch_model(model_path: str, num_classes: int = 19) -> nn.Module:
    """
    PyTorch 모델 로드

    Args:
        model_path: .pth 파일 경로
        num_classes: 클래스 수

    Returns:
        로드된 모델
    """
    model = create_model(num_classes)

    checkpoint = torch.load(model_path, map_location='cpu')

    # checkpoint 형식에 따라 처리
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 224, 224)
):
    """
    PyTorch 모델을 ONNX로 변환

    Args:
        model: PyTorch 모델
        output_path: ONNX 파일 출력 경로
        input_size: 입력 텐서 크기
    """
    print(f"Exporting to ONNX: {output_path}")

    dummy_input = torch.randn(input_size)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"ONNX model saved: {output_path}")


def convert_onnx_to_tflite_via_onnx2tf(
    onnx_path: str,
    output_dir: str,
    quantization: str = "fp16"
) -> str:
    """
    onnx2tf를 사용한 ONNX -> TFLite 변환

    Args:
        onnx_path: ONNX 파일 경로
        output_dir: 출력 디렉토리
        quantization: 양자화 방식 ("fp16", "fp32", "int8")

    Returns:
        TFLite 파일 경로
    """
    try:
        import onnx2tf
    except ImportError:
        print("onnx2tf not installed. Installing...")
        os.system("pip install onnx2tf")
        import onnx2tf

    print(f"Converting ONNX to TFLite using onnx2tf...")

    # onnx2tf 변환
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_dir,
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True
    )

    # 생성된 TFLite 파일 찾기
    tflite_files = list(Path(output_dir).glob("*.tflite"))
    if tflite_files:
        return str(tflite_files[0])

    return None


def convert_onnx_to_tflite_via_tf(
    onnx_path: str,
    output_path: str,
    quantization: str = "fp16"
) -> str:
    """
    TensorFlow를 통한 ONNX -> TFLite 변환

    Args:
        onnx_path: ONNX 파일 경로
        output_path: TFLite 출력 경로
        quantization: 양자화 방식

    Returns:
        TFLite 파일 경로
    """
    import tensorflow as tf

    # ONNX -> TF SavedModel
    try:
        from onnx_tf.backend import prepare
        import onnx

        print("Converting ONNX to TensorFlow SavedModel...")

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)

        saved_model_dir = output_path.replace('.tflite', '_saved_model')
        tf_rep.export_graph(saved_model_dir)

        print(f"SavedModel saved: {saved_model_dir}")

    except ImportError:
        print("onnx-tf not available, trying alternative method...")
        return None

    # SavedModel -> TFLite
    print("Converting SavedModel to TFLite...")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # 양자화 설정
    if quantization == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("Using FP16 quantization")
    elif quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("Using INT8 quantization (requires representative dataset)")
    else:
        print("Using FP32 (no quantization)")

    # 변환
    tflite_model = converter.convert()

    # 저장
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    # SavedModel 정리
    shutil.rmtree(saved_model_dir, ignore_errors=True)

    print(f"TFLite model saved: {output_path}")
    print(f"Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return output_path


def convert_pytorch_to_tflite_direct(
    model: nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 224, 224),
    quantization: str = "fp16"
) -> str:
    """
    PyTorch -> TFLite 직접 변환 (ai-edge-torch 사용)

    이 방법이 가장 깔끔하지만 ai-edge-torch 설치 필요

    Args:
        model: PyTorch 모델
        output_path: TFLite 출력 경로
        input_size: 입력 크기
        quantization: 양자화 방식

    Returns:
        TFLite 파일 경로
    """
    try:
        import ai_edge_torch

        print("Using ai-edge-torch for direct conversion...")

        sample_input = torch.randn(input_size)

        edge_model = ai_edge_torch.convert(
            model,
            (sample_input,)
        )

        edge_model.export(output_path)

        print(f"TFLite model saved: {output_path}")
        return output_path

    except ImportError:
        print("ai-edge-torch not available")
        return None


def generate_labels_file(
    class_mapping_path: str,
    output_path: str,
    indonesian_names: dict = None
):
    """
    Android용 labels.txt 생성

    Format: scientific_name,indonesian_name

    Args:
        class_mapping_path: class_mapping.json 경로
        output_path: labels.txt 출력 경로
        indonesian_names: 인도네시아어 이름 매핑
    """
    # 기본 인도네시아어 이름
    if indonesian_names is None:
        indonesian_names = {
            "Alepes_Djedaba": "Selar Bulat",
            "Atropus_Atropos": "Cipa-Cipa",
            "Caranx_Ignobilis": "Kuwe",
            "Chanos_Chanos": "Bandeng",
            "Decapterus_Macarellus": "Malalugis",
            "Euthynnus_Affinis": "Tongkol",
            "Katsuwonus_Pelamis": "Cakalang",
            "Lutjanus_Malabaricus": "Kakap Merah",
            "Parastromateus_Niger": "Bawal Hitam",
            "Rastrelliger_Kanagurta": "Kembung",
            "Rastrelliger_sp": "Kembung Banjar",
            "Scaridae": "Kakatua",
            "Scomber_Japonicus": "Salem",
            "Scomberomorus_Guttatus": "Tenggiri Papan",
            "Thunnus_Alalunga": "Albacore Tuna",
            "Thunnus_Obesus": "Tuna Mata Besar",
            "Thunnus_Tonggol": "Tuna",
            "Tribus_Sardini": "Kenyar",
            "Upeneus_Moluccensis": "Kuniran"
        }

    # 클래스 매핑 로드
    with open(class_mapping_path, 'r') as f:
        mapping = json.load(f)

    idx_to_class = mapping.get('idx_to_class', {})

    # labels.txt 생성
    with open(output_path, 'w') as f:
        for idx in sorted(int(k) for k in idx_to_class.keys()):
            class_name = idx_to_class[str(idx)]
            # 언더스코어를 공백으로 변환하여 인도네시아어 이름 찾기
            indo_name = indonesian_names.get(
                class_name,
                indonesian_names.get(class_name.replace(" ", "_"), class_name)
            )
            # 클래스명에서 언더스코어를 공백으로 복원
            scientific_name = class_name.replace("_", " ")
            f.write(f"{scientific_name},{indo_name}\n")

    print(f"Labels file saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model to TFLite"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to PyTorch model (.pth)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output TFLite file path"
    )
    parser.add_argument(
        "--classes", "-c",
        type=str,
        help="Path to class_mapping.json (for labels.txt generation)"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=19,
        help="Number of classes (default: 19)"
    )
    parser.add_argument(
        "--quantization", "-q",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "int8"],
        help="Quantization type (default: fp16)"
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. PyTorch 모델 로드
    print("=" * 50)
    print("Step 1: Loading PyTorch model")
    print("=" * 50)

    model = load_pytorch_model(args.model, args.num_classes)
    print(f"Model loaded: {args.model}")

    # 2. 변환 시도 (여러 방법)
    print("\n" + "=" * 50)
    print("Step 2: Converting to TFLite")
    print("=" * 50)

    tflite_path = None

    # 방법 1: ai-edge-torch (권장)
    tflite_path = convert_pytorch_to_tflite_direct(
        model,
        str(output_path),
        quantization=args.quantization
    )

    # 방법 2: ONNX -> TFLite
    if tflite_path is None:
        print("\nTrying ONNX conversion method...")

        onnx_path = str(output_path).replace('.tflite', '.onnx')
        export_to_onnx(model, onnx_path)

        # onnx2tf 시도
        temp_dir = str(output_path.parent / "temp_tflite")
        tflite_path = convert_onnx_to_tflite_via_onnx2tf(
            onnx_path,
            temp_dir,
            args.quantization
        )

        if tflite_path:
            # 원하는 위치로 이동
            shutil.move(tflite_path, str(output_path))
            tflite_path = str(output_path)
            shutil.rmtree(temp_dir, ignore_errors=True)

        # onnx-tf 시도
        if tflite_path is None:
            tflite_path = convert_onnx_to_tflite_via_tf(
                onnx_path,
                str(output_path),
                args.quantization
            )

        # ONNX 파일 정리
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

    if tflite_path is None:
        print("\nERROR: All conversion methods failed!")
        print("Please install one of:")
        print("  pip install ai-edge-torch")
        print("  pip install onnx2tf")
        print("  pip install onnx-tf")
        sys.exit(1)

    # 3. Labels 파일 생성
    if args.classes:
        print("\n" + "=" * 50)
        print("Step 3: Generating labels.txt")
        print("=" * 50)

        labels_path = output_path.parent / "labels.txt"
        generate_labels_file(args.classes, str(labels_path))

    # 완료
    print("\n" + "=" * 50)
    print("Conversion Complete!")
    print("=" * 50)
    print(f"TFLite model: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
