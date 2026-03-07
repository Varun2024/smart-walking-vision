import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def parse_args():
    parser = argparse.ArgumentParser(description='Quantize ONNX detection model to INT8')
    parser.add_argument('--input', default='yolov8n.onnx', help='Path to source FP32 ONNX model')
    parser.add_argument('--output', default='yolov8n.int8.onnx', help='Path to output INT8 ONNX model')
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f'Input model not found: {input_path}')

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    print(f'INT8 model written to: {output_path}')


if __name__ == '__main__':
    main()
