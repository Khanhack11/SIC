"""Export a verified FP32 checkpoint and optionally calibrate an INT8 candidate."""

import argparse
import hashlib
import json
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from backend.config import CLASS_NAMES, MODEL_INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD
from backend.inference_engine import CropDiseaseInferenceEngine
from ml.dataset import ManifestDataset
from ml.runtime import check_pause


def export(weights, output, data, quantized=None):
    check_pause()
    torch.set_num_threads(4)
    engine = CropDiseaseInferenceEngine(weights, device="cpu")
    if not engine.weights_loaded:
        raise ValueError(engine.load_error or "No trained checkpoint")
    if output.exists() or (quantized and quantized.exists()):
        raise ValueError("Refusing to overwrite an existing ONNX graph")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = engine.model.to(memory_format=torch.contiguous_format)
    torch.onnx.export(model, torch.zeros(1, 3, *MODEL_INPUT_SIZE), str(output),
                      input_names=["images"], output_names=["logits"],
                      dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    contract = {"architecture": "mobilenet_v2", "class_names": CLASS_NAMES,
                "input_size": list(MODEL_INPUT_SIZE), "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                "temperature": engine.temperature, "calibrated": engine.calibrated,
                "checkpoint_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(), "precision": "fp32"}

    def annotate(path, metadata):
        graph = onnx.load(str(path))
        onnx.helper.set_model_props(graph, {"sic_contract": json.dumps(metadata)})
        onnx.checker.check_model(graph)
        onnx.save(graph, str(path))

    annotate(output, contract)
    print(f"Exported {output}", flush=True)
    if quantized:
        validation = ManifestDataset(data / "validation.csv")
        indices = []
        for label in range(len(CLASS_NAMES)):
            indices.extend([index for index, row in enumerate(validation.rows) if int(row["class_index"]) == label][:10])

        class Reader(CalibrationDataReader):
            def __init__(self):
                self.iterator = iter(indices)

            def get_next(self):
                check_pause()
                index = next(self.iterator, None)
                if index is None:
                    return None
                image, _ = validation[index]
                return {"images": image.unsqueeze(0).numpy()}

        quantize_static(str(output), str(quantized), Reader(), quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                        per_channel=True, op_types_to_quantize=["Conv", "Gemm", "MatMul"])
        annotate(quantized, {**contract, "precision": "int8", "calibration_split": "validation",
                             "calibration_images": len(indices),
                             "calibration_manifest_sha256": hashlib.sha256(validation.manifest.read_bytes()).hexdigest()})
        print(f"Exported candidate {quantized}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("models/mobilenetv2_crop_disease.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/mobilenetv2_crop_disease.onnx"))
    parser.add_argument("--data", type=Path, default=Path("data/processed/plantvillage"))
    parser.add_argument("--quantized", type=Path)
    args = parser.parse_args()
    export(args.weights, args.output, args.data, args.quantized)


if __name__ == "__main__":
    main()
