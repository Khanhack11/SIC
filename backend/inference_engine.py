"""
Image Preprocessing and Model Inference Pipeline (Subsystem A)
Author: Nguyen Quoc Khanh (Backend Developer & Database Architect)
SIC Capstone Project - Group 5
"""

import io
import time
from pathlib import Path
from typing import Dict, Any
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models

from backend.preprocessing import image_transform, orient_rgb

from backend.config import (
    MODEL_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    CLASS_NAMES,
    MODEL_WEIGHTS_PATH,
)


class InputValidationError(Exception):
    """Raised when an uploaded image fails validation."""
    pass


class ModelNotReadyError(RuntimeError):
    """No verified, trained checkpoint is available for diagnosis."""


def validate_image_file(file_bytes: bytes, filename: str) -> None:
    """
    Validates uploaded image file:
    1. Checks file size against MAX_FILE_SIZE_BYTES.
    2. Checks file extension against ALLOWED_EXTENSIONS.
    3. Verifies that the file content is a readable, uncorrupted image.
    """
    if len(file_bytes) == 0:
        raise InputValidationError("File rỗng (0 bytes). Vui lòng chọn một ảnh hợp lệ.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise InputValidationError(
            f"Kích thước file vượt quá giới hạn cho phép ({max_mb:.1f} MB)."
        )

    # Extension check
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise InputValidationError(
            f"Định dạng tệp '{ext}' không được hỗ trợ. Vui lòng tải ảnh JPG hoặc PNG."
        )

    # Corruption check
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            if img.format not in {"JPEG", "PNG"}:
                raise ValueError("Nội dung ảnh phải là JPEG hoặc PNG.")
            if min(img.size) < 32 or img.width * img.height > 20_000_000:
                raise ValueError("Ảnh phải có cạnh tối thiểu 32px và không vượt 20 megapixel.")
            img.verify()  # Fast structural verification
    except Exception as e:
        raise InputValidationError(f"File ảnh bị lỗi hoặc hỏng, không thể đọc: {str(e)}")


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """
    Preprocesses raw image bytes into a normalized tensor:
    1. Decodes and converts to 3-channel RGB.
    2. Resizes to 224x224 pixels.
    3. Normalizes using ImageNet mean and std baselines.
    4. Adds batch dimension: (1, 3, 224, 224).
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            tensor = image_transform()(orient_rgb(image))
    except (OSError, ValueError) as exc:
        raise InputValidationError(f"Không thể giải mã đầy đủ ảnh: {exc}") from exc
    return tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224)


class CropDiseaseInferenceEngine:
    """
    MobileNetV2 Inference Engine for Crop Disease Classification.
    Provides calibrated confidence scores and sub-200ms response profiling.
    """

    def __init__(self, weights_path=MODEL_WEIGHTS_PATH, class_names=CLASS_NAMES, device=None, runtime="eager"):
        self.class_names = list(class_names)
        self.num_classes = len(class_names)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if runtime not in {"eager", "torchscript"}:
            raise ValueError("runtime must be eager or torchscript")
        self.runtime = runtime
        self.memory_format = torch.channels_last if self.device.type == "cpu" else torch.contiguous_format
        self.temperature = 1.0
        self.calibrated = False
        self.load_error = None

        # Initialize MobileNetV2 architecture
        self.model = self._build_model()
        self.weights_loaded = False

        weights_path = Path(weights_path) if weights_path else None
        if weights_path and weights_path.exists():
            try:
                checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
                if "state_dict" in checkpoint:
                    expected = {"class_names": self.class_names, "input_size": list(MODEL_INPUT_SIZE),
                                "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "architecture": "mobilenet_v2"}
                    for key, value in expected.items():
                        if checkpoint.get(key) != value:
                            raise ValueError(f"Checkpoint {key} differs from the serving contract")
                    state_dict = checkpoint["state_dict"]
                    self.temperature = float(checkpoint.get("temperature", 1.0))
                    if not 0.05 <= self.temperature <= 10.0:
                        raise ValueError("Invalid checkpoint temperature")
                    self.calibrated = bool(checkpoint.get("calibration"))
                else:
                    # Existing ML-lead state_dict files remain compatible, but are uncalibrated.
                    state_dict = checkpoint
                self.model.load_state_dict(state_dict)
                self.weights_loaded = True
                print(f"[InferenceEngine] Loaded model weights from {weights_path}")
            except Exception as e:
                self.load_error = str(e)
                print(f"[InferenceEngine] Checkpoint unavailable: {e}")
        else:
            print("[InferenceEngine] No trained checkpoint found; diagnosis is unavailable until weights are provided.")

        self.model.to(self.device, memory_format=self.memory_format)
        self.model.eval()
        if runtime == "torchscript" and self.weights_loaded:
            example = torch.zeros(1, 3, *MODEL_INPUT_SIZE, device=self.device).to(memory_format=self.memory_format)
            with torch.inference_mode():
                self.model = torch.jit.freeze(torch.jit.trace(self.model, example))

    def _build_model(self) -> nn.Module:
        """Builds MobileNetV2 with custom classification head for crop diseases."""
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, self.num_classes)
        return model

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Runs model inference on input image bytes.
        Profiles latency and returns top prediction with confidence and all probabilities.
        """
        if not self.weights_loaded:
            raise ModelNotReadyError("Chưa có checkpoint đã huấn luyện hợp lệ. " + (self.load_error or ""))
        start_time = time.perf_counter()

        # Preprocessing
        input_tensor = preprocess_image(image_bytes).to(self.device, memory_format=self.memory_format)

        # Inference
        with torch.inference_mode():
            outputs = self.model(input_tensor)
            # One transfer synchronizes GPU work and avoids one .item() per class.
            probabilities = torch.softmax(outputs / self.temperature, dim=1)[0].cpu().tolist()

        predicted_idx = max(range(self.num_classes), key=probabilities.__getitem__)
        confidence = float(probabilities[predicted_idx])
        predicted_class = self.class_names[predicted_idx]

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Build full probabilities dictionary
        all_probs = {
            self.class_names[i]: round(float(probabilities[i]), 4)
            for i in range(self.num_classes)
        }

        # Crop name extraction (Tomato, Potato, Corn)
        crop = predicted_class.split("_")[0]
        is_healthy = "healthy" in predicted_class.lower()

        return {
            "predicted_class": predicted_class,
            "crop": crop,
            "is_healthy": is_healthy,
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "latency_ms": latency_ms,
            "benchmark_met": latency_ms < 200.0,
            "calibrated": self.calibrated,
            "all_probabilities": all_probs,
        }
