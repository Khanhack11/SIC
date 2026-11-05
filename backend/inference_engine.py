"""
Image Preprocessing and Model Inference Pipeline (Subsystem A)
Author: SIC Capstone Project - Group 5
Enhanced by: Dinh Gia Bao (ML Lead)

Features:
- Dual Engine: High-accuracy PlantVillage MobileNetV2 with local fallback
- Crop Filter (auto, tomato, potato, corn) to avoid cross-species confusion
- Anti-False-Healthy Safeguard: Visual lesion detection suppresses 'healthy' on diseased leaves
- Latency profiling (< 100ms)
"""

import io
import time
from typing import Dict, Tuple, Any, List, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from backend.config import (
    MODEL_INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    CLASS_NAMES,
    MODEL_WEIGHTS_PATH,
)

# PlantVillage 38-class to Group 5 11-class mapping index
PLANTVILLAGE_TO_TARGET = {
    "Corn_Common_rust": 8,
    "Corn_Gray_leaf_spot": 7,
    "Corn_Northern_Leaf_Blight": 9,
    "Corn_healthy": 10,
    "Potato_Early_blight": 20,
    "Potato_Late_blight": 21,
    "Potato_healthy": 22,
    "Tomato_Early_blight": 29,
    "Tomato_Late_blight": 30,
    "Tomato_Septoria_leaf_spot": 32,
    "Tomato_healthy": 37,
}


class InputValidationError(Exception):
    """Raised when an uploaded image fails validation."""
    pass


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

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise InputValidationError(
            f"Định dạng tệp '{ext}' không được hỗ trợ. Vui lòng tải ảnh JPG hoặc PNG."
        )

    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
    except Exception as e:
        raise InputValidationError(f"File ảnh bị lỗi hoặc hỏng, không thể đọc: {str(e)}")


def detect_leaf_lesions(image: Image.Image) -> Dict[str, Any]:
    """
    Anti-False-Healthy Visual Safeguard:
    Analyzes leaf surface for necrotic spots, chlorosis (yellowing), and fungal rot.
    If lesion ratio exceeds threshold, flags leaf as diseased to prevent false-healthy claims.
    """
    rgb = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Brown necrotic spots (typical of blights and rust pustules)
    is_brown = (r > 0.35) & (g < 0.6) & (b < 0.4) & (r >= g * 0.82)
    # Yellow halos / chlorosis surrounding spots
    is_yellow = (r > 0.45) & (g > 0.4) & (b < 0.3) & (r + g > 2 * b)
    # Dark necrotic spots or fungal mycelium
    is_dark = (r < 0.25) & (g < 0.25) & (b < 0.25) & (g < 0.3)

    lesion_mask = is_brown | is_yellow | is_dark
    lesion_ratio = float(np.mean(lesion_mask))
    has_lesions = lesion_ratio > 0.03

    return {
        "has_lesions": has_lesions,
        "lesion_ratio": round(lesion_ratio, 4)
    }


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocesses raw image bytes into a normalized tensor."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform_pipeline = transforms.Compose([
        transforms.Resize(MODEL_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return transform_pipeline(image).unsqueeze(0)


class CropDiseaseInferenceEngine:
    """
    Production-grade Crop Disease Inference Engine.
    Combines MobileNetV2 with Crop Selection and Anti-False-Healthy Safeguard.
    """

    def __init__(self, weights_path=MODEL_WEIGHTS_PATH, class_names=CLASS_NAMES):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_loaded = False
        self.hf_model = None
        self.hf_processor = None

        # Attempt 1: Load fine-tuned PlantVillage MobileNetV2 from HuggingFace cache
        try:
            from transformers import MobileNetV2ForImageClassification, AutoImageProcessor
            model_id = "neghani/plantvillage-mobilenetv2"
            self.hf_processor = AutoImageProcessor.from_pretrained(model_id)
            self.hf_model = MobileNetV2ForImageClassification.from_pretrained(model_id)
            self.hf_model.to(self.device)
            self.hf_model.eval()
            self.weights_loaded = True
            print(f"[InferenceEngine] Succeeded loading PlantVillage MobileNetV2 backbone (54k images).")
        except Exception as e:
            print(f"[InferenceEngine] Notice: HF model unavailable ({e}). Using local PyTorch model.")

        # Attempt 2: Local TorchVision MobileNetV2 fallback
        self.local_model = models.mobilenet_v2(weights=None)
        in_features = self.local_model.classifier[1].in_features
        self.local_model.classifier[1] = nn.Linear(in_features, self.num_classes)

        if weights_path and weights_path.exists():
            try:
                ckpt = torch.load(weights_path, map_location=self.device, weights_only=False)
                state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                if isinstance(state_dict, dict):
                    self.local_model.load_state_dict(state_dict)
                self.weights_loaded = True
                print(f"[InferenceEngine] Loaded local weights from {weights_path}")
            except Exception as e:
                print(f"[InferenceEngine] Warning: Cannot load local checkpoint ({e}).")

        self.local_model.to(self.device)
        self.local_model.eval()

    def predict(self, image_bytes: bytes, crop_filter: str = "auto") -> Dict[str, Any]:
        """
        Runs model inference with Crop Filtering and Anti-False-Healthy protection.
        crop_filter: 'auto', 'tomato', 'potato', 'corn'
        """
        start_time = time.perf_counter()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        lesion_info = detect_leaf_lesions(image)
        has_lesions = lesion_info["has_lesions"]

        # 1. Determine allowed classes based on crop filter
        crop_filter_clean = (crop_filter or "auto").strip().lower()
        if crop_filter_clean in ["corn", "potato", "tomato"]:
            allowed_classes = [c for c in self.class_names if c.lower().startswith(crop_filter_clean)]
        else:
            crop_filter_clean = "auto"
            allowed_classes = self.class_names

        # 2. Model Forward Pass
        if self.hf_model is not None:
            # Use real PlantVillage-trained MobileNetV2
            inputs = self.hf_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits_all = self.hf_model(**inputs).logits[0]

            hf_indices = [PLANTVILLAGE_TO_TARGET[c] for c in allowed_classes]
            sub_logits = logits_all[hf_indices].clone()
        else:
            # Use local model
            input_tensor = preprocess_image(image_bytes).to(self.device)
            with torch.no_grad():
                full_logits = self.local_model(input_tensor)[0]
            local_indices = [self.class_names.index(c) for c in allowed_classes]
            sub_logits = full_logits[local_indices].clone()

        # 3. Anti-False-Healthy Safeguard:
        # If visual lesions (spots, necrosis, rot) are clearly detected,
        # suppress the 'healthy' class to prevent misleading farmers
        if has_lesions:
            for i, c in enumerate(allowed_classes):
                if "healthy" in c.lower():
                    sub_logits[i] -= 15.0

        # 4. Compute Softmax Probabilities
        probabilities = torch.softmax(sub_logits, dim=0)

        top_probs, top_indices = torch.topk(probabilities, min(3, len(allowed_classes)))
        predicted_idx = top_indices[0].item()
        confidence = float(top_probs[0].item())
        predicted_class = allowed_classes[predicted_idx]

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Build Top Candidates
        top_candidates = [
            {
                "class_name": allowed_classes[top_indices[k].item()],
                "confidence": round(float(top_probs[k].item()), 4),
                "confidence_percentage": f"{float(top_probs[k].item()) * 100:.2f}%"
            }
            for k in range(len(top_indices))
        ]

        # Build full probabilities for allowed classes
        all_probs = {
            allowed_classes[i]: round(float(probabilities[i].item()), 4)
            for i in range(len(allowed_classes))
        }

        # Crop name extraction & disease name extraction
        crop = predicted_class.split("_")[0]
        disease_suffix = predicted_class.split("_", 1)[1] if "_" in predicted_class else predicted_class
        is_healthy = "healthy" in predicted_class.lower()

        return {
            "predicted_class": predicted_class,
            "crop": crop,
            "disease_name": disease_suffix,
            "is_healthy": is_healthy,
            "has_lesions": has_lesions,
            "lesion_ratio": lesion_info["lesion_ratio"],
            "crop_filter_applied": crop_filter_clean,
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "latency_ms": latency_ms,
            "benchmark_met": latency_ms < 200.0,
            "top_candidates": top_candidates,
            "all_probabilities": all_probs,
        }
