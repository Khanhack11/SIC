"""
Configuration for SIC Crop Disease Backend System
Group 5 - Samsung Innovation Campus
"""

from pathlib import Path
import os
from importlib.util import find_spec

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"

# Knowledge base file
KNOWLEDGE_BASE_PATH = DATA_DIR / "advisory_knowledge_base.json"

# Model configuration
MODEL_WEIGHTS_PATH = MODELS_DIR / "mobilenetv2_crop_disease.pt"
MODEL_INPUT_SIZE = (224, 224)

# CPU runtime settings can be selected from ml.benchmark results.
TORCH_NUM_THREADS = int(os.environ.get("SIC_TORCH_THREADS", "2"))
INFERENCE_CONCURRENCY = int(os.environ.get("SIC_INFERENCE_CONCURRENCY", "2"))
ONNX_MODEL_PATH = Path(os.environ.get("SIC_ONNX_PATH", str(MODELS_DIR / "mobilenetv2_crop_disease.onnx")))
INFERENCE_RUNTIME = os.environ.get("SIC_INFERENCE_RUNTIME",
                                   "onnx" if ONNX_MODEL_PATH.is_file() and find_spec("onnxruntime") else "torchscript")
if TORCH_NUM_THREADS < 1:
    raise ValueError("SIC_TORCH_THREADS must be a positive integer")
if INFERENCE_CONCURRENCY < 1:
    raise ValueError("SIC_INFERENCE_CONCURRENCY must be a positive integer")

# ImageNet normalization parameters (Subsystem A requirement)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Decision-gated confidence threshold (Subsystem B requirement)
# Predictions >= 0.75 (75%) trigger full advisory cards
# Predictions < 0.75 prompt users for clearer photographs
CONFIDENCE_THRESHOLD = 0.75

# File validation constraints
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Target classes ordered index (matches PlantVillage multi-class dataset)
CLASS_NAMES = [
    "Corn_Common_rust",
    "Corn_Gray_leaf_spot",
    "Corn_Northern_Leaf_Blight",
    "Corn_healthy",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_healthy",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Septoria_leaf_spot",
    "Tomato_healthy"
]
