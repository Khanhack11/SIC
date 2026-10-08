"""
Configuration for SIC Crop Disease Backend System
Group 5 - Samsung Innovation Campus
"""

from pathlib import Path

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
