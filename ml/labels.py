"""Explicit PlantVillage-to-API labels; indices always follow backend.config."""

from backend.config import CLASS_NAMES

PLANTVILLAGE_LABELS = {
    "Corn_(maize)___Common_rust_": "Corn_Common_rust",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn_Gray_leaf_spot",
    "Corn_(maize)___Northern_Leaf_Blight": "Corn_Northern_Leaf_Blight",
    "Corn_(maize)___healthy": "Corn_healthy",
    "Potato___Early_blight": "Potato_Early_blight",
    "Potato___Late_blight": "Potato_Late_blight",
    "Potato___healthy": "Potato_healthy",
    "Tomato___Early_blight": "Tomato_Early_blight",
    "Tomato___Late_blight": "Tomato_Late_blight",
    "Tomato___Septoria_leaf_spot": "Tomato_Septoria_leaf_spot",
    "Tomato___healthy": "Tomato_healthy",
}
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def canonical_label(folder):
    if folder in CLASS_TO_INDEX:
        return folder
    return PLANTVILLAGE_LABELS.get(folder)
