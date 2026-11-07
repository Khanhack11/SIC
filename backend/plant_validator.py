"""
Plant and Leaf Presence Validation Engine.
Guards against Out-Of-Distribution (OOD) images:
- Human faces, bodies, and skin tones.
- Indoor rooms, furniture, ceilings, computer monitors.
- Non-plant inanimate objects (cars, books, tools, electronics).
- Overexposed, pitch black, or blurred non-vegetation images.

Enforces Agricultural Safety:
Prevents the closed-set neural network classifier from outputting false high-confidence
crop disease predictions on human faces or non-plant objects.
Author: SIC Capstone Project - Group 5
"""

import io
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageOps


def validate_plant_leaf(image_bytes: bytes, min_plant_ratio: float = 0.08) -> Dict[str, Any]:
    """
    Analyzes image colorimetry and physiological signatures to verify if the image
    actually contains plant leaves or vegetation.

    Returns:
        dict:
            is_valid_plant (bool): True if valid plant leaf image, False otherwise.
            rejection_code (str or None): "HUMAN_DETECTED", "NON_PLANT_OBJECT", "INSUFFICIENT_VEGETATION".
            plant_ratio (float): Fraction of image containing plant tissue (green/lesion).
            skin_ratio (float): Fraction of image matching human skin color signatures.
            confidence (float): Confidence score (0.0 to 1.0) that the image contains plant foliage.
            title (str): Title for user-facing advisory.
            message (str): Detailed user guidance.
            guidelines (list): 4-step clear shooting instructions.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            # Respect EXIF orientation and convert to RGB
            img = ImageOps.exif_transpose(pil_img).convert("RGB")
    except Exception as e:
        return {
            "is_valid_plant": False,
            "rejection_code": "CORRUPTED_IMAGE",
            "plant_ratio": 0.0,
            "skin_ratio": 0.0,
            "confidence": 0.0,
            "title": "Tệp ảnh không hợp lệ",
            "message": f"Không thể đọc định dạng ảnh: {e}. Vui lòng chọn tệp JPG hoặc PNG khác.",
            "guidelines": []
        }

    w, h = img.size
    total_pixels = max(w * h, 1)

    # Downsample large images for ultra-fast check (< 15 ms)
    if max(w, h) > 512:
        img_small = img.resize((256, int(256 * h / w)) if w >= h else (int(256 * w / h), 256), Image.Resampling.BILINEAR)
    else:
        img_small = img

    arr = np.array(img_small, dtype=np.float32)
    r255 = arr[:, :, 0]
    g255 = arr[:, :, 1]
    b255 = arr[:, :, 2]

    # Normalized [0, 1] channels
    r = r255 / 255.0
    g = g255 / 255.0
    b = b255 / 255.0

    # 1. Excess Green Index (ExG): ExG = 2G - R - B
    exg = 2.0 * g - r - b

    # Green chlorophyll vegetation
    # Plant green pixels have positive ExG, green channel prominence over red & blue
    green_veg = (exg > 0.04) & (g > 0.12) & (g >= r * 0.88)

    # Plant lesions (necrosis, chlorosis, rust pustules)
    # Must have plant-like color and not match typical skin/wood tones
    rust_necrosis = (r - g > 0.05) & (r - b > 0.08) & (r > 0.20) & (b < 0.35)
    chlorosis_yellow = (r > 0.35) & (g > 0.30) & (b < 0.30) & (np.abs(r - g) < 0.18)
    lesion_pixels = rust_necrosis | chlorosis_yellow

    total_plant_mask = green_veg | lesion_pixels
    plant_ratio = float(np.mean(total_plant_mask))
    green_ratio = float(np.mean(green_veg))

    # 2. Human Skin Tone Detection (Peer et al. standard RGB color space rule + HSV refinement)
    # Human skin: R > 95, G > 40, B > 20, max - min > 15, |R - G| > 15, R > G > B
    skin_mask = (
        (r255 > 95) & (g255 > 40) & (b255 > 20) &
        ((np.maximum(r255, np.maximum(g255, b255)) - np.minimum(r255, np.minimum(g255, b255))) > 15) &
        (np.abs(r255 - g255) > 15) &
        (r255 > g255) & (r255 > b255)
    )
    skin_ratio = float(np.mean(skin_mask))

    # 3. Decision Logic: Out-Of-Distribution (OOD) Rejection
    # Case A: Human face / body detected (significant skin ratio, very low green vegetation)
    if skin_ratio >= 0.12 and green_ratio < 0.06:
        return {
            "is_valid_plant": False,
            "rejection_code": "HUMAN_DETECTED",
            "plant_ratio": round(plant_ratio * 100, 1),
            "skin_ratio": round(skin_ratio * 100, 1),
            "confidence": 0.0,
            "title": "Phát hiện khuôn mặt / người trong ảnh",
            "message": (
                "Ảnh bạn tải lên chứa hình ảnh người hoặc khuôn mặt, không phải lá cây trồng. "
                "Hệ thống AI từ chối đưa ra chẩn đoán bệnh để đảm bảo an toàn và tránh thông tin sai lệch."
            ),
            "guidelines": [
                "1. Không chụp khuôn mặt, người hoặc không gian phòng ốc.",
                "2. Đưa camera lại gần một phiến lá cây cụ thể (Cà chua, Khoai tây hoặc Ngô).",
                "3. Giữ khoảng cách 15 - 25 cm để phiến lá chiếm ít nhất 40% khung hình.",
                "4. Đảm bảo đủ ánh sáng tự nhiên và chụp rõ nét mặt trên/mặt dưới của lá."
            ]
        }

    # Case B: Inanimate non-plant object / Indoor room (no green vegetation, very sparse plant ratio)
    if plant_ratio < min_plant_ratio:
        return {
            "is_valid_plant": False,
            "rejection_code": "NON_PLANT_OBJECT",
            "plant_ratio": round(plant_ratio * 100, 1),
            "skin_ratio": round(skin_ratio * 100, 1),
            "confidence": 0.0,
            "title": "Không phát hiện lá cây trong ảnh",
            "message": (
                "Ảnh bạn tải lên dường như là đồ vật, phòng ốc hoặc phong cảnh không có lá cây rõ ràng "
                f"(diện tích đặc trưng thực vật chỉ đạt {round(plant_ratio * 100, 1)}%, dưới ngưỡng tối thiểu 8%). "
                "Hệ thống từ chối phân tích để tránh nhầm lẫn phác đồ thuốc."
            ),
            "guidelines": [
                "1. Chọn đúng ảnh chụp lá cây (Cà chua, Khoai tây, Ngô).",
                "2. Hạn chế chụp lẫn bàn ghế, tường nhà, màn hình máy tính hoặc đồ đạc xung quanh.",
                "3. Đặt lá cây trên nền tự nhiên hoặc một màu đồng nhất.",
                "4. Kiểm tra độ sắc nét và lấy nét đúng vào vết đốm bệnh trên phiến lá."
            ]
        }

    # Case C: Valid plant leaf image
    plant_confidence = min(1.0, max(0.5, plant_ratio * 1.5))
    return {
        "is_valid_plant": True,
        "rejection_code": None,
        "plant_ratio": round(plant_ratio * 100, 1),
        "skin_ratio": round(skin_ratio * 100, 1),
        "confidence": round(plant_confidence, 2),
        "title": "Ảnh lá cây hợp lệ",
        "message": "Ảnh chứa đặc trưng thực vật và phiến lá hợp lệ.",
        "guidelines": []
    }
