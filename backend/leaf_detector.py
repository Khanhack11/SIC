"""
Automatic Leaf & Plant Disease Lesion Detection Engine.
Automatically locates individual leaves, plant clusters, and diseased spots on field images.
Extracts bounding boxes with magenta visualization (matching agricultural field standards).
Author: SIC Capstone Project - Group 5
"""

import base64
import io
import time
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from backend.plant_validator import validate_plant_leaf


def detect_leaf_regions(
    image_bytes: bytes,
    max_boxes: int = 7,
    border_color: Tuple[int, int, int] = (233, 30, 99)  # Magenta #e91e63
) -> Dict[str, Any]:
    """
    Scans input image, segments plant foliage using Excess Green (ExG) and lesion salience,
    and returns bounding boxes around individual leaves and diseased patches.
    """
    start_time = time.perf_counter()

    # Pre-flight check: Verify plant foliage presence (reject human face, indoor rooms, non-plant objects)
    plant_check = validate_plant_leaf(image_bytes)
    if not plant_check["is_valid_plant"]:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        return {
            "has_leaves": False,
            "is_valid_plant": False,
            "rejection_code": plant_check["rejection_code"],
            "title": plant_check["title"],
            "message": plant_check["message"],
            "guidelines": plant_check["guidelines"],
            "plant_ratio": plant_check["plant_ratio"],
            "skin_ratio": plant_check["skin_ratio"],
            "total_leaves_detected": 0,
            "count": 0,
            "boxes": [],
            "primary_crop_str": None,
            "primary_box": None,
            "annotated_image_preview": None,
            "execution_ms": latency_ms,
            "image_size": [0, 0]
        }

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = image.size

    arr = np.array(image, dtype=np.float32) / 255.0
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # 1. Excess Green Index: ExG = 2G - R - B
    exg = 2.0 * g - r - b

    # 2. Disease Lesion Index: yellowing (chlorosis) or brownish rust/necrosis
    rust_necrosis = (r - g > 0.05) & (r - b > 0.08) & (r > 0.20) & (b < 0.35)
    chlorosis_yellow = (r > 0.35) & (g > 0.30) & (b < 0.30) & (np.abs(r - g) < 0.18)
    lesion_mask = (rust_necrosis | chlorosis_yellow)

    # 3. Primary plant mask: Green vegetation OR Discolored lesions
    green_veg = (exg > 0.04) & (g > 0.12) & (g >= r * 0.88)
    plant_mask = green_veg | lesion_mask

    # 4. Connected components on downsampled grid (step=2) for high speed (< 30ms)
    visited = np.zeros_like(plant_mask, dtype=bool)
    raw_boxes = []

    for y in range(0, orig_h, 2):
        for x in range(0, orig_w, 2):
            if plant_mask[y, x] and not visited[y, x]:
                q = [(y, x)]
                visited[y, x] = True
                min_y, max_y = y, y
                min_x, max_x = x, x
                cnt = 0
                lesion_cnt = 0

                while q:
                    cy, cx = q.pop()
                    cnt += 1
                    if lesion_mask[cy, cx]:
                        lesion_cnt += 1
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)

                    for dy, dx in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < orig_h and 0 <= nx < orig_w and plant_mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))

                bw = max_x - min_x
                bh = max_y - min_y
                area = bw * bh
                total_area = orig_w * orig_h

                # Filter noise (< 750 pixels) and whole-frame artifacts (> 88%)
                if area > 750 and bw > 25 and bh > 25 and (area / total_area) <= 0.88:
                    # Add 8% margin padding
                    px = max(4, int(bw * 0.08))
                    py = max(4, int(bh * 0.08))
                    bx = max(0, min_x - px)
                    by = max(0, min_y - py)
                    bw_p = min(orig_w - bx, bw + 2 * px)
                    bh_p = min(orig_h - by, bh + 2 * py)

                    lesion_ratio = lesion_cnt / max(cnt, 1)

                    raw_boxes.append({
                        "x": int(bx),
                        "y": int(by),
                        "w": int(bw_p),
                        "h": int(bh_p),
                        "cnt": int(cnt),
                        "area": int(area),
                        "lesion_ratio": float(lesion_ratio)
                    })

    # Only if plant ratio is high (macro close-up where entire frame is a leaf) and no distinct isolated box found:
    if not raw_boxes and plant_check["plant_ratio"] >= 30.0:
        raw_boxes.append({
            "x": int(orig_w * 0.08),
            "y": int(orig_h * 0.08),
            "w": int(orig_w * 0.84),
            "h": int(orig_h * 0.84),
            "cnt": int(orig_w * orig_h * 0.5),
            "area": int(orig_w * orig_h * 0.7),
            "lesion_ratio": 0.1
        })

    # Sort boxes: prioritize lesion density and leaf prominence
    raw_boxes.sort(key=lambda b: (b["lesion_ratio"] * 2.0 + (b["cnt"] / (orig_w * orig_h))), reverse=True)
    selected = raw_boxes[:max_boxes]

    # Non-maximum suppression to filter out redundant heavy overlaps
    final_boxes = []
    for candidate in selected:
        cx1, cy1, cw1, ch1 = candidate["x"], candidate["y"], candidate["w"], candidate["h"]
        overlap = False
        for kept in final_boxes:
            cx2, cy2, cw2, ch2 = kept["x"], kept["y"], kept["w"], kept["h"]
            ix0 = max(cx1, cx2)
            iy0 = max(cy1, cy2)
            ix1 = min(cx1 + cw1, cx2 + cw2)
            iy1 = min(cy1 + ch1, cy2 + ch2)
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                union = (cw1 * ch1) + (cw2 * ch2) - inter
                if inter / max(union, 1) > 0.65:
                    overlap = True
                    break
        if not overlap:
            final_boxes.append(candidate)

    if not final_boxes:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
        return {
            "has_leaves": False,
            "is_valid_plant": True,
            "count": 0,
            "primary_crop_str": "0,0,1,1",
            "primary_box": None,
            "boxes": [],
            "annotated_image_preview": None,
            "execution_ms": elapsed_ms,
            "image_size": [orig_w, orig_h]
        }

    # Re-sort final boxes by size / prominence
    final_boxes.sort(key=lambda b: b["cnt"], reverse=True)

    # Format output boxes and draw magenta visualization
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    formatted = []

    for idx, b in enumerate(final_boxes, start=1):
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        norm_x = round(x / orig_w, 4)
        norm_y = round(y / orig_h, 4)
        norm_w = round(w / orig_w, 4)
        norm_h = round(h / orig_h, 4)
        is_primary = (idx == 1)
        is_diseased = b["lesion_ratio"] > 0.08

        formatted.append({
            "id": idx,
            "label": f"Lá #{idx} ({'Đốm bệnh' if is_diseased else 'Phiến lá'})",
            "is_primary": is_primary,
            "is_diseased": is_diseased,
            "area_ratio": round(norm_w * norm_h, 4),
            "lesion_ratio": round(b["lesion_ratio"] * 100, 1),
            "pixel_box": [x, y, w, h],
            "norm_box": [norm_x, norm_y, norm_w, norm_h],
            "crop_str": f"{norm_x},{norm_y},{norm_w},{norm_h}"
        })

        # Draw magenta bounding box
        width = 4 if is_primary else 3
        draw.rectangle([x, y, x + w, y + h], outline=border_color, width=width)

        # Draw tag label
        tag = f"#{idx}" + (" [BỆNH]" if is_diseased else " [LÁ]")
        badge_h = 16
        badge_w = len(tag) * 8 + 6
        draw.rectangle([x, max(0, y - badge_h), x + badge_w, y], fill=border_color)
        draw.text((x + 3, max(0, y - badge_h) + 1), tag, fill=(255, 255, 255))

    primary_crop = formatted[0]["crop_str"] if formatted else "0,0,1,1"

    # Encode annotated image preview to Base64 JPEG
    buffer = io.BytesIO()
    annotated.save(buffer, format="JPEG", quality=85)
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

    return {
        "count": len(formatted),
        "primary_crop_str": primary_crop,
        "primary_box": formatted[0] if formatted else None,
        "boxes": formatted,
        "annotated_image_preview": annotated_b64,
        "execution_ms": elapsed_ms,
        "image_size": [orig_w, orig_h]
    }


def auto_crop_leaf(image_bytes: bytes, crop_box: Optional[str] = None) -> bytes:
    """
    Crops leaf image. If crop_box (norm_x,norm_y,norm_w,norm_h) is None,
    automatically detects the primary leaf and crops it.
    """
    image = Image.open(io.BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image)
    orig_w, orig_h = image.size

    if not crop_box:
        detected = detect_leaf_regions(image_bytes)
        crop_box = detected["primary_crop_str"]

    coords = [float(v.strip()) for v in crop_box.split(",") if v.strip()]
    if len(coords) != 4:
        raise ValueError("crop_box must have 4 coordinates: x,y,w,h")

    nx, ny, nw, nh = coords
    x0 = max(0, int(nx * orig_w))
    y0 = max(0, int(ny * orig_h))
    x1 = min(orig_w, max(x0 + 10, int((nx + nw) * orig_w)))
    y1 = min(orig_h, max(y0 + 10, int((ny + nh) * orig_h)))

    cropped = image.crop((x0, y0, x1, y1))
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=92)
    return buf.getvalue()
