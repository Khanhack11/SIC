"""
Generate sample test images for SIC Crop Disease Backend testing.
"""

from pathlib import Path
from PIL import Image, ImageDraw

output_dir = Path(__file__).resolve().parent

# 1. Tomato leaf with spots (simulating Early Blight)
img_tomato = Image.new("RGB", (256, 256), color=(34, 139, 34))  # Forest green
draw = ImageDraw.Draw(img_tomato)
# Draw leaf veins
draw.line([(128, 20), (128, 240)], fill=(46, 160, 46), width=3)
draw.line([(128, 70), (50, 40)], fill=(46, 160, 46), width=2)
draw.line([(128, 70), (200, 40)], fill=(46, 160, 46), width=2)
draw.line([(128, 140), (40, 110)], fill=(46, 160, 46), width=2)
draw.line([(128, 140), (210, 110)], fill=(46, 160, 46), width=2)
# Draw brown necrotic spots with concentric circles (Early Blight concentric circles)
draw.ellipse([(60, 80), (95, 115)], fill=(139, 69, 19))
draw.ellipse([(68, 88), (87, 107)], fill=(160, 82, 45))
draw.ellipse([(73, 93), (82, 102)], fill=(101, 45, 10))
draw.ellipse([(150, 150), (190, 190)], fill=(139, 69, 19))
img_tomato.save(output_dir / "tomato_leaf_sample.jpg", format="JPEG")

# 2. Corn leaf sample (greenish yellow with rust pustules)
img_corn = Image.new("RGB", (224, 224), color=(60, 179, 113))
draw_corn = ImageDraw.Draw(img_corn)
# Parallel veins
for x in range(20, 220, 25):
    draw_corn.line([(x, 0), (x, 224)], fill=(80, 200, 120), width=1)
# Rust pustules (reddish brown)
for y in range(40, 200, 30):
    draw_corn.ellipse([(90, y), (105, y + 10)], fill=(178, 34, 34))
    draw_corn.ellipse([(130, y + 15), (145, y + 25)], fill=(178, 34, 34))
img_corn.save(output_dir / "corn_leaf_sample.png", format="PNG")

# 3. Healthy potato leaf sample (pure green, smooth)
img_potato = Image.new("RGB", (300, 300), color=(46, 139, 87))
draw_potato = ImageDraw.Draw(img_potato)
draw_potato.ellipse([(50, 40), (250, 260)], fill=(34, 150, 60))
img_potato.save(output_dir / "potato_leaf_sample.jpg", format="JPEG")

# 4. Corrupted image file (invalid bytes)
with open(output_dir / "corrupted_image.jpg", "wb") as f:
    f.write(b"NOT_A_REAL_JPEG_IMAGE_CORRUPTED_BYTES_1234567890")

# 5. Invalid format disguised as PNG
with open(output_dir / "invalid_disguised_file.png", "w", encoding="utf-8") as f:
    f.write("This is a plain text file pretending to be a PNG.")

print("[Success] Đã tạo thành công 5 file ảnh test trong thư mục sample_images/")
