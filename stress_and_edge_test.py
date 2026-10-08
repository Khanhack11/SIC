"""
Deep Stress and Edge Case Test Suite for SIC Crop Disease Backend System
Tests all endpoints, edge cases, error handling, input validation, and latency benchmarks.
"""

import sys
import io
import time
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure backend package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient
from backend.main import app
from backend.config import CLASS_NAMES, SAMPLE_IMAGES_DIR

client = TestClient(app)

def test_all():
    print("=" * 80)
    print(" KIỂM THỬ TOÀN DIỆN HỆ THỐNG VÀ CÁC TRƯỜNG HỢP BIÊN (EDGE CASES)")
    print("=" * 80)
    
    passed = 0
    total = 0

    def record_test(name, success, msg=""):
        nonlocal passed, total
        total += 1
        if success:
            passed += 1
            print(f" [PASS] {name} {msg}")
        else:
            print(f"❌ [FAIL] {name} {msg}")

    # 1. API System Endpoints
    print("\n--- 1. KIỂM THỬ CÁC ENDPOINTS CƠ BẢN ---")
    r = client.get("/")
    record_test("Root Endpoint '/'", r.status_code == 200, f"(Status: {r.status_code})")

    r = client.get("/api/health")
    record_test("Health Check '/api/health'", r.status_code == 200 and r.json()["status"] == "HEALTHY")

    r = client.get("/api/crops")
    record_test("Supported Crops '/api/crops'", r.status_code == 200 and len(r.json()["supported_crops"]) == 3)

    r = client.get("/api/diseases")
    record_test("All Diseases List '/api/diseases'", r.status_code == 200 and r.json()["total"] == 11)

    # 2. Disease Detail Endpoints
    print("\n--- 2. KIỂM THỬ TRA CỨU TỪNG BỆNH TRONG KNOWLEDGE BASE ---")
    all_diseases_ok = True
    for d_id in CLASS_NAMES:
        r = client.get(f"/api/diseases/{d_id}")
        if r.status_code != 200 or "advisory_data" not in r.json():
            all_diseases_ok = False
            break
    record_test(f"Tra cứu 11/11 bệnh chi tiết", all_diseases_ok)

    r = client.get("/api/diseases/DISEASE_NOT_EXIST")
    record_test("Tra cứu bệnh không tồn tại (Expect 404)", r.status_code == 404, f"(Code: {r.status_code})")

    # 3. Edge Cases for Image Upload (/api/diagnose)
    print("\n--- 3. KIỂM THỬ CÁC TRƯỜNG HỢP LỖI ĐẦU VÀO (EDGE CASES) ---")

    # Edge Case: File rỗng 0 bytes
    r = client.post("/api/diagnose", files={"file": ("empty.jpg", b"", "image/jpeg")})
    record_test("File rỗng 0 bytes (Expect 400)", r.status_code == 400 and "rỗng" in r.json()["detail"])

    # Edge Case: File sai định dạng (.txt, .exe, .pdf)
    r = client.post("/api/diagnose", files={"file": ("script.py", b"print('hello')", "text/x-python")})
    record_test("File không đúng đuôi ảnh .py (Expect 400)", r.status_code == 400 and "không được hỗ trợ" in r.json()["detail"])

    # Edge Case: File vượt quá 10MB
    large_fake_bytes = b"0" * (11 * 1024 * 1024)
    r = client.post("/api/diagnose", files={"file": ("huge.jpg", large_fake_bytes, "image/jpeg")})
    record_test("File vượt quá giới hạn 10MB (Expect 400)", r.status_code == 400 and "vượt quá" in r.json()["detail"])

    # Edge Case: File đuôi .jpg nhưng nội dung bị hỏng / corrupt
    r = client.post("/api/diagnose", files={"file": ("corrupt.jpg", b"\xFF\xD8\xFF\x00_CORRUPT_BYTES_DATA", "image/jpeg")})
    record_test("File ảnh hỏng cấu trúc (Expect 400)", r.status_code == 400 and "lỗi hoặc hỏng" in r.json()["detail"])

    # Edge Case: Ảnh Grayscale (1 channel)
    gray_img = Image.new("L", (200, 200), color=128)
    buf_gray = io.BytesIO()
    gray_img.save(buf_gray, format="JPEG")
    r = client.post("/api/diagnose", files={"file": ("gray.jpg", buf_gray.getvalue(), "image/jpeg")})
    record_test("Ảnh xám 1 kênh màu Grayscale (Tự convert RGB)", r.status_code == 200)

    # Edge Case: Ảnh trong suốt RGBA (4 channels)
    rgba_img = Image.new("RGBA", (300, 150), color=(50, 150, 50, 200))
    buf_rgba = io.BytesIO()
    rgba_img.save(buf_rgba, format="PNG")
    r = client.post("/api/diagnose", files={"file": ("transparent.png", buf_rgba.getvalue(), "image/png")})
    record_test("Ảnh trong suốt 4 kênh RGBA (Tự convert RGB)", r.status_code == 200)

    # Edge Case: Ảnh kích thước bất đối xứng (1920x400)
    rect_img = Image.new("RGB", (1920, 400), color=(34, 139, 34))
    buf_rect = io.BytesIO()
    rect_img.save(buf_rect, format="JPEG")
    r = client.post("/api/diagnose", files={"file": ("rect.jpg", buf_rect.getvalue(), "image/jpeg")})
    record_test("Ảnh tỷ lệ kéo dài 1920x400 (Tự resize 224x224)", r.status_code == 200)

    # 4. Parameter Validation on Query Params
    print("\n--- 4. KIỂM THỬ THAM SỐ NGƯỠNG TIN CẬY (CUSTOM THRESHOLD) ---")
    valid_img_path = SAMPLE_IMAGES_DIR / "tomato_leaf_sample.jpg"
    with open(valid_img_path, "rb") as f:
        valid_img_bytes = f.read()

    # Threshold > 1.0 (Expect 422 Unprocessable Entity do vi phạm le=1.0)
    r = client.post("/api/diagnose?custom_threshold=1.5", files={"file": ("leaf.jpg", valid_img_bytes, "image/jpeg")})
    record_test("Threshold > 1.0 (Expect 422 Unprocessable Entity)", r.status_code == 422)

    # Threshold < 0.0 (Expect 422 Unprocessable Entity do vi phạm ge=0.0)
    r = client.post("/api/diagnose?custom_threshold=-0.5", files={"file": ("leaf.jpg", valid_img_bytes, "image/jpeg")})
    record_test("Threshold < 0.0 (Expect 422 Unprocessable Entity)", r.status_code == 422)

    # Threshold hợp lệ và kiểm tra phản hồi
    r = client.post("/api/diagnose?custom_threshold=0.85", files={"file": ("leaf.jpg", valid_img_bytes, "image/jpeg")})
    diag = r.json()
    record_test("Threshold = 0.85 được áp dụng chính xác", r.status_code == 200 and diag["prediction"]["threshold_applied"] == 0.85)

    # 5. Performance & Latency Benchmark (< 200ms)
    print("\n--- 5. ĐO LƯỜNG ĐỘ TRỄ HIỆU NĂNG (BENCHMARK 30 LƯỢT SUY LUẬN) ---")
    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        r = client.post("/api/diagnose", files={"file": ("leaf.jpg", valid_img_bytes, "image/jpeg")})
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)

    lat_mean = np.mean(latencies)
    lat_median = np.median(latencies)
    lat_p95 = np.percentile(latencies, 95)
    lat_min = np.min(latencies)
    lat_max = np.max(latencies)

    print(f"  + Độ trễ nhỏ nhất (Min): {lat_min:.2f} ms")
    print(f"  + Độ trễ trung bình (Mean): {lat_mean:.2f} ms")
    print(f"  + Độ trễ trung vị (Median): {lat_median:.2f} ms")
    print(f"  + Phân vị 95 (P95): {lat_p95:.2f} ms")
    print(f"  + Độ trễ lớn nhất (Max): {lat_max:.2f} ms")

    benchmark_pass = lat_mean < 200.0 and lat_p95 < 200.0
    record_test("Đạt tiêu chuẩn Capstone latency < 200ms", benchmark_pass, f"(Mean: {lat_mean:.1f}ms, P95: {lat_p95:.1f}ms)")

    # 6. Tổng kết
    print("\n" + "=" * 80)
    print(f" KẾT QUẢ TỔNG THỂ: {passed}/{total} BÀI KIỂM THỬ THÀNH CÔNG ({passed/total*100:.1f}%)")
    if passed == total:
        print(" HỆ THỐNG BACKEND CỦA BẠN HOÀN TOÀN KHÔNG CÓ LỖI (ZERO BUGS)!")
    else:
        print(f" CÓ {total - passed} LỖI CẦN XỬ LÝ.")
    print("=" * 80)

if __name__ == "__main__":
    test_all()
