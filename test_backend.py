"""
Automated Test Suite for SIC Crop Disease Backend System
Tests all endpoints, inference engine, knowledge base, and validation rules.
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure backend package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.main import app
from backend.config import SAMPLE_IMAGES_DIR, CONFIDENCE_THRESHOLD

client = TestClient(app)

def run_tests():
    print("=" * 70)
    print("BAT DAU CHAY BO KIEM THU TU DONG (SIC BACKEND TEST SUITE)")
    print("=" * 70)

    # 1. Test Root Endpoint
    print("\n[TEST 1] Kiem tra Root Endpoint ('/')...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["backend_lead"] == "Nguyen Quoc Khanh"
    print(f"  -> PASS: Status {res.status_code}, Lead: {data['backend_lead']}")

    # 2. Test Health Check Endpoint
    print("\n[TEST 2] Kiem tra Health Check ('/api/health')...")
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "HEALTHY"
    print(f"  -> PASS: Status {data['status']}, Classes: {data['num_supported_classes']}, Latency goal: {data['target_latency']}")

    # 3. Test Supported Crops
    print("\n[TEST 3] Kiem tra danh muc cay trong ('/api/crops')...")
    res = client.get("/api/crops")
    assert res.status_code == 200
    crops = res.json()["supported_crops"]
    assert len(crops) == 3  # Tomato, Potato, Corn
    print(f"  -> PASS: Ho tro {len(crops)} loai cay: {[c['crop_name'] for c in crops]}")

    # 4. Test Disease Catalog
    print("\n[TEST 4] Kiem tra danh muc benh ('/api/diseases')...")
    res = client.get("/api/diseases")
    assert res.status_code == 200
    diseases = res.json()["diseases"]
    assert len(diseases) >= 11
    print(f"  -> PASS: Tong so benh trong Knowledge Base: {len(diseases)}")

    # 5. Test Disease Detail by ID
    print("\n[TEST 5] Kiem tra chi tiet benh ('/api/diseases/Tomato_Early_blight')...")
    res = client.get("/api/diseases/Tomato_Early_blight")
    assert res.status_code == 200
    detail = res.json()["advisory_data"]
    assert detail["scientific_name"] == "Alternaria solani"
    assert len(detail["chemical_controls"]) > 0
    print(f"  -> PASS: Lay thanh cong phac do '{detail['vietnamese_name']}', tac nhan: {detail['scientific_name']}")

    # 6. Test Disease Detail Not Found
    print("\n[TEST 6] Kiem tra ma benh khong ton tai (404 Not Found)...")
    res = client.get("/api/diseases/UNKNOWN_DISEASE_XYZ")
    assert res.status_code == 404
    print(f"  -> PASS: Tra ve dung 404 Not Found khi benh khong ton tai")

    # 7. Test Diagnose with Valid Image (Tomato Leaf)
    print("\n[TEST 7] Kiem tra chan doan anh hop le ('/api/diagnose')...")
    tomato_img_path = SAMPLE_IMAGES_DIR / "tomato_leaf_sample.jpg"
    with open(tomato_img_path, "rb") as f:
        res = client.post(
            "/api/diagnose",
            files={"file": ("tomato_leaf_sample.jpg", f, "image/jpeg")}
        )
    assert res.status_code == 200
    diag = res.json()
    assert "prediction" in diag
    assert "latency_ms" in diag["prediction"]
    latency = diag["prediction"]["latency_ms"]
    print(f"  -> PASS: Chan doan thanh cong!")
    print(f"     + Status: {diag['diagnosis_status']}")
    print(f"     + Predicted: {diag['prediction']['disease_id']}")
    print(f"     + Confidence: {diag['prediction']['confidence_percentage']}")
    print(f"     + Latency: {latency} ms (Dat chuan < 200ms: {latency < 200})")

    # 8. Test Diagnose with Low Confidence Override
    print("\n[TEST 8] Kiem tra bo loc Confidence-Gated (Threshold cut-off)...")
    # Set threshold to 0.99 so prediction falls below threshold -> UNCERTAIN
    with open(tomato_img_path, "rb") as f:
        res = client.post(
            "/api/diagnose?custom_threshold=0.999",
            files={"file": ("tomato_leaf_sample.jpg", f, "image/jpeg")}
        )
    assert res.status_code == 200
    diag = res.json()
    assert diag["diagnosis_status"] == "UNCERTAIN"
    assert "remediation_notice" in diag
    print(f"  -> PASS: Bo loc hoat dong chinh xac khi confidence < threshold!")
    print(f"     + Status: {diag['diagnosis_status']}")
    print(f"     + Warning: {diag['remediation_notice']}")

    # 9. Test Corrupted Image Rejection
    print("\n[TEST 9] Kiem tra loc anh loi/hong (400 Bad Request)...")
    corrupt_img_path = SAMPLE_IMAGES_DIR / "corrupted_image.jpg"
    with open(corrupt_img_path, "rb") as f:
        res = client.post(
            "/api/diagnose",
            files={"file": ("corrupted_image.jpg", f, "image/jpeg")}
        )
    assert res.status_code == 400
    print(f"  -> PASS: Da chan dung file anh hong voi ma loi 400: {res.json()['detail']}")

    # 10. Test Fake PNG Rejection
    print("\n[TEST 10] Kiem tra loc file sai dinh dang gia mao PNG (400 Bad Request)...")
    fake_png_path = SAMPLE_IMAGES_DIR / "invalid_disguised_file.png"
    with open(fake_png_path, "rb") as f:
        res = client.post(
            "/api/diagnose",
            files={"file": ("invalid_disguised_file.png", f, "image/png")}
        )
    assert res.status_code == 400
    print(f"  -> PASS: Da phat hien va chan file gia mao thanh cong")

    print("\n" + "=" * 70)
    print("TAT CA 10/10 BAI KIEM THU DA VUOT QUA XUAT SAC (100% PASS)!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
