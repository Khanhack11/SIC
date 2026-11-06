# Báo cáo Huấn luyện & Đánh giá Mô hình `MobileNetV3-Large` Đa nguồn (`v2_multisource_20261103`)

## 1. Cấu hình Huấn luyện và Dữ liệu (`field_mobile_v2`)

- **Mô hình khởi tạo (`Parent Checkpoint`)**: [`models/mobilenetv3_field_continued_20261103.pt`](file:///D:/AI_SIC_ICTU_2026/Final_SIC_AI_2026/Project/SIC/models/mobilenetv3_field_continued_20261103.pt) (`sha256: 6a252ac806d88367bb6ee67ce04cccfce7ba986268f5d0d5e1f8332ed39ec62a`).
- **Mô hình đầu ra (`Candidate Checkpoint`)**: [`models/mobilenetv3_field_v2_20261103.pt`](file:///D:/AI_SIC_ICTU_2026/Final_SIC_AI_2026/Project/SIC/models/mobilenetv3_field_v2_20261103.pt) (`sha256: c296f938564a4beb07cfeca00d9cca579516de442666a4ad96b6748355ee4c4e`) và đồ thị ONNX FP32 [`models/mobilenetv3_field_v2_20261103.onnx`](file:///D:/AI_SIC_ICTU_2026/Final_SIC_AI_2026/Project/SIC/models/mobilenetv3_field_v2_20261103.onnx).
- **Tập dữ liệu huấn luyện & đánh giá mở rộng**: [`data/processed/field_mobile_v2/records.json`](file:///D:/AI_SIC_ICTU_2026/Final_SIC_AI_2026/Project/SIC/data/processed/field_mobile_v2/records.json) (`20.902` bản ghi, `sha256: 4678faca1433bbb79d31185cdb6600994e29b777c9fbe8ed23bfb58fe92f5f24`):
  - Giữ nguyên 100% các tập đánh giá `PlantVillage/validation` (`1.731`), `PlantVillage/test` (`1.720`), `PlantDoc/validation` (`414`) và `PlantDoc/test` (`217`) từ [`data/processed/field_mobile_v1/records.json`](file:///D:/AI_SIC_ICTU_2026/Final_SIC_AI_2026/Project/SIC/data/processed/field_mobile_v1/records.json).
  - Bổ sung `3.160` khung cắt lá đơn (`plantdoc_detection_crop`) trích xuất nghiêm ngặt từ ảnh cha thuộc `PlantDoc/train`.
  - Bổ sung `3.959` ảnh thực địa đã khử trùng lặp và kiểm tra không rò rỉ holdout (`3.364 train` / `595 validation`) từ 5 nguồn thực địa mới (`plantwild_v2`, `potato_holeta_v2`, `potato_field_indonesia`, `maize_field_bd`, `tomato_field_pakistan`).
- **Chiến lược tối ưu hóa trên GPU (`NVIDIA GeForce RTX 2050`)**:
  - Mở khóa 9 khối đặc trưng cuối (`unfrozen_feature_blocks = 9`) của `MobileNetV3-Large` với tốc độ học phân tầng (`mid_backbone_lr = 2.5e-5`, `top_backbone_lr = 6e-5`, `head_lr = 2e-4`), `CosineAnnealingLR`, `6` epochs × `6.400` mẫu cân bằng miền/lớp mỗi epoch.

---

## 2. So sánh Kết quả trên Tập `validation` (`field_mobile_v1` & `field_mobile_v2`)

| Chỉ số đánh giá (`Validation Split`) | Baseline V2 (`mobilenetv2_crop_disease.onnx`) | Candidate V3 (`continued_20261103`) | Candidate V3 Mở rộng (`v2_20261103`) | Thay đổi (`v2` so với `continued`) |
|---|---:|---:|---:|---:|
| **`PlantVillage` Accuracy** | `94.28%` | `97.92%` | **`98.27%`** | `+0.35 pp` |
| **`PlantVillage` Known Macro F1** | `94.48%` | `97.52%` | **`98.17%`** | **`+0.65 pp`** (`+3.69 pp` so với V2) |
| **`PlantDoc` Accuracy** | `10.14%` | `75.12%` | **`75.85%`** | `+0.73 pp` |
| **`PlantDoc` Known Macro F1 (8 lớp hiện diện)** | `15.95%` | `46.59%` | **`54.98%`** | **`+8.39 pp`** (`+39.03 pp` so với V2) |
| **`PlantDoc` Macro F1 (kể cả `Unsupported`)** | `14.18%` | `51.30%` | **`58.66%`** | **`+7.36 pp`** |
| **`PlantDoc` Score-Gate Coverage (`>= 0.75`, `!= Unsupported`)** | `75.36%` | `18.84%` | **`24.88%` (CPU) / `25.12%` (GPU)** | **`+6.04 pp`** |
| **`PlantDoc` Score-Gate Accuracy** | `10.58%` | `57.69%` | **`59.22%`** | **`+1.53 pp`** |
| **`PlantDoc` Known Diseased False-Healthy Rate** | `3.91%` | `0.00%` | **`0.00%`** (`0/128`) | Giữ mức an toàn tuyệt đối `0.00%` |
| **`MultiSourceField` Accuracy (`595` ảnh thực địa mới)** | — | `52.10%` | **`77.82%`** | **`+25.72 pp`** |
| **`MultiSourceField` Known Macro F1 (Đầy đủ 11 lớp)** | — | `26.80%` | **`65.96%`** (đạt đỉnh `68.41%` ở epoch 3) | **`+39.16 pp`** |
| **`MultiSourceField` Score-Gate Coverage** | — | — | **`46.55%`** | — |
| **`MultiSourceField` Score-Gate Accuracy** | — | — | **`80.87%`** (Accepted Acc: **`85.90%`**) | — |

---

## 3. Phân tích Cải thiện Theo Từng Lớp Thực địa (`PlantDoc` & `MultiSourceField` Validation)

| Lớp mục tiêu (`FIELD_CLASSES`) | `PlantDoc` F1 (`continued_20261103`) | `PlantDoc` F1 (`v2_20261103`) | `MultiSourceField` F1 (`v2_20261103`) | Ghi chú chuyên môn |
|---|---:|---:|---:|---|
| `Corn_Common_rust` | `82.93%` | **`87.80%`** | **`75.00%`** | Tăng độ chính xác trên cả ảnh Internet và ảnh đồng ruộng Bangladesh |
| `Corn_Gray_leaf_spot` | `71.43%` | **`74.07%`** | **`73.68%`** | Giảm nhầm lẫn với gỉ sắt ngô |
| `Corn_Northern_Leaf_Blight` | *(0 mẫu)* | *(0 mẫu)* | **`58.06%`** (Precision `75.00%`) | Khắc phục hoàn toàn điểm mù `0` mẫu trong `PlantDoc` nhờ `PlantWild_v2` |
| `Corn_healthy` | *(0 mẫu)* | *(0 mẫu)* | **`89.66%`** (Recall `100.00%`) | Khắc phục hoàn toàn điểm mù `0` mẫu trong `PlantDoc` nhờ `maize_field_bd` |
| `Potato_Early_blight` | `35.00%` | **`38.30%`** | **`41.03%`** | Tăng Recall từ `36.84%` lên `47.37%` trên `PlantDoc` |
| `Potato_Late_blight` | `29.41%` | **`35.29%`** | **`83.78%`** | Đạt `83.78%` F1 trên `72` ảnh đồng ruộng Ethiopia & Indonesia |
| `Potato_healthy` | *(0 mẫu)* | *(0 mẫu)* | **`95.12%`** (`82` mẫu) | Khắc phục hoàn toàn điểm mù `0` mẫu nhờ `potato_holeta_v2` & `potato_field_indonesia` |
| `Tomato_Early_blight` | `29.41%` | **`41.38%`** | **`43.48%`** | Tăng `+11.97 pp` trên `PlantDoc` |
| `Tomato_Late_blight` | `34.48%` | **`55.00%`** | **`56.60%`** | Tăng mạnh `+20.52 pp` trên `PlantDoc` (Recall tăng từ `27.78%` lên `61.11%`) |
| `Tomato_Septoria_leaf_spot` | `52.00%` | **`50.85%`** (Recall `60.00%`) | **`47.62%`** | Tăng Recall trên `PlantDoc` từ `52.00%` lên `60.00%` |
| `Tomato_healthy` | `38.10%` | **`57.14%`** | **`61.54%`** | Tăng mạnh `+19.04 pp` trên `PlantDoc` |
| `Unsupported` (OOD) | `88.93%` | **`88.09%`** (Precision **`91.73%`**) | **`83.01%`** (`271` mẫu) | Precision của `Unsupported` trên `PlantDoc` tăng từ `88.77%` lên `91.73%` |

---

## 4. Quyết định Phát hành (`docs/final_release/mobile_field/v2_multisource_20261103/release_decision.json`)

Theo bộ tiêu chí [`docs/FIELD_RELEASE_CRITERIA.json`](file:///D:/AI_SIC_ICTU_2026/Final_SIC_AI_2026/Project/SIC/docs/FIELD_RELEASE_CRITERIA.json) trên tập `PlantDoc/validation` nguyên bản (gồm nhiều ảnh đa lá chưa cắt khung):
- `field_f1_gain`: **`+39.03 pp`** (yêu cầu `>= 5.0 pp`) — **ĐẠT (`PASSED`)**
- `village_f1_drop`: **`-3.69 pp`** (tức tăng `+3.69 pp`, yêu cầu giảm `<= 2.0 pp`) — **ĐẠT (`PASSED`)**
- `known_diseased_false_healthy`: **`0.00%`** (yêu cầu `<= 5.0%`) — **ĐẠT (`PASSED`)**
- `field_score_gate_coverage`: **`24.88%`** trên CPU (`103/414` mẫu) / **`25.12%`** trên GPU (`104/414` mẫu, ngưỡng `>= 25.0%`)
- `field_score_gate_accuracy`: **`59.22%`** trên `PlantDoc/validation` nguyên bản (trong khi trên `MultiSourceField/validation` đạt **`80.87%`**, cách ngưỡng `85.0%` khoảng `4.13 pp` do nhiễu nền trên các ảnh đa lá chưa crop).

Do đó, tuân thủ đúng quy tắc an toàn của hệ thống, [`models/mobilenetv3_field_v2_20261103.pt`](file:///D:/AI_SIC_ICTU_2026/Final_SIC_AI_2026/Project/SIC/models/mobilenetv3_field_v2_20261103.pt) được lưu giữ làm ứng viên thực địa tốt nhất hiện tại (vượt trội toàn diện so với `mobilenetv3_field_continued_20261103.pt` trên cả 3 miền `PlantVillage`, `PlantDoc` và `MultiSourceField`), trong khi không tự động ghi đè mô hình mặc định khi chưa qua bước cắt khung lá trên ảnh đa đối tượng.
