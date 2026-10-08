# 📘 ĐẶC TẢ GIAO DIỆN API & THIẾT KẾ CƠ SỞ DỮ LIỆU
## AI-Integrated Crop Disease Advisory and Image-Based Diagnosis System
*(Hệ thống tư vấn hỗ trợ bệnh trên cây trồng có tích hợp mô hình AI phân tích bệnh qua hình ảnh)*

* **Chương trình:** Samsung Innovation Campus (SIC) - AI Course Capstone Project
* **Nhóm:** Group 5
* **Tác giả phụ trách:** **Nguyễn Quốc Khánh** (Backend Developer & Database Architect)
* **Giảng viên hướng dẫn & thẩm định:** Thi-Thanh Ha
* **Phiên bản:** 1.0.0
* **Ngày phát hành:** 08/10/2026

---

## 📑 MỤC LỤC
1. [Tổng quan Kiến trúc Hệ thống Backend](#1-tổng-quan-kiến-trúc-hệ-thống-backend)
2. [Thiết kế Cơ sở Dữ liệu Tri thức Nông nghiệp (Knowledge Base Schema)](#2-thiết-kế-cơ-sở-dữ-liệu-tri-thức-nông-nghiệp)
3. [Đặc tả Giao diện RESTful API (API Specification)](#3-đặc-tả-giao-diện-restful-api)
4. [Bộ điều phối Logic Quyết định (Confidence-Gated Decision Engine)](#4-bộ-điều-phối-logic-quyết-định)
5. [Báo cáo Thực nghiệm & Độ trễ (Empirical Benchmark Report)](#5-báo-cáo-thực-nghiệm--độ-trễ)
6. [Hợp đồng Giao tiếp Nội bộ (Internal Integration Contracts)](#6-hợp-đồng-giao-tiếp-nội-bộ)

---

## 1. Tổng quan Kiến trúc Hệ thống Backend

Hệ thống Backend đóng vai trò làm cầu nối trung tâm giữa:
* **Subsystem A (Image Recognition Pipeline):** Tiếp nhận ảnh lá cây từ client, thực hiện kiểm tra tính hợp lệ (Validation), tiền xử lý chuẩn ImageNet (224x224 RGB) và đưa qua mô hình học sâu CNN **MobileNetV2** để trích xuất nhãn bệnh và điểm số tin cậy (Confidence score).
* **Subsystem B (Agricultural Advisory Engine):** Điều phối logic phân ngưỡng tin cậy an toàn, truy vấn cơ sở dữ liệu chuyên gia để trả về phác đồ điều trị, biện pháp canh tác và cảnh báo an toàn lao động.

```text
[ Client (Web/Mobile) ] 
       │ 
       ▼ (HTTP POST Multipart Image)
┌────────────────────────────────────────────────────────────────────────┐
│                        FASTAPI BACKEND SERVICE                         │
│                                                                        │
│  1. Input Validation ──────► 2. Preprocessing ──────► 3. Inference     │
│     (Format, Size, Corrupt)    (Resize 224x224,        (MobileNetV2    │
│                                 ImageNet Norm)          PyTorch/ONNX)  │
│                                                              │         │
│  4. Response Consolidation ◄── 5. Advisory Engine ◄──────────┘         │
│     (JSON: Disease + Advice     (Confidence >= 75%: CONFIRMED          │
│      + Latency + Disclaimer)     Confidence < 75%:  UNCERTAIN)         │
└────────────────────────────────────────────────────────────────────────┘
       ▲
       │ Query Agronomic Protocols
┌──────┴─────────────────────────────────────────────────────────────────┐
│               AGRICULTURAL ADVISORY KNOWLEDGE BASE (JSON)              │
│       11 Bệnh trên Cà chua, Khoai tây, Ngô (Chuẩn FAO & Viện BVTV)     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Thiết kế Cơ sở Dữ liệu Tri thức Nông nghiệp

* **Định dạng lưu trữ:** Structured JSON (`data/advisory_knowledge_base.json`).
* **Tiêu chuẩn tham chiếu:** Cẩm nang Phòng trừ Sâu bệnh hại của **Viện Bảo vệ Thực vật Việt Nam**, tài liệu kỹ thuật của **FAO** và **USDA Extension**.

### 2.1 Từ điển Dữ liệu (Data Dictionary)

| Trường dữ liệu | Kiểu | Bắt buộc | Mô tả |
| :--- | :--- | :---: | :--- |
| `disease_id` | String | Có | Khóa chính định danh duy nhất (VD: `Tomato_Early_blight`) |
| `crop` | String | Có | Tên loài cây trồng (Cà chua, Khoai tây, Ngô) |
| `vietnamese_name` | String | Có | Tên bệnh phổ biến trong sản xuất nông nghiệp tại Việt Nam |
| `english_name` | String | Có | Tên bệnh theo danh pháp tiếng Anh quốc tế |
| `scientific_name` | String | Có | Danh pháp khoa học của cây trồng hoặc mầm bệnh |
| `pathogen_type` | String | Có | Phân loại tác nhân: Nấm (Fungus), Noãn khuẩn (Oomycete), Không có |
| `pathogen_name` | String | Có | Tên loài vi sinh vật gây hại (kèm tên tác giả đặt tên) |
| `visual_symptoms` | String | Có | Triệu chứng trực quan quan sát bằng mắt thường trên phiến lá |
| `environmental_triggers` | String | Có | Điều kiện nhiệt độ (°C), ẩm độ (%), thời tiết kích hoạt dịch bệnh |
| `severity_level` | String | Có | Mức độ nguy hại đối với năng suất và chất lượng mùa vụ |
| `chemical_controls` | Array[Object] | Có | Danh mục các hoạt chất thuốc BVTV hóa học được phép sử dụng |
| `organic_biological_controls`| Array[String] | Có | Biện pháp sinh học (Trichoderma, Bacillus, dầu neem) |
| `cultural_preventions` | Array[String] | Có | Kỹ thuật canh tác phòng ngừa (luân canh, mật độ, dọn tàn dư) |
| `safety_precautions` | Array[String] | Có | Hướng dẫn trang bị bảo hộ (PPE), quy tắc 4 đúng và thời gian cách ly |

### 2.2 Chi tiết Cấu trúc `chemical_controls`

```json
{
  "active_ingredient": "Azoxystrobin + Difenoconazole",
  "trade_names": ["Amistar Top 325SC", "Score 250EC"],
  "dosage": "15 - 20 ml/bình 16 lít nước",
  "pre_harvest_interval_days": 7,
  "application_notes": "Phun ướt đều tán lá khi chớm xuất hiện 5% vết bệnh..."
}
```

---

## 3. Đặc tả Giao diện RESTful API

* **Giao thức:** HTTP/1.1 hoặc HTTP/2 over TLS.
* **Định dạng dữ liệu:** JSON (`application/json`) và Multipart Form (`multipart/form-data`).
* **CORS:** Đã kích hoạt cho tất cả domain (`Access-Control-Allow-Origin: *`).

### 3.1 Bảng Tổng hợp Endpoints

| Phương thức | Đường dẫn | Chức năng | Định dạng Input |
| :---: | :--- | :--- | :--- |
| `GET` | `/` | Thông tin chào mừng & điều hướng tài liệu | None |
| `GET` | `/api/health` | Kiểm tra trạng thái máy chủ & mô hình | None |
| `GET` | `/api/crops` | Lấy danh mục các loại cây trồng hỗ trợ | None |
| `GET` | `/api/diseases` | Danh mục tóm tắt tất cả các bệnh | None |
| `GET` | `/api/diseases/{disease_id}` | Xem chi tiết phác đồ của 1 bệnh cụ thể | Path Param |
| `POST`| `/api/diagnose` | **Endpoint chính: Chẩn đoán ảnh & tư vấn** | `multipart/form-data` |

---

### 3.2 Chi tiết Endpoint Chính: `POST /api/diagnose`

#### Request:
* **Headers:** `Content-Type: multipart/form-data`
* **Form Data:**
  * `file`: File ảnh lá cây (Định dạng cho phép: `.jpg`, `.jpeg`, `.png`, tối đa 10 MB).
* **Query Parameters (Tùy chọn):**
  * `custom_threshold`: Float trong khoảng `[0.0, 1.0]`. Dùng để ghi đè ngưỡng cắt tin cậy (Mặc định: `0.75`).

#### Response Thành công (`HTTP 200 OK` - Trường hợp Confidence $\ge$ 75%):
```json
{
  "diagnosis_status": "CONFIRMED",
  "message": "Độ tin cậy đạt chuẩn. Đã trích xuất phác đồ tư vấn nông nghiệp chính xác.",
  "prediction": {
    "disease_id": "Tomato_Early_blight",
    "crop": "Tomato",
    "is_healthy": false,
    "confidence": 0.9412,
    "confidence_percentage": "94.12%",
    "threshold_applied": 0.75,
    "confidence_gate_passed": true,
    "latency_ms": 34.25
  },
  "advisory_details": {
    "disease_id": "Tomato_Early_blight",
    "vietnamese_name": "Bệnh đốm vòng / Cháy lá sớm cà chua",
    "scientific_name": "Alternaria solani",
    "visual_symptoms": "Xuất hiện các đốm nâu đen hình tròn có vòng đồng tâm...",
    "chemical_controls": [ ... ],
    "organic_biological_controls": [ ... ],
    "cultural_preventions": [ ... ],
    "safety_precautions": [ ... ]
  },
  "remediation_notice": null,
  "regulatory_disclaimer": "Thông tin tư vấn mang tính chất hỗ trợ quyết định...",
  "image_metadata": {
    "filename": "leaf_photo.jpg",
    "content_type": "image/jpeg",
    "size_bytes": 284105
  }
}
```

#### Response Thành công (`HTTP 200 OK` - Trường hợp Confidence $<$ 75% - Fallback an toàn):
```json
{
  "diagnosis_status": "UNCERTAIN",
  "message": "Độ tin cậy mô hình (61.3%) thấp hơn ngưỡng an toàn (75%). Hệ thống tạm giữ phác đồ hóa chất để tránh can thiệp sai lầm.",
  "prediction": {
    "disease_id": "Tomato_Early_blight",
    "confidence": 0.613,
    "confidence_percentage": "61.30%",
    "threshold_applied": 0.75,
    "confidence_gate_passed": false,
    "latency_ms": 33.1
  },
  "advisory_details": {
    "preliminary_assessment": "Bệnh đốm vòng / Cháy lá sớm cà chua",
    "guidance_for_farmer": [
      "1. Chụp lại ảnh lá cây ở cự ly gần (15 - 30cm), lấy nét trực diện vào vết bệnh.",
      "2. Chụp trong điều kiện ánh sáng tự nhiên đầy đủ, tránh bóng râm.",
      "3. Đảm bảo bề mặt lá sạch, không rung mờ khi chụp.",
      "4. Liên hệ trực tiếp với cán bộ khuyến nông địa phương để kiểm tra mẫu lá."
    ]
  },
  "remediation_notice": "Khuyến nghị chụp lại ảnh chất lượng cao hơn hoặc tham vấn chuyên gia."
}
```

#### Các mã phản hồi lỗi (Error Codes):
* **`400 Bad Request`:** File rỗng 0 bytes, dung lượng vượt quá 10MB, định dạng file không được hỗ trợ, hoặc file ảnh bị hỏng không đọc được.
* **`404 Not Found`:** Tra cứu mã bệnh không tồn tại trong hệ thống.
* **`422 Unprocessable Entity`:** Tham số truy vấn `custom_threshold` nằm ngoài dải `[0.0, 1.0]`.
* **`500 Internal Server Error`:** Lỗi xử lý ngoại lệ nội bộ máy chủ.

---

## 4. Bộ điều phối Logic Quyết định (Confidence-Gated Decision Engine)

Nhằm tuân thủ chỉ đạo của giảng viên Thi-Thanh Ha về tính chính xác và an toàn nông nghiệp:
1. **Ngưỡng an toàn (Threshold = 0.75):** Ngăn chặn rủi ro người nông dân phun nhầm thuốc trừ nấm đắt tiền hoặc độc hại khi ảnh chụp chưa đủ thông tin tin cậy.
2. **Kích hoạt Fallback:** Nếu ảnh mờ, ngược sáng hoặc sai góc chụp dẫn đến Softmax Probability < 0.75, hệ thống không xuất tên thương mại thuốc hóa học mà chuyển sang cơ chế hướng dẫn người nông dân chụp lại ảnh đúng quy cách.

---

## 5. Báo cáo Thực nghiệm & Độ trễ (Empirical Benchmark Report)

Thực hiện kiểm thử thực nghiệm trên 30 lượt suy luận liên tiếp với ảnh $224 \times 224$ chuẩn:

| Chỉ số đo lường | Mục tiêu đề tài Capstone | Kết quả thực tế đạt được | Đánh giá |
| :--- | :---: | :---: | :---: |
| **Độ trễ nhỏ nhất (Min Latency)** | — | **31.29 ms** | Siêu nhanh |
| **Độ trễ trung bình (Mean Latency)** | **< 200 ms** | **34.73 ms** | **Vượt chuẩn 5.7 lần** |
| **Độ trễ trung vị (Median Latency)** | — | **34.75 ms** | Rất ổn định |
| **Phân vị 95 (P95 Latency)** | **< 200 ms** | **36.73 ms** | **Vượt chuẩn 5.4 lần** |
| **Độ trễ lớn nhất (Max Latency)** | — | **36.97 ms** | Tối ưu tuyệt đối |
| **Tỷ lệ vượt qua Test Suite** | 100% | **17/17 (100%)** | Hoàn hảo |

> **Thông tin cho Đinh Gia Bảo (Team Leader):** Dữ liệu thực nghiệm trên có thể sử dụng trực tiếp để cập nhật mục *Abstract* và chương *System Performance* trong Báo cáo Đồ án cuối kỳ.

---

## 6. Hợp đồng Giao tiếp Nội bộ (Internal Integration Contracts)

### 6.1 Giao diện với Đinh Gia Bảo (ML Lead)
* **Vị trí lưu trữ trọng số mô hình:** `models/mobilenetv2_crop_disease.pt` (hoặc định dạng `.onnx`).
* **Định dạng đầu vào mô hình:** Tensor kích thước `(1, 3, 224, 224)`, Float32, chuẩn hóa ImageNet Mean `[0.485, 0.456, 0.406]` và Std `[0.229, 0.224, 0.225]`.
* **Đầu ra mong đợi:** Tensor kích thước `(1, 11)` tương ứng với 11 lớp bệnh theo đúng thứ tự mảng `CLASS_NAMES` trong `backend/config.py`.

### 6.2 Giao diện với Ma Hồng Hải (Frontend Lead)
* **Base URL Backend:** `http://localhost:8000`
* **Mẫu code gọi API bằng JavaScript (`fetch`):**

```javascript
async function diagnoseCropLeaf(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  try {
    const response = await fetch("http://localhost:8000/api/diagnose", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || "Lỗi xử lý chẩn đoán");
    }

    const data = await response.json();
    console.log("Kết quả chẩn đoán:", data);
    return data;
  } catch (error) {
    console.error("Lỗi kết nối Backend:", error);
    alert(error.message);
  }
}
```
