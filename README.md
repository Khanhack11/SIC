# 🌿 AI-Integrated Crop Disease Advisory and Image-Based Diagnosis System
**Hệ thống tư vấn hỗ trợ bệnh trên cây trồng có tích hợp mô hình AI phân tích bệnh qua hình ảnh**

* **Chương trình:** Samsung Innovation Campus (SIC) - AI Course
* **Nhóm thực hiện:** Group 5
* **Thành viên & Phân công:**
  * **Nguyễn Quốc Khánh:** **Backend Developer & Database Architect** (Chịu trách nhiệm toàn bộ hệ thống Backend, API và Cơ sở dữ liệu tri thức)
  * **Đinh Gia Bảo:** Team Leader & ML Engineer (Huấn luyện mạng CNN MobileNetV2)
  * **Trần Đặng Công Tâm:** Data & Optimization Engineer (Tập dữ liệu PlantVillage & Tối ưu hóa suy luận)
  * **Ma Hồng Hải:** Frontend Developer & Agronomic QA (Giao diện Web & Kiểm định nông học)

---

## ⚡ Hướng Dẫn Nhanh: Lấy Code Về Chạy Được Ngay (Quick Start)

Dành cho các thành viên trong nhóm (Bảo, Tâm, Hải) hoặc giảng viên muốn tải về và chạy ngay:

### Bước 1: Clone dự án về máy
```bash
git clone https://github.com/Khanhack11/SIC.git
cd SIC
```

### Bước 2: Cài đặt thư viện phụ thuộc
Khuyến nghị tạo môi trường ảo (virtualenv) hoặc cài trực tiếp:
```bash
pip install -r requirements.txt
```

### Bước 3: Lựa chọn cách chạy

#### 🔹 Lựa chọn A: Mở và chạy trực tiếp trên VS Code bằng Jupyter Notebook (`.ipynb`)
1. Mở thư mục dự án trên **Visual Studio Code**.
2. Mở file `backend_development.ipynb`.
3. Bấm **Select Kernel** $\rightarrow$ chọn Python của bạn.
4. Bấm **Run All** để xem trực quan bảng dữ liệu 11 loại bệnh, biểu đồ phân tích và kết quả test API.

#### 🔹 Lựa chọn B: Khởi động Server Backend (Kết nối Frontend Web của bạn Hải)
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
* **Giao diện test API trực quan (Swagger UI):** Mở trình duyệt truy cập `http://localhost:8000/docs`
* **Endpoint chẩn đoán bệnh:** `POST http://localhost:8000/api/diagnose` (Hỗ trợ upload ảnh lá cây)
* *Đã tích hợp sẵn CORS Middleware (`allow_origins=["*"]`) cho phép mọi ứng dụng Frontend (React, Vue, Vite, HTML5) kết nối tự do mà không bị chặn.*

#### 🔹 Lựa chọn C: Chạy kiểm thử tự động toàn diện (Test Suite)
```bash
python test_backend.py
```
hoặc kiểm thử chuyên sâu toàn bộ edge cases:
```bash
python stress_and_edge_test.py
```
*(Kết quả: 17/17 test cases PASS 100%, độ trễ xử lý ~34ms đạt chuẩn Capstone < 200ms)*.

---

## 📁 Cấu trúc Thư mục Dự án

```text
SIC/
├── backend_development.ipynb         # [CHÍNH] Jupyter Notebook phát triển, kiểm thử & demo trên VS Code
├── test_backend.py                  # Bộ kiểm thử chức năng tự động (10/10 PASS)
├── stress_and_edge_test.py          # Bộ kiểm thử chuyên sâu & trường hợp biên (17/17 PASS)
├── requirements.txt                 # Danh sách thư viện phụ thuộc
├── README.md                        # Hướng dẫn chi tiết sử dụng và tích hợp
│
├── backend/                         # Package mã nguồn Backend hoàn chỉnh
│   ├── __init__.py
│   ├── config.py                    # Cấu hình ngưỡng tin cậy (0.75), kích thước ảnh (224x224), 11 lớp bệnh
│   ├── inference_engine.py          # Tiền xử lý, lọc ảnh hỏng & suy luận MobileNetV2 (Latency < 200ms)
│   ├── advisory_engine.py           # Bộ điều phối logic quyết định (Confidence-Gated Decision Logic)
│   └── main.py                      # RESTful API Service sử dụng FastAPI & chuẩn OpenAPI
│
├── data/
│   └── advisory_knowledge_base.json # Database tri thức nông nghiệp chuẩn FAO, USDA & Viện BVTV Việt Nam
│
├── models/                          # Nơi chứa file trọng số mobilenetv2_crop_disease.pt từ bạn Bảo (ML Lead)
│   └── .gitkeep
│
└── sample_images/                   # Bộ ảnh mẫu phục vụ kiểm thử ngay
    ├── tomato_leaf_sample.jpg       # Ảnh lá cà chua bệnh đốm vòng
    ├── corn_leaf_sample.png         # Ảnh lá ngô bệnh gỉ sắt
    ├── potato_leaf_sample.jpg       # Ảnh lá khoai tây khỏe mạnh
    ├── corrupted_image.jpg          # Ảnh hỏng để test tính năng lọc lỗi
    └── invalid_disguised_file.png   # File giả mạo định dạng
```

---

## 🛡️ Điểm Nổi Bật Đáp Ứng Chuẩn Giảng Viên & Samsung SIC
1. **Confidence-Gated Safeguard:** Ngăn ngừa việc nông dân dùng nhầm thuốc bảo vệ thực vật khi ảnh chụp mờ hoặc không đủ độ tin cậy ($< 75\%$).
2. **Tuân thủ quy chuẩn y tế & nông nghiệp:** Đính kèm thời gian cách ly (PHI), liều lượng, hoạt chất và hướng dẫn trang bị bảo hộ lao động (PPE).
3. **Hiệu năng cao:** Tối ưu hóa pipeline xử lý tensor, độ trễ trung bình đạt ~34ms (vượt xa mục tiêu < 200ms của đề tài).
