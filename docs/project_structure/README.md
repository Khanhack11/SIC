# Kiểm kê và sắp xếp hệ thống ngày 03/11/2026

Đợt này tập trung vào cấu trúc thư mục, các đường dẫn mà API và công cụ đang sử dụng, cùng khả năng chạy lại demo sau khi chuyển file. Thống kê bao gồm file ẩn, môi trường Python, dữ liệu, trọng số, Git và file sinh ra. Không đi theo symlink/junction.

## Cấu trúc hiện tại

```text
SIC/
  frontend/           HTML, frontend.js và crop.js
  backend/            API, inference và kho tư vấn
  ml/                 Chuẩn bị dữ liệu, train, đánh giá và export
  data/               Kho tri thức, raw và processed
  models/             Trọng số và trạng thái thí nghiệm
  tests/              Toàn bộ kiểm thử và tests/results
  docs/               Báo cáo và bằng chứng có phiên bản
  notebooks/          Notebook phát triển ban đầu
  scripts/            Công cụ kiểm kê hệ thống
  sample_images/      Ảnh minh họa và fixture
  artifacts/          Cache, log, công cụ local và packages
  .archify/           Sơ đồ cùng bằng chứng nguồn đã ghim revision
  .venv/              Môi trường phục vụ và kiểm thử
  .venv-gpu/          Môi trường CUDA để train
  run_demo.cmd        Mở server baseline
  run_field_demo.cmd  Mở server V3 thử nghiệm
  run_mobile_demo.cmd Server truy cập trong mạng nội bộ
```

## Những thay đổi đã thực hiện

| Trước | Sau |
|---|---|
| frontend_demo.html | frontend/frontend_demo.html |
| frontend.js | frontend/frontend.js |
| crop.js | frontend/crop.js |
| backend_development.ipynb | notebooks/backend_development.ipynb |
| Bốn thư mục .chart-data-* ở gốc | artifacts/visualizations/archive_20261103/ |

Backend xác định vị trí frontend bằng `FRONTEND_DIR`. Các URL `/app`, `/frontend_demo.html`, `/frontend.js`, `/crop.js` vẫn hoạt động. Bộ test camera mô phỏng đọc file từ thư mục mới. Công cụ đóng gói lấy frontend, notebook, công cụ kiểm kê và các tài liệu mới; đầu ra mặc định nằm trong `artifacts/packages/`.

Notebook tự tìm thư mục gốc khi mở từ SIC hoặc notebooks. Các ô còn lại là ví dụ phát triển ban đầu, có payload cũ; chưa chạy lại toàn bộ notebook. Dùng API và bộ test hiện tại để kiểm tra hành vi phục vụ.

## Phân biệt các nhóm dữ liệu và kết quả

- Dữ liệu bổ sung nằm trong `data/raw/dataset_hub_20261103`, có báo cáo và danh sách review riêng. Snapshot cũ được giữ tại chỗ vì manifest đã lưu đường dẫn tuyệt đối và hash.
- `models/README.md` phân biệt baseline đang phục vụ, V3 thử nghiệm, thí nghiệm V2/CNN và file phục hồi. Không chuyển hoặc đổi tên trọng số để giữ đúng hợp đồng checkpoint/ONNX.
- `docs/README.md` là mục lục tài liệu. Các bản trong final_release, final_report và .archify được giữ làm hồ sơ của từng thời điểm; liên kết nguồn Archify vẫn trỏ revision cũ hợp lệ.
- Các script phát sinh cũ trong artifacts có đường dẫn phụ thuộc thư mục hiện tại; được lập danh mục và giữ nguyên để tái hiện lịch sử. Cache biểu đồ ở gốc đã được gom riêng.

## Những việc còn cần làm cho model

639 ảnh ứng viên mới cần duyệt nhãn trước khi tạo manifest. Pipeline train hiện còn gắn với domain PlantVillage và PlantDoc; cần nối manifest đã duyệt và cập nhật đánh giá theo domain khi thêm nguồn. V3 chưa qua tiêu chí phát hành trên ảnh thực tế. Việc sắp xếp thư mục không tạo ra kết quả train mới hay cải thiện độ chính xác.

## Kiểm tra và thống kê

Số liệu trước/sau, danh sách đổi đường dẫn và kết quả kiểm thử được lưu cùng tài liệu này. Danh sách từng file và danh mục mã nguồn đầy đủ nằm ở `artifacts/project_inventory/` trên máy local. Dung lượng là tổng kích thước logic, không phải dung lượng cấp phát vật lý trên ổ đĩa.

Chạy lại kiểm kê:

```powershell
.\.venv\Scripts\python.exe scripts/inventory_project.py --output artifacts/project_inventory/new_snapshot
```

Kiểm kê đọc metadata file và phân tích cú pháp Python, không import mã nguồn được kiểm kê. Kết quả không thay thế đánh giá đầy đủ mọi lỗi nghiệp vụ hoặc kiểm định chuyên môn nhãn bệnh cây.

## Số liệu kiểm kê

Trước sắp xếp: 130041 file, 17.808 GB. Snapshot sau: 130112 file, 17.915 GB. Phần tăng gồm gói ZIP bàn giao, log/screenshot kiểm thử, danh sách file CSV và tài liệu mới. Không xóa dữ liệu hay môi trường Python.

| Nhóm | File sau sắp xếp | GB (thập phân) |
|---|---:|---:|
| data | 60489 | 11.0359 |
| .venv-gpu | 23110 | 4.4519 |
| artifacts | 2364 | 1.1265 |
| .venv | 42856 | 0.9109 |
| models | 41 | 0.2602 |
| tests | 410 | 0.0995 |
| docs | 142 | 0.0133 |
| .git | 540 | 0.0128 |
| .archify | 52 | 0.0033 |
| ml | 63 | 0.0005 |
| sample_images | 11 | 0.0002 |
| backend | 16 | 0.0001 |
| frontend | 4 | 0.0000 |
| notebooks | 2 | 0.0000 |
| scripts | 3 | 0.0000 |
| (root files) | 9 | 0.0000 |

Đã phân tích cú pháp 61 file Python thuộc danh mục mã nguồn; không có lỗi cú pháp và không có lỗi đọc file trong snapshot. Danh mục model có 39 file trọng số, metadata và trạng thái thí nghiệm, kèm SHA-256 trong model_catalog.csv.

## Kết quả chạy kiểm thử

- 50 unit/integration, 10 backend và 17 stress/edge: đạt.
- 25 kiểm tra frontend, 4 hồi quy camera/crop: đạt.
- Chromium và WebKit ở viewport 360/390/412/844: đạt, chưa thử thiết bị vật lý.
- Test riêng công cụ kiểm kê được chạy lại sau khi bổ sung xử lý gói không có Git: đạt.
- Nội dung ba file frontend khớp Git trước khi chuyển (bỏ qua quy ước xuống dòng khi checkout); bootstrap notebook được thử từ thư mục SIC và notebooks.
- Manifest dữ liệu hiện tại vẫn khớp checksum của audit ban đầu.

Gói kiểm tra mới: artifacts/packages/SIC_organized_20261103.zip. Kết quả smoke của gói giải nén nằm trong package_verification.json. Các file thống kê mô tả thời điểm chụp snapshot; lần chạy test hoặc đóng gói tiếp theo có thể tạo thêm file.
