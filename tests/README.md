# Kiểm thử demo

Toàn bộ mã test nằm trong thư mục này. Chạy từ thư mục project:

```powershell
.\.venv\Scripts\python.exe tests/run_all.py
```

Lệnh trên chạy unit/integration, backend và stress/edge. Log từng bộ test cùng `summary.json` nằm trong `tests/results/<thời điểm chạy>/`. Lệnh trả exit code khác 0 nếu có bộ test lỗi.

Để chạy cả browser cần Node.js, package `playwright` và Google Chrome:

```powershell
npm --prefix tests install
.\tests\node_modules\.bin\playwright.cmd install webkit
.\.venv\Scripts\python.exe tests/run_all.py --web --mobile
```

Có thể mở `run_tests.cmd` để chạy cả browser và giữ cửa sổ xem kết quả. Nếu Chrome cài ở vị trí khác, đặt biến `CHROME_PATH` trỏ đến file thực thi. Nếu Playwright đã có ở thư mục dùng chung, đặt `NODE_PATH` trỏ đến thư mục chứa module đó. Không cần tải Chromium khi dùng Chrome sẵn có.

| File | Nội dung |
|---|---|
| `test_*.py` | Unit/integration; `test_backend.py` có thêm 10 kiểm tra khi chạy trực tiếp |
| `stress_and_edge_test.py` | 17 tình huống edge/concurrency và latency |
| `frontend_demo.cjs` | 24 kiểm tra Chrome, gồm camera mô phỏng, ảnh lỗi, disease/healthy/uncertain, lỗi mạng/API, retry, crop/khôi phục và pixels preview/upload |
| `mobile_browsers.cjs` | Chromium và WebKit với thao tác chạm và viewport 360/390/412/844; không phải điện thoại thật |
| `field_demo.py` | Chạy riêng bằng Python khi có weights V3: kiểm tra API thật, crop và đo latency CPU sau warmup |
| `architecture_diagrams.cjs` | Kiểm tra riêng file HTML Archify bằng Chrome ở 1440/1920 px; truyền thư mục sơ đồ làm đối số |
| `run_all.py` | Chạy các bộ test, giữ log và mã thoát |
| `fixtures/` | Ảnh nhân tạo và file lỗi để kiểm tra API; không dùng đo chất lượng model |

Browser tự chạy server tại port 8001, đóng server và Chrome khi xong. Giữ port này trống khi test. Screenshot, kết quả browser và log được lưu cùng thư mục kết quả. Test không sửa model phục vụ. Camera dùng thiết bị mô phỏng, chưa xác nhận phần cứng camera thật.
