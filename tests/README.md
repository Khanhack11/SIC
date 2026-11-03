# Kiểm thử demo

Toàn bộ mã test nằm trong thư mục này. Chạy từ thư mục project:

```powershell
.\.venv\Scripts\python.exe tests/run_all.py
```

Lệnh trên chạy unit/integration, backend và stress/edge. Log từng bộ test cùng `summary.json` nằm trong `tests/results/<thời điểm chạy>/`. Lệnh trả exit code khác 0 nếu có bộ test lỗi.

Để chạy cả browser cần Node.js, package `playwright` và Google Chrome:

```powershell
npm --prefix tests install
.\.venv\Scripts\python.exe tests/run_all.py --web
```

Có thể mở `run_tests.cmd` để chạy cả browser và giữ cửa sổ xem kết quả. Nếu Chrome cài ở vị trí khác, đặt biến `CHROME_PATH` trỏ đến file thực thi. Nếu Playwright đã có ở thư mục dùng chung, đặt `NODE_PATH` trỏ đến thư mục chứa module đó. Không cần tải Chromium khi dùng Chrome sẵn có.

| File | Nội dung |
|---|---|
| `test_*.py` | Unit/integration; `test_backend.py` có thêm 10 kiểm tra khi chạy trực tiếp |
| `stress_and_edge_test.py` | 17 tình huống edge/concurrency và latency |
| `frontend_demo.cjs` | 19 kiểm tra Chrome: camera mô phỏng, ảnh lỗi, disease/healthy/uncertain, responsive, lỗi mạng/API, retry và chọn ảnh khi tải chậm |
| `run_all.py` | Chạy các bộ test, giữ log và mã thoát |
| `fixtures/` | Ảnh nhân tạo và file lỗi để kiểm tra API; không dùng đo chất lượng model |

Browser tự chạy server tại port 8001, đóng server và Chrome khi xong. Giữ port này trống khi test. Screenshot, kết quả browser và log được lưu cùng thư mục kết quả. Test không sửa model phục vụ. Camera dùng thiết bị mô phỏng, chưa xác nhận phần cứng camera thật.
