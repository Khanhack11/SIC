# Giao diện web

`frontend_demo.html` chứa bố cục và CSS, `frontend.js` xử lý upload, gọi API và hiển thị kết quả, `crop.js` xử lý vùng chọn ảnh.

Chạy `run_demo.cmd` từ thư mục SIC rồi mở `http://127.0.0.1:8000/app`. FastAPI phục vụ file trong thư mục này qua các URL `/app`, `/frontend_demo.html`, `/frontend.js` và `/crop.js`. Đường dẫn URL được giữ sau khi sắp xếp mã nguồn.

Luồng chính là chọn ảnh có sẵn. Camera nằm trong mục bổ sung. Kiểm thử giao diện nằm ở `tests/frontend_demo.cjs`, `tests/frontend_async.cjs` và `tests/mobile_browsers.cjs`.
