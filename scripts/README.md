# Công cụ quản lý project

Chạy từ thư mục SIC:

```powershell
.\.venv\Scripts\python.exe scripts/inventory_project.py --output artifacts/project_inventory/new_snapshot
```

Kết quả gồm thống kê dung lượng, danh sách từng file CSV và danh mục mã nguồn JSON. Công cụ không import hay chạy các module được kiểm kê, không đi theo symlink/junction và không sửa dữ liệu/model.

Mỗi lần chạy dùng thư mục kết quả riêng. Bản trước và sau đợt sắp xếp 03/11 nằm trong `artifacts/project_inventory/`.
