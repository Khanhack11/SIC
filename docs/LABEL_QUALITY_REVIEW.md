# Kiểm tra chất lượng nhãn hiện tại

Audit ngày 03/11/2026 trên manifest `field_mobile_v2`, SHA-256 `4678faca1433bbb79d31185cdb6600994e29b777c9fbe8ed23bfb58fe92f5f24`. Đây là kiểm tra metadata; chưa phải người có chuyên môn đọc ảnh và xác nhận bệnh.

| Phát hiện trong 20.902 record | Số lượng | Cách hiểu |
|---|---:|---|
| Không ghi người duyệt của project | 20.902 | Không có bằng chứng duyệt của nhóm trong manifest; không phủ nhận việc nguồn gốc có thể đã được chuyên gia gán nhãn |
| Không có original_label riêng | 11.494 | Chủ yếu PlantVillage; tên thư mục vẫn có trong path để truy lại |
| Nhãn tên lá chung được ánh xạ sang healthy | 176 | Cần kiểm chứng ý nghĩa annotation gốc và xem ảnh; chưa kết luận 176 nhãn đều sai |
| Crop thiếu hash ảnh cha hoặc tọa độ ngay trong record | 3.160 | Có group và quy trình tạo crop, nhưng dấu vết trong từng record chưa đầy đủ |
| Ánh xạ corn rust / Potato Phytophthora cần duyệt thuật ngữ | 458 | Tên gọi nguồn chưa tự chứng minh tác nhân/bệnh đích tương ứng |
| Unsupported | 5.633 | Gộp nhiều loại ảnh ngoài phạm vi; chưa phải bộ phát hiện mọi ảnh lạ |

Các nhóm có thể chồng lấp, không cộng chúng thành số ảnh sai. Một nhãn/ảnh phù hợp với bài toán phân loại đơn nhãn; mức chi tiết phải theo mục tiêu. Tự crop cần vị trí lá; chỉ ra vết bệnh cần annotation vùng tổn thương. Thêm cột chưa làm bệnh được xác nhận đúng.

## Hàng đợi duyệt đã tạo

`artifacts/label_review/quality_audit_20261103/train_review_queue.csv`: 16.225 ảnh train, giữ path, SHA, nhãn cũ, nguồn, group và lý do cần xem lại. Nhãn đề xuất, triệu chứng, reviewer và bằng chứng đều để trống.

| Ưu tiên | Ảnh | Căn cứ xếp hàng |
|---|---:|---|
| 1 | 549 | Tên lá chung ánh xạ healthy hoặc thuật ngữ bệnh cần xác nhận |
| 2 | 3.642 | Healthy ngoài PlantVillage hoặc crop, sau khi loại ưu tiên 1 |
| 3 | 3.991 | Các ảnh train ngoài PlantVillage còn lại |
| 4 | 8.043 | PlantVillage còn lại |

Ưu tiên là heuristic kiểm tra metadata, không phải xác suất nhãn sai và không dựa trên việc model đồng ý với nhãn. Summary JSON cùng thư mục ghi số lượng và SHA manifest. CSV này chỉ là biểu mẫu hàng đợi, không có UI gán nhãn hoặc chức năng tự áp dụng sửa nhãn.

## Quy trình cho phiên bản nhãn tiếp theo

1. **Định nghĩa lớp:** ghi triệu chứng được chấp nhận, trường hợp dễ nhầm, điều kiện loại ảnh và nghĩa của healthy. Healthy theo ảnh chỉ thể hiện không quan sát thấy triệu chứng trong ảnh, không bảo đảm cả cây không có bệnh.
2. **Duyệt ảnh và nguồn:** người duyệt ghi observed_symptoms, image_quality, multiple_leaves và multiple_disorders. Nhãn nguồn được giữ để truy nguyên. Không chép dự đoán model thành nhãn thật.
3. **Ghi quyết định:** review_decision chọn accept, relabel, exclude hoặc needs_expert; proposed_label ghi đề xuất có căn cứ; reviewer, review_date và evidence ghi người duyệt, thời điểm, căn cứ. Chuyên gia phân xử các trường hợp tranh chấp/không rõ; ảnh chưa xác định không tự trở thành Unsupported hoặc healthy.
4. **Gán vị trí theo nhiệm vụ:** leaf_bbox_xyxy dùng tọa độ pixel trên ảnh đang duyệt sau chỉnh EXIF, dạng x1,y1,x2,y2. Vùng tổn thương lưu ở lesion_annotation_file khi có người gán/duyệt. Với crop cũ, cần bổ sung riêng ảnh cha, parent_sha256, hộp crop và hệ tọa độ ảnh cha. Không dùng heatmap chú ý của model làm annotation bệnh đã xác nhận.
5. **Khóa phiên bản mới:** nhóm ảnh cùng lá/phiên chụp, kiểm trùng và xung đột; xuất manifest mới và audit. Hàng đợi này chỉ chứa train. Giữ holdout cũ để đối chiếu, thu thêm bộ đánh giá độc lập được duyệt mù; không sửa test theo lỗi của model để nâng benchmark.
6. **Train và đánh giá lại:** train trên phần đã duyệt, báo tỷ lệ nhãn duyệt theo nguồn/lớp, precision/recall/F1 từng lớp và lỗi bệnh bị báo khỏe. Chỉ chọn model bằng validation theo tiêu chí đã khóa; bằng chứng metadata đầy đủ chưa thay thế đánh giá ảnh thực tế.

Thứ tự thực hiện phù hợp là chuẩn hóa và duyệt nhãn trước, thử crop lá được gán vị trí, rồi train/đánh giá lại. Chưa có đủ dữ liệu để khẳng định đổi kiến trúc hoặc train thêm epoch sẽ sửa lỗi nhãn.
