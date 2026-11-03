# Tiếp tục train với Antigravity, ngày 03/11/2026

[Kết quả đã hoàn tất](RESULTS.md): chạy bốn epoch, giữ trọng số parent vì không epoch nào giảm validation loss. F1 PlantDoc CPU vẫn 54,98%; calibration tăng độ đúng trong nhóm qua cổng lên 66,67%, nhưng coverage còn 18,84%. Chưa đạt tiêu chí phát hành.

Kết nối qua CLI `agy` đã cài trên máy, mã phiên lưu local trong `artifacts/antigravity_continuation_20261103/connection.json`. Antigravity tư vấn và rà soát mã bằng văn bản; Codex thực thi training và đánh giá qua terminal của phiên Codex. Chưa xác minh việc đồng bộ cuộc trò chuyện này với cửa sổ desktop.

Lần gọi headless đầu bị Antigravity tự từ chối `RunCommand` vì không thể hỏi quyền tương tác. Không đổi cấu hình quyền hoặc dùng cờ bỏ qua kiểm tra. Những lượt trao đổi sau không yêu cầu Antigravity gọi tool. Cách gọi CLI và tiếp tục conversation được mô tả trong [tài liệu chính thức](https://antigravity.google/docs/cli/reference/).

## Cấu hình thí nghiệm

Warm-start từ `models/mobilenetv3_field_v2_20261103.pt`, checkpoint đa nguồn sáu epoch trước. Khởi tạo AdamW mới từ trọng số parent; đây không phải khôi phục optimizer của lượt sáu epoch cũ. Lượt mới lưu model, optimizer, GradScaler, thứ tự lấy mẫu, vị trí batch và RNG mỗi 50 batch và khi pause để phục hồi đúng trạng thái của chính lượt này.

| Tham số | Giá trị |
|---|---|
| Model | MobileNetV3-Large, 11 lớp SIC + Unsupported |
| Manifest | `data/processed/field_mobile_v2/records.json` |
| Chia split | 16.225 train, 2.740 validation, 1.937 test lịch sử |
| Số epoch | 4, mỗi epoch lấy mẫu có hoàn lại 6.400 lượt |
| Batch / seed | 32 / 42 |
| Backbone | Mở chín block cuối; LR phần giữa 0,000005, bốn block cuối 0,00001 |
| Classifier | LR 0,00003 |
| Lấy mẫu / loss | Cân bằng domain và lớp qua `weights_for`; cross-entropy, label smoothing 0,05; AdamW weight decay 0,0001 |
| Chọn checkpoint | Validation loss trung bình đều theo domain; yêu cầu F1 PlantVillage ≥95%; parent cũng là ứng viên |
| Calibration | Temperature scaling bằng NLL validation cân bằng domain, không tối ưu theo cổng confidence |
| Đầu ra | `models/mobilenetv3_antigravity_warmstart_20261103.pt` |

Giữ BatchNorm ở chế độ eval khi fine-tune. Đây là thí nghiệm với cấu hình đầy đủ trong protocol và metadata, chưa phải ablation chỉ thay learning rate. Antigravity chưa trả verdict rà soát mã trong giới hạn 90 giây; kiểm tra mã và tám kiểm thử ML được Codex thực hiện trước khi chạy. Phản hồi tư vấn của một agent không thay thế kiểm thử hoặc đánh giá model.

Manifest khớp hash mà parent đã học. Kiểm tra không thấy train trùng SHA chính xác với holdout; điều này chưa chứng minh mọi lá/phiên chụp đã tách biệt. Nhãn nguồn chưa được chuyên gia của nhóm duyệt lại toàn bộ. Holeta đã có trong train, không phải external test độc lập của model này. Lượt mới không đọc test để lựa chọn model hoặc điều chỉnh ngưỡng.

## Chạy và khôi phục

```powershell
.venv-gpu/Scripts/python.exe -u -m ml.continue_field --source models/mobilenetv3_field_v2_20261103.pt --output models/mobilenetv3_antigravity_warmstart_20261103.pt --data data/processed/field_mobile_v2 --epochs 4 --batch-size 32 --samples-per-epoch 6400 --unfreeze-blocks 9 --backbone-lr 5e-6 --top-backbone-lr 1e-5 --head-lr 3e-5 --min-village-f1 .95 --balanced-calibration
```

Khi có `.progress.pt` và chưa có checkpoint cuối, chạy lại cùng lệnh để khôi phục. Cần giữ nguyên mã trainer, transform, manifest và cấu hình. Checkpoint cuối có sẵn sẽ bị từ chối ghi đè. Sau training, đánh giá CPU baseline/parent/candidate trên cùng manifest validation và áp dụng tiêu chí trong `docs/FIELD_RELEASE_CRITERIA.json`; model web không tự thay đổi.
