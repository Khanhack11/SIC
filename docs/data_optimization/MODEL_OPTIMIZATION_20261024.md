# Thử nghiệm cải thiện model ngày 24 tháng 10 năm 2026

Giữ checkpoint đang triển khai vì các ứng viên trong vòng này chưa cải thiện chất lượng trên test. Không thay file model hoặc graph ONNX mặc định. Mã huấn luyện, resume, kiểm thử và kết quả của các thử nghiệm được lưu để tái lập.

## Kết quả

| Phép đo | Baseline đang triển khai | Ứng viên Linear được chọn bằng validation |
|---|---:|---:|
| Validation macro F1 | 94,4801% | 94,8793% |
| Test accuracy | 94,1860% | 93,8953% |
| Test macro F1 | 93,7632% | 93,3147% |
| Test NLL sau calibration | 0,16317 | 0,18170 |
| Test ECE sau calibration | 0,00565 | 0,01198 |

Sáu cấu hình Linear được thử bằng feature cache của 16.086 ảnh augmented train và 1.731 ảnh validation. Tối ưu bằng LBFGS, chuẩn hóa feature và L2 regularization; class weighting thử exponent 0,5 và 1,0, L2 thử 0,0001, 0,001 và 0,01. Cấu hình validation tốt nhất là exponent 0,5 và L2 0,001. Chuẩn hóa được fold vào weight và bias của Linear để giữ nguyên kiến trúc MobileNetV2 và chi phí inference.

Trước khi dùng cache, công cụ đối chiếu hợp đồng nhãn, hash manifest, nhãn validation và tái tính feature trên 16 ảnh validation. Kết quả validation của ứng viên sau đó được xác nhận bằng suy luận FP32 từ toàn bộ ảnh, khớp kết quả cache.

Fine-tune projection Conv1×1 cuối và Linear cũng được thử với learning rate 0,0001 và 0,00001. Các lớp trước và BatchNorm được đóng băng. Dùng AdamW, trọng số lớp căn bậc hai, cosine schedule và dừng sau 3 epoch không cải thiện macro F1. Cả hai thử nghiệm không vượt ứng viên Linear trên validation; checkpoint tốt nhất của chúng vẫn là epoch 0. Không chọn các trọng số fine-tune này.

Chỉ ứng viên Linear đã chọn bằng validation được đánh giá trên test. Test gồm 1.720 ảnh; macro F1 giảm 0,4486 điểm phần trăm, NLL và ECE cũng kém hơn baseline. Vì không vượt qua bước xác nhận cuối, ứng viên không được triển khai. Không dùng kết quả này để tiếp tục tìm hyperparameter trên test.

ONNX FP32 xuất từ ứng viên cho cùng accuracy và macro F1 test với eager. Các phép đo tốc độ được lưu riêng trong `results/optimization_20261024/selected_benchmark.json`; ứng viên vẫn bị loại do chất lượng, bất kể tốc độ.

## Những gì đã bổ sung

- `ml/optimize_head.py`: tìm có giới hạn 6 cấu hình Linear, lưu resume sau mỗi nhóm bước LBFGS, giữ nguyên checkpoint đang triển khai.
- `ml/finetune_projection.py`: cache activation FP16 theo shard trên đĩa, giữ RAM thấp trên máy 8GB, fine-tune projection cuối và head, lưu optimizer/RNG/vị trí batch để tiếp tục sau khi dừng.
- `ml/evaluate.py --split validation`: xác nhận validation từ ảnh đầy đủ trước khi chọn ứng viên; mặc định vẫn là test để tương thích lệnh cũ.
- Ba kiểm thử mới xác nhận fold chuẩn hóa, đường chạy từ activation cache và phạm vi gradient. Bộ kiểm thử 17/17 pass.

Lượt Linear đầu gặp khóa file khi rename trên Windows; công cụ thêm retry có giới hạn và lượt v2 hoàn tất. Tiến độ lượt đầu được giữ riêng, không dùng lẫn với lượt v2.

## Tái lập và tiếp tục

```powershell
python -m ml.optimize_head --output models/head_reproduction.pt --report artifacts/head_reproduction.json
python -m ml.finetune_projection --weights models/head_reproduction.pt --output models/projection_reproduction.pt
python -m ml.evaluate --split validation --weights models/projection_reproduction.pt --output artifacts/projection_validation.json
python -m unittest discover -s tests -v
```

Lượt đã chạy với learning rate thấp có thể kiểm tra từ `models/mobilenetv2_projection_lr1e5_20261024.json`. Cache gốc tương ứng nằm trong `artifacts/projection_170eab8cc1ae9b44`; `--activation-cache` kiểm tra fingerprint đầu vào, manifest, seed, batch size và dtype trước khi cho tái sử dụng. Không ghi đè checkpoint cuối; chọn output mới nếu thực hiện thử nghiệm khác.

Khi Limits còn không quá 5% ở một cửa sổ, automation ghi `artifacts/PAUSE_REQUESTED` rồi lưu và dừng task. Đây là yêu cầu hiện hành ngày 24/10: không tắt máy. Muốn tiếp tục sau khi task đã dừng, kiểm tra hạn mức, xóa đúng marker rồi chạy lại lệnh có cùng tham số. LBFGS lưu từng nhóm 10 bước; fine-tune lưu định kỳ mỗi 20 batch và ngay khi đọc marker pause. Cache activation lưu ngay từng shard.

## Giới hạn

Validation và test này đều đã được sử dụng ở các vòng trước. Chênh lệch nhỏ không chứng minh cải thiện tổng quát; không nên tiếp tục lựa chọn nhiều model trên cùng test. Kết quả vẫn giới hạn trong subset PlantVillage có nền được kiểm soát, chưa đánh giá trên ảnh thực địa độc lập. FP16 chỉ dùng làm cache activation để huấn luyện projection; phép xác nhận model đã chọn dùng đầy đủ ảnh FP32.

Bước tiếp theo có giá trị là bổ sung một tập ảnh thực địa được gán nhãn độc lập và định trước tiêu chí đánh giá. Không có bằng chứng trong vòng này để thay baseline đang hoạt động.
