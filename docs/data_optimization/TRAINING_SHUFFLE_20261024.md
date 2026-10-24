# Sửa cách trộn ảnh khi fine tune

Đã sửa pipeline để xáo trộn từng ảnh trên toàn bộ tập train, thay vì chỉ xáo trộn các shard vốn được sắp theo lớp. Bộ kiểm thử 18/18 pass. Model đang triển khai được giữ nguyên vì chưa có bằng chứng cải thiện chất lượng đủ tin cậy.

## Vấn đề và cách sửa

Manifest chuẩn bị dữ liệu được sắp theo đường dẫn. Các ảnh cùng lớp nằm cạnh nhau và được ghi thành shard 32 ảnh. Trước đây fine-tune xáo trộn thứ tự shard rồi cập nhật optimizer một lần mỗi shard, khiến nhiều batch chỉ có một lớp. Việc này làm batch thiếu đa dạng dù toàn bộ epoch đã được shuffle.

`CachedSamples` hiện tạo chỉ mục từng ảnh, xáo trộn chỉ mục trên toàn bộ tập train rồi nạp đúng feature/nhãn theo chỉ mục. LRU cache giữ tối đa 16 shard; các tensor batch được tạo tạm thời nên bộ nhớ không tăng theo toàn bộ bộ dữ liệu. `--batch-shards 4` với shard size 32 tạo batch huấn luyện thực tế 128 ảnh; đây là ghép batch, không phải gradient accumulation.

Để bắt đầu từ baseline gốc và tái sử dụng cache đã có, công cụ kiểm tra manifest, batch size, seed, dtype và SHA-256 checkpoint nguồn cache. Sau đó so sánh tất cả state tensor trước projection giữa checkpoint nguồn và checkpoint đầu vào. Nếu backbone khác, công cụ từ chối cache. Projection và head được phép khác vì không nằm trong phần activation đã cache.

Kiểm thử mới dùng shard được sắp theo lớp để xác nhận việc lấy mẫu có thể trộn nhiều lớp trong một batch, không làm lệch nhãn và duyệt đủ mỗi mẫu đúng một lần trong epoch.

## Thử nghiệm có giới hạn

Cả hai lượt mới bắt đầu từ baseline đang triển khai, train projection Conv1×1 cuối cùng head, giữ BatchNorm frozen và một augmented view đã seed. Tối đa 8 epoch, dừng sau 3 epoch không tăng macro F1 validation.

| Lượt | Best epoch | Macro F1 validation |
|---|---:|---:|
| Baseline trước huấn luyện | 0 | 94,4801% |
| Shuffle từng ảnh, batch 128, LR 1e-5 | 1 | 94,5225% |
| Shuffle từng ảnh, batch 128, LR 1e-4 | 0 | 94,4801% |

Lượt LR 1e-5 kết thúc sau 4 epoch và được xác nhận bằng suy luận FP32 trực tiếp từ đủ 1.731 ảnh validation. Lượt LR 1e-4 kết thúc sau 3 epoch, giữ trọng số epoch 0. Tập test không được đọc lại trong lượt tiếp tục này.

Mức tăng chỉ khoảng 0,0424 điểm phần trăm trên validation đã được dùng nhiều lần. Vì chênh lệch quá nhỏ và chưa có tập đánh giá độc lập mới, không thay model hoặc ONNX đang triển khai. Các lượt cũng thay cả batch size và checkpoint khởi tạo so với lượt trước, nên chưa phải phép ablation đủ để kết luận riêng tác động của shuffle lên F1. Điều đã xác nhận trực tiếp là cách trộn batch đúng và pipeline hoạt động ổn định hơn trong lượt này.

## Dừng và tiếp tục

Đã kiểm tra marker pause trong lúc job LR 1e-4 chạy: job ghi checkpoint và dừng, sau đó khôi phục optimizer, scheduler, RNG và tiếp tục từ epoch index 2, batch index 29 (epoch 3 khi đếm từ 1). Lượt đó hoàn tất mà không chạy lại các batch đã xác nhận hoàn thành. Đây là kiểm tra chức năng phục hồi, không phải so sánh bit-for-bit với một lượt không bị ngắt.

```powershell
python -m ml.finetune_projection --weights models/mobilenetv2_crop_disease.pt --cache-source-weights models/mobilenetv2_head_v2_20261024.pt --activation-cache artifacts/projection_170eab8cc1ae9b44 --batch-shards 4 --lr .0001 --epochs 8 --output models/projection_shuffle_new_run.pt
python -m unittest discover -s tests -v
```

Không ghi đè file checkpoint cuối đã tồn tại; chọn output khác cho lượt mới. Khi checkpoint cuối chưa có, chạy lại cùng tham số để tiếp tục resume. Cache/source weights và dữ liệu cần được giữ local hoặc tái lập theo các báo cáo trước.

Kết quả ở `results/optimization_20261024_shuffle/`. Các thay đổi có sẵn trong REPORT.md, benchmark.json và environment.json vẫn được giữ riêng. Push GitHub hiện thiếu credential; các commit được lưu local.

Theo yêu cầu hiện hành, ngưỡng 5% Limits chỉ dùng để lưu và dừng task, không tắt máy. Khi không còn job chạy sau khi lượt thử nghiệm hoàn tất, xóa lịch theo dõi để tránh tiêu hao hạn mức do polling.
