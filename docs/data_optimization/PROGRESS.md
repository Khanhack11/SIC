# Tiến độ phần Data và Optimization của Trần Đặng Công Tâm

Ngày làm việc 16/10/2026. Nhánh local `tran-dang-cong-tam`.

## Lượt tiếp tục ngày 24 tháng 10 năm 2026

- Đã kiểm tra commit `fedf43c` còn local. Push không tương tác xác nhận Git thiếu credential (`Cannot prompt because user interactivity has been disabled`); chưa đẩy được lên GitHub.
- Phát hiện pipeline fine-tune chỉ shuffle thứ tự shard trong khi manifest/shard được sắp theo lớp. Đã sửa shuffle ở mức từng ảnh trên toàn bộ train, ghép batch thực tế 128 ảnh qua `--batch-shards 4`; cache LRU 16 shard giữ bộ nhớ có giới hạn.
- Bắt đầu lại từ baseline gốc, không từ ứng viên Linear đã bị loại. Reuse cache chỉ khi manifest/seed/dtype khớp và toàn bộ state trước projection bằng nhau giữa checkpoint nguồn cache và checkpoint đầu vào.
- Lượt shuffle LR 1e-5 hoàn tất sau 4 epoch, best epoch 1, macro F1 validation cache 94,5225%. File `models/mobilenetv2_projection_shuffled_20261024.pt`. Đang xác nhận full FP32 validation.
- Đã hoàn tất đối chứng LR 1e-4 sau 3 epoch, best epoch 0; không vượt baseline. LR 1e-5 được xác nhận full FP32 validation đạt macro F1 94,5225%, tăng khoảng 0,0424 điểm phần trăm so với baseline 94,4801%.
- Đã kiểm tra pause/resume thực tế: lưu và dừng ở epoch index 2/batch index 29, chạy lại cùng lệnh tiếp tục và hoàn tất. 18/18 kiểm thử pass.
- Giữ checkpoint/ONNX đang triển khai, SHA-256 checkpoint vẫn khớp baseline đã ghi trong báo cáo test gốc. Chênh lệch trên validation cũ chưa đủ để triển khai ứng viên mới. Báo cáo `TRAINING_SHUFFLE_20261024.md`, kết quả `results/optimization_20261024_shuffle/`.
- Không dùng test để chọn cấu hình và chưa đọc lại test trong lượt này. Mọi ứng viên vẫn là thử nghiệm, chưa thay baseline/ONNX mặc định.
- Automation `l-u-v-d-ng-t-i-u-sic-khi-limits-c-n-5` theo dõi 5 phút khi task đang chạy; <=5% thì lưu và dừng, không tắt máy. Khi lượt này hoàn tất và không còn job, xóa lịch theo dõi để tránh polling vô ích. Automation cũ đã xóa sau khi lượt trước kết thúc.

Lệnh tiếp tục đúng lượt hiện tại:

```powershell
python -u -m ml.finetune_projection --weights models/mobilenetv2_crop_disease.pt --cache-source-weights models/mobilenetv2_head_v2_20261024.pt --activation-cache artifacts/projection_170eab8cc1ae9b44 --batch-shards 4 --lr .0001 --epochs 8 --output models/mobilenetv2_projection_shuffled_lr1e4_20261024.pt
```

## Vòng tối ưu model ngày 24 tháng 10 năm 2026

Yêu cầu hiện hành: khi Limits còn không quá 5% ở một trong hai cửa sổ, lưu và dừng task, KHÔNG tắt máy. Automation `l-u-sic-v-t-t-m-y-khi-codex-c-n-5` đã được đổi tên/nội dung sang dừng task và bật mỗi 5 phút. Nội dung yêu cầu tắt máy ở các mục lịch sử bên dưới đã hết hiệu lực.

- Checkpoint đang triển khai vẫn giữ nguyên tại `models/mobilenetv2_crop_disease.pt` và graph ONNX tương ứng.
- Đã thử 6 cấu hình đầu Linear bằng feature cache đã kiểm tra hợp đồng nhãn, hash manifest và tái tính feature của 16 ảnh validation. Cấu hình tốt nhất: class-weight exponent 0,5; L2 0,001; validation macro F1 94,8793% so với baseline 94,4801%. Standardization được fold vào weight/bias để không tăng chi phí inference.
- Ứng viên head: `models/mobilenetv2_head_v2_20261024.pt`. Báo cáo `results/optimization_20261024/head_search.json`. Lượt `mobilenetv2_head_20261024_search` đầu tiên gặp file lock khi rename trên Windows; lượt v2 có retry và hoàn tất.
- Đang chạy `python -u -m ml.finetune_projection`: fine-tune projection Conv1×1 cuối và head, BatchNorm frozen, activation shard FP16 trên đĩa để phù hợp máy 8GB RAM. Lưu resume trong `artifacts/projection_*/resume.pt`; dừng tại batch boundary khi có `artifacts/PAUSE_REQUESTED`.
- Chọn ứng viên bằng validation; test chưa được đọc trong vòng tối ưu này. Validation đã dùng ở các vòng trước nên cải thiện cần được xem cùng nguy cơ overfit validation.
- Đã kết thúc hai thử nghiệm projection (LR 1e-4 và 1e-5); cả hai không vượt epoch 0 và bị loại bằng validation. File ứng viên chỉ lưu lại trọng số epoch 0, không được triển khai.
- Chọn Linear bằng validation rồi xác nhận test: accuracy 93,8953%, macro F1 93,3147%, thấp hơn baseline 93,7632%; NLL/ECE cũng kém hơn. Giữ baseline, không thử tiếp hyperparameter dựa trên test này.
- ONNX ứng viên đã xuất riêng tại `models/mobilenetv2_head_20261024.onnx`, có cùng macro F1 và accuracy với eager trên test. Model và graph mặc định không bị thay thế.
- Bộ kiểm thử 17/17 pass. Báo cáo `MODEL_OPTIMIZATION_20261024.md`, kết quả `results/optimization_20261024/`, quyết định `decision.json`.
- Đã kết thúc đánh giá và benchmark ứng viên, không còn tiến trình tối ưu của vòng này chạy. ONNX ứng viên có p95 9,85 ms serial và 25,36 ms concurrency 4 trong process nhưng không được triển khai vì chất lượng không cải thiện. Mã và báo cáo được commit trên nhánh `tran-dang-cong-tam`; weights/search/cache giữ local.

Lệnh tiếp tục:

```powershell
python -u -m ml.finetune_projection
python -m ml.evaluate --split validation --weights models/mobilenetv2_projection_20261024.pt --output docs/data_optimization/results/optimization_20261024/projection_validation.json
```

## Vòng tối ưu ngày 17 tháng 10 năm 2026

- Người dùng yêu cầu tiếp tục tối ưu và tắt máy khi Limits còn không quá 5%. Theo dõi nền đã được tạo thành công, automation `l-u-sic-v-t-t-m-y-khi-codex-c-n-5`, kiểm tra mỗi 5 phút. Tiếp tục kiểm tra trực tiếp khi làm việc.
- Đã sửa API để CPU inference chạy trong threadpool, giới hạn số lượt inference đồng thời bằng `SIC_INFERENCE_CONCURRENCY` (mặc định 2), và áp dụng ngưỡng tư vấn riêng cho request thay vì đổi biến dùng chung. Upload được đọc tối đa giới hạn file cộng 1 byte.
- Đã bổ sung ONNX Runtime FP32 với metadata kiểm tra nhãn, input contract và SHA-256 checkpoint. Model local: `models/mobilenetv2_crop_disease.onnx`.
- Validation đủ 1.731 ảnh: eager, TorchScript và ONNX FP32 cùng macro F1 94,4801%, prediction agreement 100%. INT8 chỉ đạt 90,7411%, đã loại khỏi lựa chọn triển khai. Không đổi checkpoint baseline.
- Perceptual audit trên 11.494 ảnh không gắn cờ cặp gần trùng khác split theo pHash <=2 và dHash <=4. Không suy ra mọi lá vật lý đã tách biệt; metadata dự phòng vẫn là giới hạn.
- Các kết quả mới được ghi riêng ở `results/optimization_20261017/`. Các thay đổi REPORT.md, benchmark.json và environment.json có sẵn đầu lượt được giữ lại; bản sao nằm trong `artifacts/optimization_20261017_baseline/`.
- Đang đo HTTP localhost bằng `python -m ml.api_benchmark`; registry server do công cụ sở hữu: `artifacts/api_active_process.json`. Công cụ tự dừng server của mình trong finally khi được yêu cầu pause.
- Đã chọn ONNX FP32, 2 threads, 2 inference slots. Vòng xác nhận 150 request mỗi mức tải: HTTP mean/p95 19,28/35,63 ms với 1 client; 45,15/54,39 ms với 4 client; 81,31/93,96 ms với 8 client.
- 14 kiểm thử unit/integration, 10 kiểm thử backend và 17 kiểm thử stress cuối đều pass. Server benchmark đã được dừng; registry active=false.
- Mặc định API tự chọn ONNX nếu graph và dependency có sẵn, còn lại dùng TorchScript. Báo cáo đầy đủ: `OPTIMIZATION_20261017.md`. Vòng tối ưu đang được commit và push trên nhánh `tran-dang-cong-tam`; các thay đổi REPORT.md, benchmark.json và environment.json có sẵn đầu lượt vẫn được giữ riêng trong working tree.

Lệnh tiếp tục sau khi kiểm tra/xóa marker pause cũ nếu đã được người dùng cho phép tiếp tục:

```powershell
python -m ml.runtime_validation
python -m ml.near_duplicate_audit
python -m ml.api_benchmark --threads 2 4 --slots 1 2 --concurrency 1 4 8 --runs 80
python -m unittest discover -s tests -v
```

## Trạng thái hiện tại

- Đã tải 11.504 ảnh RGB của 11 lớp từ PlantVillage, kiểm tra Git blob hash từng ảnh.
- Nguồn được ghim ở commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a` của `spMohanty/PlantVillage-Dataset`.
- Đã loại 10 ảnh trùng pixel, giữ 11.494 ảnh; chia 8.043 train / 1.731 validation / 1.720 test. Hash file, pixel và nhóm đã xác định không giao nhau giữa các tập.
- Có 5.671 ảnh khớp leaf-map, 5.833 ảnh dùng identifier dự phòng; không khẳng định mọi lá vật lý đã được tách biệt.
- Đã viết pipeline augmentation, tiền xử lý dùng chung với backend, huấn luyện baseline, đánh giá và benchmark.
- Đã hoàn thành trích xuất feature và huấn luyện baseline 59 epoch, best epoch 47. Checkpoint cuối: `models/mobilenetv2_crop_disease.pt`.
- Đã kiểm tra cách bố trí tensor: batch 64 dùng channels-last khoảng 1,5 giây, contiguous khoảng 3,1 giây ở 4 threads. Đây là microbenchmark kỹ thuật, chưa phải latency API.
- Test 1.720 ảnh: accuracy 94,19%, macro F1 93,76%, đạt mục tiêu F1 ≥85% trên subset PlantVillage này.
- Benchmark serial được khuyến nghị: TorchScript, 2 CPU threads; mean 14,93 ms, p95 19,45 ms. Chưa gồm HTTP/network/advisory. Cấu hình mặc định API vẫn là eager 4 threads; dùng biến môi trường trong README để chọn cấu hình benchmark.
- 8 kiểm thử mới của pipeline/API đã pass. Kết quả backend và stress test được lưu trong `results/`.
- Đã tạo `REPORT.md`, sao chép manifests và lưu audit, training, evaluation, benchmark, environment, checksum checkpoint trong bộ bàn giao này.
- Không còn tiến trình huấn luyện/đánh giá/benchmark của chat đang chạy tại thời điểm hoàn thành chuỗi lệnh.

## Lệnh tiếp tục

Chạy trong thư mục gốc SIC. Python trên máy hiện có torch và torchvision bản CPU.

```powershell
python -m ml.download_dataset
python -m ml.prepare_dataset
$env:TORCH_HOME = "$PWD/artifacts/torch"
python -u -m ml.train
python -u -m ml.evaluate
python -u -m ml.benchmark
python -m unittest discover -s tests -v
python test_backend.py
python stress_and_edge_test.py
```

Downloader kiểm tra lại và tiếp tục các file chưa đủ. Nếu đã có `data/processed/plantvillage/audit.json`, không chạy lại prepare vào cùng thư mục; dùng manifest đã có hoặc chọn thư mục output mới. Huấn luyện lưu feature theo batch và trạng thái optimizer theo epoch; chạy lại cùng tham số để tiếp tục khi chưa có checkpoint cuối cùng.

## Yêu cầu về hạn mức

Yêu cầu hiện hành ngày 24/10/2026: khi một cửa sổ Codex còn không quá 5%, lưu checkpoint/tiến độ và dừng task của chat; KHÔNG tắt máy. Automation đã cập nhật và đang active, kiểm tra mỗi 5 phút. Lần kiểm tra gần nhất trong vòng này: còn 79% cửa sổ 5 giờ, 97% tuần.

Để yêu cầu các job ML dừng có lưu trạng thái, tạo `artifacts/PAUSE_REQUESTED`. Chỉ xóa đúng file này khi được cho phép tiếp tục. Không gọi lệnh tắt máy.

## Phạm vi Git

Nhánh `tran-dang-cong-tam` chứa phần Data & Optimization. Ngày 17/10/2026, người dùng yêu cầu commit và đẩy các thay đổi lên `https://github.com/Khanhack11/SIC.git`. Ảnh gốc, feature cache và checkpoint được giữ local theo `.gitignore`; bộ bàn giao trên Git gồm mã nguồn, manifests, kết quả đo và tài liệu tái lập.
