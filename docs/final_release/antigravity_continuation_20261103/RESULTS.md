# Kết quả warm-start phối hợp Antigravity

Đã chạy bốn epoch trên NVIDIA GeForce RTX 2050 trong 13.46 phút. Không epoch nào giảm validation loss thấp hơn parent, nên checkpoint cuối giữ trọng số parent và cập nhật temperature bằng NLL validation cân bằng domain. Model web giữ nguyên.

## So sánh CPU trên cùng manifest

| Tập validation | Số ảnh | V2 baseline | V3 parent | Candidate sau calibration |
|---|---:|---:|---:|---:|
| PlantVillage | 1731 | 94.48% | 98.17% | 98.17% |
| PlantDoc | 414 | 15.95% | 54.98% | 54.98% |
| MultiSourceField | 595 | 21.74% | 66.33% | 66.33% |

Bảng dùng macro F1 trên các lớp có ground truth, giữ lỗi dự đoán sang cả 12 nhãn. Trọng số parent và candidate được so sánh từng tensor, kết quả giống hệt nhau. Temperature thay đổi từ 0,65 sang 1,005059, nên nhãn top-1 không đổi. Một số số đo CPU và GPU trong log có thể lệch ở các trường hợp sát ranh giới; phép so sánh trên dùng cùng runtime CPU.

## Bốn epoch đã chạy

| Ứng viên | Validation loss cân bằng domain | F1 PlantDoc GPU | Chọn |
|---|---:|---:|---|
| Parent trước lượt này | 0.530871 | 54,98% | Có |
| Epoch 1 | 0.558525 | 56.21% | Không |
| Epoch 2 | 0.548837 | 55.20% | Không |
| Epoch 3 | 0.553304 | 56.20% | Không |
| Epoch 4 | 0.543065 | 56.06% | Không |

Đây là kết quả âm của lượt warm-start: đã cập nhật trọng số trong bốn epoch, nhưng các trạng thái mới không đạt tiêu chí chọn checkpoint đã khóa. Không chọn lại epoch theo F1 cao nhất sau khi xem kết quả. Không thể báo checkpoint cuối đã tăng F1 lên 56,21% chỉ vì epoch 1 có số đo đó.

## Hiệu chuẩn và cổng score PlantDoc

| Chỉ số | Parent | Candidate | Yêu cầu phát hành |
|---|---:|---:|---:|
| Độ đúng trong nhóm qua cổng | 59.22% | 66.67% | ≥85% |
| Tỷ lệ ảnh qua cổng | 24.88% | 18.84% | ≥25% |
| Dự đoán qua cổng đúng / tổng | 61 / 103 | 52 / 78 | |

Cổng được định nghĩa là confidence ≥75% và dự đoán khác Unsupported, trước bước review kết quả healthy của API. Không dùng `calibration.accepted_accuracy` làm accuracy tư vấn: trường đó tính cả dự đoán Unsupported, khác với `score_gate_accuracy` được tiêu chí phát hành sử dụng.

Temperature lớn hơn làm model bớt tự tin, một số ảnh bị loại khỏi nhóm qua cổng. Độ đúng của tập con tăng, nhưng lượng ảnh được giữ lại giảm. Điều này không chứng minh khả năng phân loại toàn bộ ảnh đã tăng.

| Domain | NLL parent | NLL candidate | ECE parent | ECE candidate |
|---|---:|---:|---:|---:|
| PlantVillage | 0.0513 | 0.0917 | 1.07% | 4.82% |
| PlantDoc | 0.9463 | 0.7789 | 12.70% | 6.03% |
| MultiSourceField | 0.8410 | 0.7216 | 10.25% | 4.25% |

Calibration cải thiện NLL/ECE trên hai domain ảnh field của validation, nhưng làm NLL/ECE PlantVillage tăng. Chưa có phép đo độc lập mới để khẳng định cải thiện ngoài môi trường thực tế. Cả accuracy và coverage PlantDoc vẫn trượt cổng, nên không thay model phục vụ.

## Phạm vi kết nối và kiểm chứng

Antigravity đã nhận kế hoạch và phản hồi kết quả qua CLI. Nó đồng ý giữ parent theo tiêu chí NLL. Lượt rà soát mã riêng trước đó hết thời gian 90 giây, không nhận verdict. Codex thực hiện mã, tám kiểm thử ML, training và đánh giá CPU. Quyền RunCommand của Antigravity headless bị tự từ chối; không thay cấu hình quyền hoặc dùng cờ bỏ qua.

- [Kết nối và vai trò](connection.json), [phản hồi kết quả của Antigravity](antigravity_results_review.json).
- [Protocol](locked_protocol.json), [training metadata](training.json), [log train](training_log.txt).
- [So sánh tensor và hash](weight_comparison.json), [hash mã nguồn](implementation_sha256.json), [tám kiểm thử](unit_tests.txt).
- [Baseline validation](baseline_validation.json), [parent validation](parent_validation.json), [candidate validation](candidate_validation.json), [quyết định phát hành](release_decision.json).

Test lịch sử 1.937 ảnh không được đọc trong lượt này. Holeta đã có trong train của parent, không được đánh giá như một nguồn chưa từng học. Nhãn nguồn chưa có chuyên gia của nhóm duyệt lại toàn bộ; việc không trùng SHA không bảo đảm tách biệt mọi lá vật lý.

Trọng số thử nghiệm nằm tại `models/mobilenetv3_antigravity_warmstart_20261103.pt`. Trạng thái optimizer/scaler/sampler/RNG nằm trong file `.progress.pt` cùng tên. Nếu lặp lại cùng dữ liệu, seed và cấu hình từ parent, không được coi đó là phép đo độc lập mới.
