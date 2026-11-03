# Kết quả expanded_20261103

Đã train bốn epoch trên NVIDIA GeForce RTX 2050 trong 25.15 phút. Chọn `continued/3` bằng validation loss; temperature = 0.835299. Model chưa đạt tiêu chí phát hành.

## Validation CPU trên cùng manifest

| Tập | Số ảnh | V2 baseline | V3 parent | V3 expanded |
|---|---:|---:|---:|---:|
| PlantVillage | 1731 | 94.48% | 97.52% | 98.21% |
| PlantDoc | 414 | 15.95% | 46.59% | 45.83% |
| FieldSupplement | 103 | 44.13% | 50.21% | 65.17% |

Chỉ số trong bảng là macro F1 trên các lớp có ground truth trong từng tập, giữ nguyên lỗi dự đoán sang cả 12 nhãn. FieldSupplement validation chỉ có bảy lớp bệnh PlantWild_v2 và 103 ảnh, chưa có healthy Ecuador trong validation.

Expanded tăng 14,96 điểm phần trăm F1 trên validation bổ sung so với parent, nhưng giảm 0,77 điểm trên PlantDoc. Epoch 4 có F1 PlantDoc GPU 49,45% trong training log; đó không phải checkpoint được chọn. Đánh giá CPU và chỉ số GPU trong lúc train có thể khác ở các quyết định sát ranh giới; dùng bảng CPU để so sánh các model.

## Tiêu chí phát hành

Ở PlantDoc, expanded có accuracy 56,82% trong các dự đoán vượt ngưỡng score 75%, thấp hơn yêu cầu 85%. Coverage 21,26%, thấp hơn yêu cầu 25%. Đây là cổng score trước bước review healthy của API, không phải tỷ lệ kết quả tư vấn cuối cùng được ứng dụng chấp nhận.

Có 0/128 ảnh bệnh thuộc các lớp đã biết bị dự đoán healthy với confidence cao trong tập này. Con số này không đánh giá ảnh lá thủng do sâu hoặc mọi ảnh ngoài miền. Quyết định validation là giữ default V2; V3 expanded vẫn thử nghiệm. Xem [quyết định máy đọc được](release_decision.json).

## External Holeta sau khi khóa model

| Model | Accuracy | Macro F1 hai lớp | Recall Healthy | Recall Late Blight |
|---|---:|---:|---:|---:|
| parent | 17.70% | 37.20% | 10.83% | 53.73% |
| candidate | 25.12% | 29.94% | 16.24% | 71.64% |
| Luôn đoán Potato_healthy (đối chứng theo tỷ lệ lớp) | 83.97% | 45.64% | 100,00% | 0,00% |

Tập gồm 351 Healthy và 67 Late Blight. Expanded tăng accuracy từ 17,70% lên 25,12%, nhưng macro F1 giảm từ 37,20% xuống 29,94%; precision Late Blight chỉ 20,60%. Cả hai model còn kém đối chứng luôn đoán lớp đông nhất trên nguồn này. Không thể dùng kết quả này để tuyên bố model đã chạy tốt ngoài đồng.

Kết quả cho thấy cải thiện ở validation cùng nguồn chưa chuyển thành cải thiện trên Holeta. Chưa có thí nghiệm ablation để kết luận nguyên nhân. Không dùng external test này để chọn lại epoch, fit temperature hoặc sửa ngưỡng. Nếu điều chỉnh model sau khi xem kết quả, cần công bố việc đó và giữ thêm nguồn chưa từng dùng cho đánh giá độc lập tiếp theo.

## Nhánh đa nguồn khác trong workspace

Một thí nghiệm khác đã hoàn tất sáu epoch và chọn `field_v2/epoch_6`, dùng `field_mobile_v2` (20.902 record). Hồ sơ CPU trên benchmark cũ ghi F1 PlantDoc 54,98%, accuracy ở cổng score 59,22% và coverage 24,88%. Đây là F1 PlantDoc cao nhất trong các checkpoint được đối chiếu ở đây, nhưng vẫn trượt yêu cầu accuracy 85% và coverage 25%.

Đã đối chiếu SHA-256 của checkpoint với báo cáo validation, hash manifest train với metadata train và hash manifest đánh giá với benchmark v1. Giữ nguyên các holdout cũ và không thấy train trùng SHA chính xác với holdout trong manifest v2; kiểm tra này không chứng minh tách biệt mọi lá/phiên chụp. Không chạy lại phép đánh giá đó trong lượt expanded này, cũng chưa duyệt lại toàn bộ nhãn mới của nhánh v2.

Model này đã học Holeta, nên không dùng Holeta để tuyên bố khả năng tổng quát hóa của nó. Xem [phạm vi xác minh](separate_workspace_experiment.json), [validation snapshot](separate_workspace_validation.json) và [training snapshot](separate_workspace_training.json). Trọng số local là `models/mobilenetv3_field_v2_20261103.pt`; vẫn giữ trạng thái thử nghiệm.

## Bằng chứng và kiểm thử

- [Training metadata](training.json), [training log](training_log.txt), [hash mã thực thi](implementation_sha256.json).
- [Baseline validation](baseline_validation.json), [parent validation](parent_validation.json), [expanded validation](candidate_validation.json).
- [Cấu hình/hash khóa trước external test](evaluation_protocol.json), [parent external](parent_external_test.json), [expanded external](candidate_external_test.json).
- [11 kiểm thử ML đạt](verification.json), [log kiểm thử](unit_tests.txt). Không sửa frontend hoặc inference contract trong lượt này.

Trọng số: `models/mobilenetv3_expanded_20261103.pt`. SHA-256: `2285c544286d6fbde3bd280052894674a20be8b40ad9ad9456d87dc9db96f06d`. Đây là file thử nghiệm, không thay parent và không thay model web.
