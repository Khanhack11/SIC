# Fine-tune bằng nguồn ảnh bổ sung, bắt đầu 03/11/2026

Đã hoàn tất 6 epoch trên NVIDIA GeForce RTX 2050. Checkpoint được chọn: `continued/3`; 99 tensor state khác parent. Tiêu chí phát hành: chưa đạt. Đây là một lượt thử nghiệm, không tự thay model mặc định.

## Dữ liệu

Manifest `data/processed/field_alternative_v3_20261103/records.json` có SHA-256 `c8883301ff26b47bb12c8c4ec4869cebf621cca32de1837e762c5cadc1d1925e`.

| Phần | Record |
|---|---:|
| Train | 15.458 |
| Validation giữ nguyên | 2.740 |
| Test giữ nguyên | 1.937 |
| Ảnh mới được thêm vào train | 2.829 |

Nguồn bổ sung:

| Nguồn | Record sau lọc | Phạm vi sử dụng |
|---|---:|---|
| [CCMT Ghana, Mensah và cộng sự](https://data.mendeley.com/datasets/bwh3zbpkpv/1) | 2.535 | Septoria cà chua → Tomato_Septoria_leaf_spot; sâu keo ngô → Unsupported |
| [BARI, Ayesha Banu và Kaushik Deb](https://data.mendeley.com/datasets/d5b3fzpw3g/1) | 71 | File mang tên orig_; Healthy/Late Blight và các lớp ngoài phạm vi |
| [Ecuador, CornLeafDiseaseCollection](https://data.mendeley.com/datasets/w56xxnykcc/1) | 223 | Corn_healthy; không ánh xạ tên rust mâu thuẫn |

Các nguồn công bố CC BY 4.0; metadata và hash nguồn lưu trong `data/raw/dataset_hub_20261103/`. CCMT công bố có chuyên gia kiểm chứng; project chưa tự xác nhận lại toàn bộ ảnh. Gói BARI thực tế có 84 file mang tên orig_, không coi mô tả 804 ảnh gốc là 804 file gốc đã tải được.

Bộ chọn CCMT loại 1 file có SHA trùng ở hai thư mục nhãn và 9 file có tên bệnh không khớp thư mục sâu keo. Ảnh mới được kiểm hash nhà phát hành (hoặc hash ZIP + CRC khi giải nén), giải mã, kích thước và trùng/gần giống với dữ liệu đang giữ. 505 ứng viên mới bị loại trong bước kiểm ảnh/trùng. Các ngưỡng hash là heuristic, chưa chứng minh ảnh thuộc các lá độc lập.

Phần train cũ loại 3.160 crop đang thiếu thông tin truy nguyên trong record, 392 record ánh xạ thuật ngữ cần duyệt và 44 record tên lá chung ánh xạ healthy. Đây là loại khỏi lượt train thử nghiệm, chưa phải kết luận mọi nhãn đó sai. Manifest cũ giữ nguyên.

Ảnh mới đều train-only vì thiếu mã lá/phiên chụp đáng tin cậy. Các báo cáo validation sau train vẫn dùng ảnh lịch sử, chưa phải phép kiểm chứng tổng quát hóa trên bộ thực địa độc lập mới. Test không dùng chọn checkpoint.

[Thống kê và hash dữ liệu](data_summary.json). Bảng nhãn local: `artifacts/label_review/alternative_sources_20261103.csv`.

## Cấu hình train

Warm-start từ `models/candidate/mobilenetv3_field.pt`, SHA `0eeb20ec4ef5f3f11cd477920ccc63f33e2b721bb37e394900467f64dc284577`. Giữ 12 đầu ra (11 lớp SIC và Unsupported).

6 epoch × 6.400 lượt lấy mẫu cân bằng domain/lớp, batch 32, CUDA AMP. Mở 9 block cuối; learning rate backbone 8e-6, bốn block cuối 2e-5, classifier 8e-5; cosine schedule và label smoothing 0,04. Đây là 38.400 lượt lấy mẫu có hoàn lại, không phải 38.400 ảnh độc lập.

Chọn checkpoint bằng NLL trung bình các domain validation, yêu cầu F1 PlantVillage tối thiểu 0,95. Calibrate temperature trên validation; áp dụng lại tiêu chí phát hành đã khóa sau khi train. Checkpoint gồm optimizer, scaler, RNG và vị trí batch để khôi phục.

```powershell
.venv-gpu/Scripts/python.exe -u -m ml.continue_field --source models/candidate/mobilenetv3_field.pt --data data/processed/field_alternative_v3_20261103 --output artifacts/training_runs/alternative_sources_20261103/model.pt --expanded-data --epochs 6 --samples-per-epoch 6400 --batch-size 32 --unfreeze-blocks 9 --backbone-lr 0.000008 --top-backbone-lr 0.00002 --head-lr 0.00008 --min-village-f1 0.95 --balanced-calibration --label-smoothing 0.04 --cosine-lr
```

Log: `artifacts/training_runs/alternative_sources_20261103/train.log`. Không ghi đè checkpoint cuối đã có; dùng output mới cho thí nghiệm tiếp theo. Nhãn nguồn còn giới hạn, thay bộ dữ liệu chưa tự bảo đảm chẩn đoán ảnh mới chính xác hơn.

## Kết quả trên CPU, cùng validation

Các cột F1 lấy trung bình các lớp SIC hiện diện, giữ nhầm lẫn sang cả Unsupported. Không trộn số đo GPU trong log train với bảng CPU này.

| Domain | Ảnh | V2 F1 | V3 parent F1 | V3 sau train F1 |
|---|---:|---:|---:|---:|
| PlantVillage | 1731 | 94.48% | 98.17% | 98.10% |
| PlantDoc | 414 | 15.95% | 54.98% | 51.38% |
| MultiSourceField | 595 | 21.74% | 66.33% | 64.86% |

| PlantDoc qua cổng score ≥75%, khác Unsupported | Parent | Sau train | Yêu cầu |
|---|---:|---:|---:|
| Accuracy trên nhóm qua cổng | 66.67% | 66.18% | ≥85% |
| Coverage trên toàn PlantDoc | 18.84% | 16.43% | ≥25% |

[Kết luận tiêu chí phát hành](release_decision.json), [tóm tắt và hash](summary.json), [lịch sử train](training_history.json), [precision/recall/F1 từng lớp](per_class.csv).

JSON đầy đủ: [V2](baseline_validation.json), [parent](parent_validation.json), [sau train](trained_validation.json). Bao gồm confusion matrix, calibration và support từng lớp. Support = 0 nghĩa là chưa được đo cho lớp đó trong domain.

Nhãn còn là nhãn nguồn; chưa có kiểm chứng chuyên gia của project hoặc bộ test nông hộ Việt Nam độc lập mới. Tăng dữ liệu và giảm train loss chưa chứng minh nhận diện bệnh thực địa tốt hơn. Không dùng test để chọn model.

## Diễn giải kết quả

Accuracy toàn PlantDoc tăng từ 75,85% lên 77,05%, nhưng số này gồm cả Unsupported. F1 các lớp bệnh đã biết giảm từ 54,98% xuống 51,38%. Lượt này chưa chứng minh cải thiện nhận diện bệnh; model mặc định được giữ nguyên.

Thí nghiệm đồng thời bổ sung nguồn, loại một phần train cũ và fine-tune 9 block. Đây chưa phải ablation tách riêng ảnh hưởng từng nguồn, nên không quy kết kết quả giảm cho một bộ dữ liệu cụ thể.

`run_alternative_demo.cmd` mở bản thử nghiệm ở http://127.0.0.1:8004/app sau khi đối chiếu ONNX đạt. Model cũ ở http://127.0.0.1:8003/app. Weights trial: `artifacts/training_runs/alternative_sources_20261103/model.pt` và graph cùng tên `.onnx`; hai file là cùng một model.

## Tạo lại dữ liệu ở phiên bản khác

Cần có manifest parent `field_mobile_v2` và ảnh nguồn của parent trên máy. Lấy metadata V1 của nhà phát hành, tải Ecuador, rồi tạo output mới:

```powershell
New-Item -ItemType Directory artifacts/retraining_sources_20261103 -Force
Invoke-WebRequest https://data.mendeley.com/public-api/datasets/bwh3zbpkpv -OutFile artifacts/retraining_sources_20261103/ccmt_metadata.json
Invoke-WebRequest https://data.mendeley.com/public-api/datasets/d5b3fzpw3g -OutFile artifacts/retraining_sources_20261103/bari_metadata.json
.venv/Scripts/python.exe -m ml.download_corn_ecuador
.venv/Scripts/python.exe -m ml.prepare_alternative_field --output data/processed/field_alternative_next --ccmt-metadata artifacts/retraining_sources_20261103/ccmt_metadata.json --bari-metadata artifacts/retraining_sources_20261103/bari_metadata.json
```

Bộ chuẩn bị yêu cầu đúng ID/version 1, kiểm checksum và không ghi đè output đã khóa. Nếu nhà phát hành đổi version, cần xem lại metadata trước khi sử dụng. Nhãn cần chuyên gia duyệt vẫn là nhãn cần duyệt, kể cả khi kiểm tra file và đối chiếu runtime đều đạt.
