# Hai model đang giữ trong project

| Vai trò | Trọng số | Runtime |
|---|---|---|
| Phục vụ mặc định | `mobilenetv2_crop_disease.pt` | PyTorch CPU/CUDA |
| Graph của cùng baseline | `mobilenetv2_crop_disease.onnx` | ONNX CPU |
| Candidate nghiên cứu | `candidate/mobilenetv3_field.pt` | PyTorch CPU/CUDA |
| Graph của cùng candidate | `candidate/mobilenetv3_field.onnx` | ONNX CPU |

File `.json` cùng tên lưu lịch sử/config train. PT và ONNX của một model dùng cùng trọng số, không phải hai lần train độc lập. Mã mạng/forward nằm trong `backend/inference_engine.py`, `backend/mobile_model.py` và `backend/field_inference.py`.

Baseline là V2, 11 lớp. Candidate là V3-Large, 11 lớp SIC + Unsupported. Candidate được chọn từ `mobilenetv3_codex_collab_20261103.pt`, SHA `0eeb20ec4ef5f3f11cd477920ccc63f33e2b721bb37e394900467f64dc284577`. Các lượt warm-start không giảm loss so với parent nên trọng số mạng được giữ; temperature khoảng 1,005059. Những file metadata khác nhau không chứng minh model nhận diện tốt hơn.

Candidate có F1 PlantDoc CPU 54,98%, score-gate accuracy 66,67%, coverage 18,84%. Chưa qua tiêu chí phát hành, không tự thay baseline. [Benchmark](../docs/BENCHMARKS.md).

## Lịch sử và đầu ra mới

Model cũ/progress/resume được chuyển sang `Reports/SIC_Model_Archive_20261103_222709`, ngoài project. Manifest ở đó ghi tên cũ/mới và SHA. Một số PT có cùng tensor nhưng khác calibration/metadata nên hash cả file vẫn khác; chúng được giữ để truy nguyên thí nghiệm.

Lượt train mới ghi vào `artifacts/training_runs/<run>/model.pt`. Chỉ đưa một ứng viên đã chọn vào `models/candidate/` sau khi kiểm chất lượng và runtime. Gói demo chỉ chứa hai model hiện tại. Weights không lưu trong Git; bàn giao riêng. [Lệnh CPU/GPU và native libraries](../docs/RUNTIME.md).
