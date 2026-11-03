# Kế hoạch hoàn thiện dự án và huấn luyện model để nộp ngày 01 tháng 11 năm 2026

Ngày lập: 31/10/2026. Múi giờ: Asia/Saigon. Phạm vi: hệ thống SIC Group 5, 3 cây trồng, 11 lớp.

Mục tiêu là hoàn thiện bản nộp theo Action Plan, giữ một bản demo hoạt động ổn định và chạy thêm một thí nghiệm GPU có kiểm soát. Model hiện tại đã vượt mục tiêu macro F1 theo báo cáo lưu trong repo; train mới phục vụ đối chứng và cải thiện, không phải điều kiện để giữ được bản demo. Chưa biết giờ nộp cụ thể, vì vậy các khoảng thời gian dưới đây tính từ lúc bắt đầu triển khai, phải dừng train mới tối thiểu 6 giờ trước giờ nộp thực tế.

## 1. Căn cứ và hiện trạng

Nguồn yêu cầu: `D:/AI_SIC_ICTU_2026/Final_SIC_AI_2026/Reports/SIC_AI_Capstone Project_Action Plan.docx`.

Tài liệu yêu cầu MobileNetV2, baseline CNN tự xây, train/validation/test khoảng 70/15/15, confidence được hiệu chỉnh, macro F1 ít nhất 85%, inference trung bình dưới 200 ms, web có camera và upload, tư vấn được kiểm chứng, báo cáo cuối, slide và video demo. Giảng viên yêu cầu trình bày rõ hai subsystem, có căn cứ chuyên môn cho tư vấn và cập nhật abstract bằng kết quả thực nghiệm.

| Hạng mục | Bằng chứng hiện có | Việc cần chốt |
|---|---|---|
| Dữ liệu | 11.494 ảnh sau loại trùng; 8.043 train, 1.731 validation, 1.720 test theo báo cáo audit | Kiểm tra hash và đường dẫn; giữ nguyên split; ghi giới hạn leaf-map dự phòng |
| MobileNetV2 | Baseline pretrained ImageNet, backbone đóng băng, train classifier bằng feature cache; có temperature scaling | Mô tả đúng chiến lược; phân biệt với fine-tune backbone |
| Chất lượng baseline | Test accuracy 94,1860%, macro F1 93,7632% trong `results/evaluation.json` | Xác nhận checkpoint và manifest khớp báo cáo; đây là kết quả cũ, chưa đo lại trong lượt lập plan |
| Thí nghiệm cũ | Linear candidate test F1 93,3147%; projection shuffle chỉ tăng validation khoảng 0,0424 điểm phần trăm | Giữ baseline; không chạy lại cùng cấu hình chỉ để tìm điểm đẹp hơn |
| CNN tự xây | Chưa tìm thấy implementation/nhật ký đối chứng trong phần mã đã kiểm tra | Thêm baseline nhỏ nếu kịp; nếu thiếu phải ghi rõ là mục chưa hoàn thành trong Action Plan |
| Suy luận | Có eager, TorchScript, ONNX FP32; kiểm tra metadata và hash checkpoint | Chốt runtime trên CPU phục vụ; train GPU không bắt buộc serving GPU |
| Hiệu năng HTTP | Báo cáo ONNX, 2 threads, 2 slots: mean/p95 19,28/35,63 ms ở 1 client; 81,31/93,96 ms ở 8 client | Xác nhận lại ở bản nộp nếu code/model/môi trường serving thay đổi; tách inference với HTTP latency |
| Backend | Có health, diagnose, danh mục cây/bệnh; giới hạn upload; xử lý confidence riêng từng request | Chạy test, xử lý đúng trạng thái chưa có model, thống nhất cách khởi động |
| Frontend | HTML demo gọi API, xem ảnh, confidence và tư vấn | Bổ sung luồng camera, validation phía client, panel mở/đóng, trạng thái model chưa sẵn sàng |
| Tri thức tư vấn | Có JSON cho các lớp, triệu chứng, canh tác, hóa chất và disclaimer | Chưa thấy nguồn dẫn/biên bản kiểm chứng theo từng record; đây là phần còn thiếu quan trọng |
| Hồ sơ nộp | Có báo cáo Data & Optimization và tài liệu API | Chưa thấy bộ report tổng hợp, slide và video trong project này; kiểm tra thêm thư mục Reports của nhóm khi triển khai |

Số liệu và quyết định cũ tham chiếu `docs/data_optimization/results/evaluation.json`, `results/optimization_20261017/api_benchmark_confirmed.json`, `results/optimization_20261024/decision.json` và `results/optimization_20261024_shuffle/decision.json`.

Phần dữ liệu trong Action Plan ghi Kaggle, nhưng pipeline thực tế dùng repository `spMohanty/PlantVillage-Dataset` được ghim commit. Báo cáo cuối phải nêu nguồn thực tế và provenance, không giữ mô tả Kaggle nếu nhóm không dùng nguồn đó.

## 2. Tài nguyên GPU đã kiểm tra

- GPU: NVIDIA GeForce RTX 2050, 4.096 MiB VRAM; driver 610.78, đọc bằng `nvidia-smi` ngày 31/10/2026.
- Python trong `.venv`: import được `torch 2.14.1+cpu`, `torch.version.cuda = None`, `torch.cuda.is_available() = False`.
- Cần tạo môi trường train CUDA riêng, ví dụ `.venv-gpu`, để giữ môi trường demo hiện tại. Chọn cặp torch/torchvision và bản CUDA wheel tương thích Python, Windows, driver và GPU từ hướng dẫn chính thức; lưu lại phiên bản thực tế sau cài đặt.
- Cổng kiểm tra môi trường: CUDA available, nhận đúng tên GPU, chạy thử tensor CUDA và một bước forward/backward MobileNetV2 với loss hữu hạn.
- Chưa đo thời gian mỗi epoch và VRAM lúc train; mọi budget dưới đây là giới hạn công việc, không phải dự đoán thời gian train chắc chắn.

## 3. Thứ tự thực hiện trước hạn nộp

Các công việc của thành viên có thể làm đồng thời. Bảng phân công là đề xuất theo Action Plan, không có nghĩa đã giao việc hoặc đã gửi thông báo cho thành viên.

| Giai đoạn | Budget kể từ lúc bắt đầu | Phụ trách đề xuất | Công việc và đầu ra bắt buộc |
|---|---|---|---|
| A. Giữ bản hiện tại và kiểm tra | 0–1 giờ | Khánh + Tâm | Lưu hash checkpoint/ONNX, git status, kết quả test ban đầu; giữ nguyên split và các file đã sửa sẵn |
| B. Chuẩn bị GPU | 1–2 giờ | Bảo + Tâm | Môi trường CUDA riêng; smoke test; đo 100 batch để ước lượng runtime và chọn batch |
| C. Đối chứng và fine-tune | 2–7 giờ, dừng sớm nếu sát hạn | Bảo + Tâm | Baseline CNN nhỏ và tối đa một cấu hình MobileNetV2 fine-tune; checkpoint riêng, history, cấu hình, calibration và validation đầy đủ |
| D. Hoàn thiện web/API | 1–7 giờ | Hải + Khánh | Camera/upload, validation, trạng thái tải/lỗi/UNCERTAIN/healthy, panel tư vấn; phục vụ frontend và API cùng origin nếu thuận tiện |
| E. Kiểm chứng tư vấn | 1–7 giờ | Hải, Khánh hỗ trợ schema | Bảng nguồn và người kiểm tra cho từng bệnh; rà soát hóa chất, liều lượng, PHI, tên thương mại; bỏ hoặc ẩn nội dung chưa xác minh |
| F. Chốt bản tích hợp | 7–9 giờ | Khánh + Tâm | Chọn checkpoint theo quy trình ở mục 5; export/kiểm tra ONNX nếu có model mới; test và benchmark bản nộp |
| G. Report, slide, video | Soạn từ giờ 1; chốt giờ 9–12 | Bảo điều phối, cả nhóm cung cấp phần mình | Abstract có số liệu cuối, report tổng hợp, slide, video demo và hướng dẫn chạy trên máy khác |
| H. Dự phòng | Phần thời gian còn lại trước giờ nộp | Cả nhóm | Chạy thử từ đầu, sửa lỗi cản trở demo, sao lưu bộ nộp; không mở thêm thí nghiệm |

Nếu không đủ budget: giữ model baseline và ưu tiên D, E, F, G. Nếu CNN đối chứng không hoàn tất thì ghi thiếu hạng mục này; không tuyên bố đáp ứng đầy đủ phương pháp trong Action Plan. Không trì hoãn bộ nộp để chờ một model tốt hơn.

## 4. Kế hoạch train trên RTX 2050 4 GB

### 4.1 Giữ dữ liệu và kết quả so sánh nhất quán

1. Khóa danh sách 11 lớp và thứ tự index theo `backend/config.py`.
2. Dùng đúng manifests hiện có. Không xáo lại split, không đưa test vào train và không ghi đè thư mục processed hiện tại.
3. Tái sử dụng `ManifestDataset` và preprocessing chung; augmentation chỉ cho train, validation/test giữ deterministic.
4. Kiểm tra hash ảnh, manifest và checkpoint. Ghi rõ audit không chứng minh tuyệt đối mọi lá vật lý đã tách biệt vì một phần dùng group dự phòng.
5. Lưu mỗi thí nghiệm với ID, seed, cấu hình, optimizer, scheduler, epoch tốt nhất, log và file checkpoint riêng. Lưu state để resume; hỗ trợ marker `artifacts/PAUSE_REQUESTED`, không tự xóa marker cũ.

### 4.2 Bổ sung tối thiểu cho mã train

- Tạo entry point `ml/train_gpu.py` dùng lại dataset, metrics và calibration hiện có; tham số `--architecture small_cnn|mobilenet_v2`, `--device`, `--output`, `--epochs`, `--batch-size`, `--lr`, `--patience`, `--seed` và cấu hình layer được mở khóa.
- Không chuyển nguyên script projection cache sang GPU một cách máy móc: script hiện tại tạo model/tensor trên CPU và tối ưu phần đuôi từ activation cache. Nhánh mới cần đọc ảnh để backprop qua các block được mở khóa.
- Batch ban đầu 16 ở 224×224; nếu OOM giảm 8 rồi 4. Chọn sau smoke test; validation cũng có batch riêng nếu cần. Windows bắt đầu `num_workers=0`, chỉ tăng khi đo thấy có lợi.
- Dùng autocast FP16 và GradScaler cho train GPU; validation chọn model bằng FP32. Gradient clipping nếu có phải thực hiện sau unscale. Lưu cả scaler, optimizer, scheduler và RNG khi resume.
- Save tensor checkpoint về CPU, kèm đầy đủ metadata giống hợp đồng hiện tại, để backend CPU load được.
- Thêm kiểm tra nhỏ cho layer được train/đóng băng, checkpoint reload và một batch train. Không xây thêm framework training tổng quát.

Các entry point và tham số mới ở mục này là việc cần triển khai, chưa có sẵn để chạy.

### 4.3 Thí nghiệm A — CNN tự xây để đối chứng

- Mục đích: bổ sung hạng mục custom convolutional baseline trong Action Plan.
- Kiến trúc đề xuất: 3 block Conv/ReLU/MaxPool, số kênh 16/32/64, AdaptiveAvgPool và Linear ra 11 lớp; không pretrained.
- Dùng class-weighted cross-entropy, Adam LR 1e-3, cosine scheduler, tối đa 15 epoch, early stopping validation loss với patience 3, seed 42.
- Giới hạn khoảng 60–90 phút sau smoke test. Không cần vượt MobileNetV2; ghi đúng hiệu quả và learning curve, kể cả khi chưa hội tụ.
- Đánh giá trước trên validation; chỉ đánh giá test khi cấu hình và checkpoint đối chứng đã khóa. Không tìm thêm cấu hình bằng kết quả test.

### 4.4 Thí nghiệm B — Fine-tune MobileNetV2

- Khởi tạo từ checkpoint baseline đang triển khai để tận dụng phần đã học; lưu đầu ra mới, không ghi đè baseline.
- Mở khóa `features[17]`, `features[18]` và classifier; các block trước tiếp tục frozen. Giữ BatchNorm ở eval để không cập nhật running statistics trong lượt batch nhỏ này.
- LR khởi đầu cho phần backbone mở khóa: 1e-5; classifier: 1e-4. Đây là cấu hình thử đầu, chưa phải cấu hình đã chứng minh tốt hơn.
- Adam, weight decay 1e-4, cosine scheduler; class-weighted cross-entropy với cùng cách tính trọng số của baseline để giảm số yếu tố thay đổi.
- Tối đa 10 epoch, patience 3 theo validation loss; lưu checkpoint loss tốt nhất và báo cáo đồng thời macro F1 theo epoch.
- Chỉ chạy một cấu hình trước hạn nộp. Nếu loss không hữu hạn, lỗi dữ liệu hoặc runtime vượt budget, lưu log/checkpoint rồi dừng; không mở search dài.
- Sau khi chọn checkpoint, hiệu chỉnh temperature bằng validation, báo cáo NLL/ECE trước và sau. Ngưỡng 0,75 là chính sách gating hiện tại, không phải cam kết rằng 75% confidence tương đương 75% độ đúng ở thực địa.
- Kiểm tra riêng Corn_Gray_leaf_spot và Tomato_Early_blight vì F1 trong báo cáo test cũ còn thấp hơn các lớp khác. Phân tích lỗi trên validation; không dùng các lỗi test cũ để chọn thêm hyperparameter.

## 5. Quy tắc chọn model và đánh giá cuối

Test hiện có đã được xem trong các vòng trước. Vì vậy không gọi nó là holdout hoàn toàn chưa từng sử dụng cho dự án. Trong vòng này, khóa cấu hình trước khi chạy test; kết quả test dùng để báo cáo và kiểm tra hồi quy phát hành, không để điều chỉnh tiếp.

1. So sánh baseline và ứng viên trên full validation FP32, cùng manifest và preprocessing. Quy tắc đề xuất cho lượt sát hạn: ứng viên phải tăng ít nhất 0,5 điểm phần trăm macro F1 và không giảm quá 2 điểm phần trăm recall ở bất kỳ lớp nào. Đây là ngưỡng vận hành do nhóm đề xuất, không phải kết luận có ý nghĩa thống kê.
2. Nếu không đạt: giữ baseline, vẫn ghi đầy đủ kết quả thử nghiệm. Không cần đọc lại test cho ứng viên bị loại bằng validation.
3. Nếu đạt: khóa weights, temperature và cấu hình; chạy test một lần, báo cáo accuracy, macro precision/recall/F1, từng lớp, confusion matrix và calibration. Không chỉnh hyperparameter theo kết quả này.
4. Với bản phát hành, yêu cầu test macro F1 ít nhất 85% và không thấp hơn baseline đã lưu; kiểm tra NLL/ECE không xấu đi đáng kể và công bố số cụ thể. Nếu không đạt, giữ bản baseline; ghi đây là kiểm tra hồi quy trên test cũ, không phải bằng chứng generalization độc lập mới.
5. Export ONNX vào file ứng viên riêng, kiểm tra metadata/hash, so sánh logits và nhãn với eager. Sau khi chấp nhận model mới phải cập nhật đồng bộ PT và ONNX rồi kiểm tra health; không thay riêng một file vì engine kiểm tra hash.
6. Đo inference CPU và HTTP với cấu hình phục vụ đã chốt. Tiêu chí gốc: inference mean <200 ms; thêm mục tiêu nội bộ HTTP p95 <200 ms ở 1/4/8 client trên máy demo. Ghi số request, warmup, runtime, threads, concurrency và phạm vi phép đo.
7. Nếu có ảnh thực địa độc lập kịp thời: tạo tập kiểm tra riêng, ghi nguồn, nhãn và số mẫu; báo cáo tách PlantVillage. Không nhập vào train sau khi xem kết quả. Nếu không có, nêu rõ chưa đo generalization ngoài thực địa.

## 6. Hoàn thiện sản phẩm và hồ sơ tư vấn

- Frontend: luồng chụp camera trên điện thoại, preview/đổi ảnh, kiểm tra JPEG/PNG và 10 MB trước upload, xử lý lỗi decode, đang chờ request và tránh gửi lặp. Kiểm thử camera trên thiết bị thực tế; giải pháp dùng camera trực tiếp qua trình duyệt phải có ngữ cảnh truy cập phù hợp.
- Tách trạng thái server online khỏi model sẵn sàng; hiện đúng MODEL_NOT_READY và lỗi 503. Hiển thị tiếng Việt, healthy/UNCERTAIN và disclaimer rõ ràng.
- Cố định ngưỡng tư vấn 0,75 cho luồng người dùng; nếu giữ custom threshold thì đưa vào phần thử nghiệm để tránh người dùng hạ ngưỡng rồi hiểu thành kết luận chắc chắn.
- Bổ sung panel tư vấn mở/đóng; low confidence không hiển thị hóa chất. Hiển thị đủ hướng dẫn canh tác và an toàn; kiểm tra nhánh healthy không xuất thuốc vô cớ.
- Tạo bảng kiểm chứng tri thức: disease_id, nội dung được đối chiếu, tổ chức/tài liệu, URL hoặc trang, ngày truy cập, người kiểm tra, trạng thái. Tên FAO/USDA/Viện BVTV trong README chưa thay thế được nguồn cụ thể cho mỗi hướng dẫn.
- Đối chiếu riêng hoạt chất, nồng độ chế phẩm, cây/bệnh áp dụng, liều lượng và PHI. Không suy ra một liều chung từ tên hoạt chất hoặc dùng cùng liều cho các sản phẩm khác nồng độ. Ẩn phần chưa xác minh trước bản demo chính thức.
- Chạy API và frontend theo một hướng dẫn duy nhất; ưu tiên cùng origin và địa chỉ API tương đối để tránh hard-code localhost khi demo bằng điện thoại/máy khác.

## 7. Kiểm thử và bộ nộp

Kiểm thử bắt buộc sau thay đổi:

- `python -m unittest discover -s tests -v`, `python tests/test_backend.py`, `python tests/stress_and_edge_test.py` bằng môi trường phục vụ bản nộp.
- Thêm kiểm tra cần thiết cho logic mới: checkpoint GPU lưu về CPU, freeze layer, frontend/model chưa sẵn sàng và nguồn tư vấn theo schema mới.
- Demo end-to-end: một lá khỏe, một ảnh bệnh đủ confidence, một trường hợp UNCERTAIN được xác nhận thực tế; thêm file hỏng, file quá dung lượng, server offline và camera bị từ chối nếu dùng getUserMedia. Không gán sẵn kết quả đúng cho ảnh chỉ vì tên file.
- Nếu API khó phát sinh UNCERTAIN ổn định, dùng fixture riêng trong test tự động; video demo phân biệt rõ fixture và inference thật.
- Kiểm tra ảnh mẫu là ảnh lá thật trước khi dùng để chứng minh chất lượng model; dữ liệu hình học/tổng hợp chỉ dùng kiểm thử giao thức.

Ví dụ lệnh đánh giá bản baseline hiện có, phải dùng output mới để giữ kết quả cũ:

```powershell
python -m ml.evaluate --split validation --weights models/mobilenetv2_crop_disease.pt --output docs/final_submission_20261101/baseline_validation.json
python -m ml.api_benchmark --runtimes onnx --threads 2 --slots 2 --concurrency 1 4 8 --runs 150 --output docs/final_submission_20261101/api_benchmark.json
```

Lệnh benchmark HTTP hiện dùng checkpoint mặc định; nếu đánh giá ứng viên chưa triển khai phải bổ sung đường dẫn weights/graph trước, không suy diễn kết quả benchmark baseline cho ứng viên.

Bộ nộp cần có:

1. Source và hướng dẫn cài/chạy đã kiểm tra; phân biệt môi trường train GPU và serving CPU, ghi rõ dependencies.
2. Weights, ONNX, class mapping, preprocessing contract và SHA-256. Các file model đang bị gitignore nên phải có gói bàn giao hoặc vị trí tải riêng; không xem việc có source trên Git là đã giao model.
3. Báo cáo dữ liệu: nguồn thực tế, split, chống trùng, phân bố lớp và giới hạn audit.
4. Bảng đối chứng CNN nhỏ/baseline MobileNetV2/fine-tune, learning curve, confusion matrix, calibration và benchmark có phạm vi đo.
5. Hồ sơ kiểm chứng tư vấn, tài liệu API và kết quả kiểm thử hệ thống.
6. Báo cáo capstone tổng hợp, slide, video demo; cập nhật abstract bằng số liệu của model thực sự được chọn, đúng yêu cầu giảng viên.
7. Mô tả hạn chế: PlantVillage nền kiểm soát, chưa chứng minh hiệu quả ngoài đồng, ảnh ngoài 11 lớp có thể vẫn nhận confidence cao; confidence gate hiện không phải bộ phát hiện ảnh ngoài miền.

## 8. Các quyết định đã chốt và phạm vi lượt này

- Người dùng đã chọn hạn nộp 01/11/2026 và train bằng GPU NVIDIA trên máy này.
- Chỉ lập kế hoạch và kiểm tra tài nguyên trong lượt này; chưa cài CUDA wheel, sửa code train hoặc bắt đầu job train.
- Giữ baseline làm bản dự phòng. Không thêm cây/lớp mới, chatbot, tài khoản, database server hoặc kiến trúc frontend mới trước hạn.
- Giữ nguyên các thay đổi có sẵn ở `docs/data_optimization/REPORT.md`, `results/benchmark.json`, `results/environment.json`.
- Chưa kiểm thử lại API, chưa xác minh nguồn nông học và chưa kiểm tra bố cục Word bằng render trong lượt lập kế hoạch; nội dung kế hoạch được trích từ paragraph và bảng OOXML.

## 9. Tham chiếu kỹ thuật chính thức

- [PyTorch Start Locally](https://pytorch.org/get-started/locally/): chọn bản cài CUDA phù hợp khi triển khai.
- [PyTorch Transfer Learning Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial): phân biệt feature extractor và fine-tuning.
- [PyTorch Automatic Mixed Precision Examples](https://docs.pytorch.org/docs/main/notes/amp_examples.html): autocast và GradScaler cho huấn luyện mixed precision.

Các batch, LR, epoch, budget và quy tắc chọn model trong kế hoạch là cấu hình đề xuất cho project này; tài liệu tham chiếu không đảm bảo chúng tối ưu trên RTX 2050 hoặc đạt chất lượng cao hơn baseline.
