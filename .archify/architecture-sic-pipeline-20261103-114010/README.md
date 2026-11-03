# Hiểu pipeline SIC qua ba sơ đồ

Điện thoại gửi ảnh đến API. Model trên máy chủ dùng trọng số đã học để phân loại ảnh, sau đó hệ thống tra thông tin chăm sóc và hiển thị kết quả.

Mở lần lượt các file HTML bằng Chrome hoặc Edge. Chúng hoạt động độc lập, không cần Internet để xem sơ đồ.

1. [Toàn bộ hệ thống: từ điện thoại đến kết quả](he-thong.html).
2. [Huấn luyện: tải dữ liệu, chia tập và học trọng số](huan-luyen.html).
3. [Suy luận: từ pixel ảnh đến điểm dự đoán](suy-luan.html).

Nội dung sơ đồ bằng tiếng Việt. Thanh công cụ của Archify dùng tiếng Anh; có thể dùng `+` để phóng to, `Still` để dừng chuyển động, `Export` để xuất hình. Bấm một nút trong sơ đồ để xem phần mã nguồn liên quan.

## Model học bằng cách nào?

Một ảnh train đi kèm nhãn do bộ dữ liệu cung cấp, chẳng hạn `Corn_Common_rust`. Máy tính giải mã ảnh thành các số biểu diễn pixel RGB. CNN biến các số này thành đặc trưng, đầu phân loại tạo điểm cho từng lớp.

Hàm loss so sánh đầu ra với nhãn có sẵn. Backprop tính cách thay đổi trọng số để giảm loss, rồi AdamW cập nhật trọng số. Quá trình lặp lại trên nhiều batch. Một epoch là một lượt lấy mẫu train; ở pipeline mới, bộ lấy mẫu có trọng số có thể chọn lặp ảnh để cân bằng miền dữ liệu và lớp.

Trọng số là các con số trong mạng. File `.pt` lưu những con số đó cùng thứ tự nhãn và hợp đồng tiền xử lý. Model học mối liên hệ thống kê giữa ảnh và nhãn, không lưu một danh sách ảnh để tìm ảnh giống nhất khi bạn tải ảnh mới.

SIC dùng transfer learning: bắt đầu từ MobileNet đã học trên ImageNet, sau đó học cho bài toán lá cây. V3 mới giữ cố định phần CNN trong giai đoạn đầu để học đầu phân loại, rồi mở ba khối cuối để fine-tune. GPU NVIDIA của máy được dùng cho train; API hiện chạy model trên CPU.

Khi bạn bấm **Phân tích ảnh**, hệ thống thực hiện suy luận. Trọng số giữ nguyên. Ảnh tải lên không tự trở thành dữ liệu train và không tự làm model học thêm.

## Hai model trong project

| Nội dung | MobileNetV2 | MobileNetV3-Large mới |
|---|---|---|
| Trạng thái | Mặc định của API | Ứng viên thử nghiệm, chưa đạt tiêu chí thay bản mặc định |
| Số đầu ra | 11 lớp trong phạm vi | 11 lớp cũ và `Unsupported` |
| Dữ liệu | PlantVillage | PlantVillage kết hợp PlantDoc đã lọc |
| Chuẩn bị hình | Resize trực tiếp 224 × 224 | Giữ tỷ lệ, thêm viền xám thành 224 × 224 |
| Đầu phân loại V3 | Không áp dụng | 960 → 256 → 12, Hardswish và Dropout khi train |
| Runtime API | ONNX nếu có graph hợp lệ, có lựa chọn PyTorch | PyTorch CPU, bật bằng `SIC_FIELD_MODEL_PATH` |

API chọn một model khi khởi động. V3 không chạy sau V2 để sửa kết quả của V2. `Unsupported` là lớp được học từ một số ảnh ngoài phạm vi, chưa phải bộ phát hiện mọi ảnh lạ.

11 lớp cũ gồm ngô gỉ sắt, đốm xám, cháy lá phía Bắc, ngô khỏe; khoai tây cháy sớm, mốc sương, khoai tây khỏe; cà chua cháy sớm, mốc sương, đốm Septoria và cà chua khỏe. Bộ nhãn này chưa bao gồm mọi sâu hại, thiếu dinh dưỡng hoặc tổn thương cơ học.

## Dữ liệu được tải và chia như thế nào?

Script tải file từ GitHub tại revision cố định, đối chiếu hash rồi lưu trong `data/raw/`. File `source.json` ghi nguồn. Tên file PlantDoc không hợp lệ trên Windows được thay bằng hash; thông tin tên gốc vẫn được giữ.

Pipeline kiểm tra ảnh đọc được, kích thước, nhãn và ảnh gần trùng. Nhóm PlantDoc có nhãn xung đột bị loại. Nhãn chung `Corn leaf blight` cũng bị loại vì chưa đủ cơ sở coi là cháy lá phía Bắc. Nhãn gỉ ngô được ánh xạ sang lớp gỉ sắt của project nhưng còn giới hạn về loài tác nhân.

`records.json` là danh mục đường dẫn, nhãn, hash và split. Khi train, DataLoader đọc ảnh theo danh mục này; ảnh không được chép vào checkpoint. Chỉ ảnh train được tăng cường bằng lật, biến đổi hình nhẹ và đổi màu/sáng. Validation và test không dùng tăng cường ngẫu nhiên.

| Bộ dữ liệu sau kiểm tra | Train | Validation | Test |
|---|---:|---:|---:|
| PlantVillage | 8.043 | 1.731 | 1.720 |
| PlantDoc | 1.658 | 414 | 217 |

Train cập nhật trọng số. Validation chọn checkpoint và học một số temperature `T` để hiệu chỉnh softmax, không sửa CNN bằng gradient validation. Test dùng để đo sau khi chốt model và hiệu chỉnh. Không dùng test để chọn epoch hoặc chỉnh ngưỡng.

PlantDoc là ảnh thu thập trên Internet với nhãn từ nguồn dữ liệu. Nhóm ảnh gần trùng đã được xử lý bằng hash cảm nhận; kiểm tra này chưa chứng minh mọi lá vật lý đều tách biệt, cũng chưa thay cho việc chuyên gia tại địa phương kiểm tra nhãn. Xem [audit dữ liệu](evidence/data_audit.json).

## Crop, resize và chuẩn hóa ảnh khác nhau ở đâu?

Crop lấy hình chữ nhật bạn chọn và bỏ phần bên ngoài. Bạn kéo vùng chọn hoặc nhập tọa độ trong web, bấm áp dụng, rồi xem lại ảnh. Web gửi chính file ảnh sau crop đang hiển thị. Nút khôi phục lấy lại ảnh gốc. Hãy giữ đủ phần lá có tổn thương, tránh cắt mất vết bệnh.

Web hiện chưa có detector tìm lá hay segmentation tách nền. Client khác có thể gửi ảnh gốc kèm `crop_box=x,y,width,height` từ 0 đến 1 cho API; web hiện crop trước khi gửi, không đồng thời áp dụng crop_box ở server. Mỗi cạnh vùng cắt phải còn ít nhất 32 pixel.

Resize chuẩn bị kích thước đầu vào cho CNN. V2 kéo ảnh về 224 × 224 nên tỷ lệ hình có thể thay đổi. V3 giữ tỷ lệ ảnh và thêm viền xám, gọi là letterbox. Ví dụ ảnh rộng 448 × 224 được thu còn 224 × 112 rồi thêm viền trên và dưới; đây là ví dụ hình học, không phải số đo ảnh bạn gửi.

Server xử lý hướng ảnh từ EXIF và chuyển RGB. V3 có thể giải mã JPEG ở độ phân giải giảm trước để tiết kiệm bộ nhớ. Pixel được đổi thành tensor, chuẩn hóa bằng mean/std ImageNet. Đầu vào cuối cùng là `1 × 3 × 224 × 224`: một ảnh, ba kênh màu, cao và rộng 224 pixel.

Web/API nhận JPG và PNG, tối đa 10 MB, tối đa 20 megapixel. Ảnh HEIC cần chuyển sang JPG/PNG trước. Crop giúp giảm nền thừa nhưng không bảo đảm sửa một dự đoán sai do thiếu lớp bệnh trong dữ liệu.

## CNN đánh giá ảnh thế nào?

Các lớp tích chập tạo bản đồ đặc trưng bằng các bộ lọc đã học. Những lớp sau kết hợp tín hiệu từ lớp trước. Pooling gom các tín hiệu thành vector, rồi đầu phân loại tạo `logits`, tức các điểm thô cho từng lớp.

Softmax tính `p_i = exp(z_i / T) / Σ exp(z_j / T)`, với `z_i` là logit của lớp i. Các điểm cộng lại bằng 1. Hệ thống lấy lớp có điểm cao nhất. `T` được học từ validation để điều chỉnh độ tập trung của điểm số; nó không thêm nhãn mới và không bảo đảm phát hiện ảnh ngoài phạm vi.

Điểm 99,99% không có nghĩa ảnh đó chắc chắn đúng 99,99%. Nếu mọi lựa chọn đều chưa phù hợp, model vẫn có thể nghiêng rất mạnh về một lớp. Nền ảnh và ánh sáng khác ảnh train cũng có thể làm model sai.

Phần lời khuyên được tra theo nhãn trong `data/advisory_knowledge_base.json`. Đây là nội dung đã soạn sẵn. CNN không tự viết lời khuyên và hệ thống này không dùng mô hình ngôn ngữ để chẩn đoán.

## Lỗi trong screenshot của bạn

Bản trích vùng lá từ screenshot bị V2 dự đoán `Corn_healthy` với điểm gần 100%. V3 mới dự đoán `Corn_Gray_leaf_spot` ở khoảng 49,59%, dưới ngưỡng 75%. Không thể coi nhãn mới này là chẩn đoán đúng: nhóm chưa có ảnh gốc hoặc nhãn chuyên gia cho trường hợp đó.

Screenshot chỉ dùng để tái hiện định tính, không dùng train hoặc tính accuracy. Giao diện hiện ghi rõ đây là điểm dự đoán. Với ảnh người dùng, nhãn khỏe luôn chuyển sang yêu cầu kiểm tra; nếu bạn đánh dấu lá có tổn thương thì có cảnh báo xung đột cụ thể. Hệ thống không còn dùng điểm cao để xác nhận lá thực địa khỏe.

## Đọc kết quả kiểm định mới

Checkpoint khôi phục tốt nhất là `finetune/2`. Phiên làm việc bị ngắt giữa epoch 3; trọng số chưa lưu của epoch này đã bỏ, không tính là epoch hoàn tất. Checkpoint đã được hiệu chỉnh bằng validation, sau đó đo test một lần với cấu hình cố định.

| Macro F1 trên các lớp trong phạm vi có mẫu | V2 | V3 |
|---|---:|---:|
| PlantVillage validation | 94,48% | 97,96% |
| PlantDoc validation | 15,95% | 45,90% |
| PlantVillage test | 93,76% | 97,53% |
| PlantDoc test | 15,31% | 53,54% |

F1 kết hợp precision và recall; macro F1 lấy trung bình theo lớp để lớp đông ảnh không che hết lỗi lớp ít ảnh. PlantDoc validation chỉ có 8 lớp trong phạm vi xuất hiện, nên chỉ số này không đại diện đủ 11 lớp. Ma trận nhầm lẫn vẫn tính cả các đầu ra khác, kể cả `Unsupported`.

V3 cải thiện F1 nhưng chưa qua cổng triển khai: trên validation PlantDoc, 80/414 ảnh có điểm ≥75% và nhãn khác Unsupported (coverage 19,32%); chỉ 44/80 dự đoán trong nhóm đó đúng (55%). Tiêu chí đã đặt trước yêu cầu coverage ít nhất 25% và độ chính xác nhóm này ít nhất 85%. Đây là cổng điểm thô trước chính sách xem lại nhãn khỏe, không phải tỷ lệ API xác nhận bệnh. Xem [quyết định giữ bản mặc định](evidence/release_decision.json).

Cần thêm ảnh đúng phạm vi từ điện thoại, có nhãn chuyên gia, đặc biệt các tổn thương lỗ lá và lớp thiếu dữ liệu thực địa. Đổi kiến trúc đơn thuần chưa giải quyết được giới hạn dữ liệu. Không tiếp tục chỉnh model dựa vào tập test đã đo.

## Nguồn và cách tạo sơ đồ

Đã tải toàn bộ mã nguồn [Archify](https://github.com/tt-a1i/archify) dạng archive tại revision trong [TOOL_PROVENANCE.json](TOOL_PROVENANCE.json), đồng thời dùng gói skill/CLI chính thức để tạo ba sơ đồ native. Không sửa mã của công cụ. JSON candidate đi kèm cho phép tái tạo sơ đồ.

Các nút có dẫn chứng mã SIC tại commit cố định `6b5857919c47424eae1931bb4def61a34ebb7df1`. Sơ đồ mô tả pipeline đã cài; file kết quả mới được liên kết riêng. Archify đã qua validate, deliver và kiểm tra artifact. Browser-check của Archify lỗi đóng tiến trình trên Windows với Chrome và Edge. Đã kiểm tra độc lập bằng Playwright Chrome ở 1440 và 1920 px, không tràn ngang hoặc lỗi JavaScript; đã xem ảnh chụp cả ba sơ đồ. Không ghi browser-check của Archify là đạt.

Tham khảo kỹ thuật: [MobileNetV3-Large của Torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html), [PlantDoc gốc](https://github.com/pratikkayal/PlantDoc-Dataset), [bài báo PlantDoc](https://arxiv.org/abs/1911.10317).
