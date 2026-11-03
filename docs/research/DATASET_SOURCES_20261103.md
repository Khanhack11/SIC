# Nguồn dữ liệu bổ sung cho SIC, ngày 03/11/2026

Mục tiêu là bổ sung ảnh lá ngô, khoai tây và cà chua có nguồn rõ ràng, nhãn đối chiếu được với 11 lớp của SIC. Độ tin cậy được xem theo nguồn phát hành, cách gán nhãn, mức chi tiết của nhãn và kiểm tra file thực tế. Việc lưu trên Mendeley, Zenodo hoặc GitHub không tự bảo đảm mọi ảnh đúng nhãn.

## Danh mục nguồn mới

| Nguồn chính chủ | Dữ liệu và nhãn được công bố | Mức phù hợp, quyết định |
|---|---|---|
| [CornLeafDiseaseCollection, Murillo Parraga/Silva Villafuerte, Mendeley V1](https://data.mendeley.com/datasets/w56xxnykcc/1) | 1.079 ảnh, 5 lớp; các khu trồng tại Manabí, Ecuador. Cùng smartphone và khoảng cách chụp được kiểm soát. CC BY 4.0. | Đã tải 232 ảnh Healthy, kiểm SHA-256 từng ảnh. Nhãn “common rust (Southern rust)” tự mâu thuẫn nên chưa ánh xạ vào Common Rust của SIC. |
| [Potato Leaf Healthy/Late Blight, Natnael Tilahun, Mendeley V2](https://data.mendeley.com/datasets/v4w72bsts5/2) | Công bố 426 ảnh, hai lớp, chụp ở Holeta, Ethiopia với hỗ trợ của hai chuyên gia bệnh cây. CC BY 4.0. | ZIP/RAR tải thực tế có 430 JPG: 363 Healthy, 67 Late Blight, khác số 63 Late Blight trong mô tả. Đã kiểm SHA-256 RAR; dành cả nguồn làm external_test, không đưa vào train của thí nghiệm này. |
| [Potato Leaf Disease Dataset, Ayesha Banu/Kaushik Deb, BARI, Mendeley V1](https://data.mendeley.com/datasets/d5b3fzpw3g/1) | 804 ảnh gốc chụp tại BARI bằng iPhone 15, sáu lớp Healthy, Bacterial Soft Rot, Fungal Late Blight, Viral Leaf Roll, PVX, PVY. Sau augmentation có 2.400 ảnh. CC BY 4.0. | Có ích cho Healthy/Late Blight và ngoài phạm vi. Cần xác định ảnh gốc và quan hệ biến thể trước khi chia split; 2.400 không phải 2.400 lần chụp độc lập. Chưa đưa vào lượt train này. |
| [Potato Leaf Disease Dataset in Uncontrolled Environment, Shabrina và cộng sự, Mendeley V1](https://data.mendeley.com/datasets/ptz377bwb8/1) | 3.076 ảnh tại Central Java, Indonesia; bảy nhóm Healthy, Bacteria, Fungi, Nematode, Pest, Phytophthora, Virus. CC BY 4.0. | Có nguồn phù hợp về ảnh khỏe và tổn thương ngoài miền. Nhãn nhóm tác nhân rộng, chưa đủ để đồng nhất Fungi với Early Blight hoặc Phytophthora với một bệnh cụ thể nếu chưa đối chiếu bổ sung. Có tập con local nhưng không dùng trong thí nghiệm này. |
| [Maize Leaf Multi-Format, Rabbany và cộng sự, Mendeley V1](https://data.mendeley.com/datasets/sb4wwjs8bz/1) | Ảnh vùng trồng ngô Bangladesh, tháng 6/2026, có phân loại và annotation detection. CC BY 4.0. | Cần đối chiếu thư mục thực tế với danh sách lớp công bố. Trên máy có các bản biểu diễn lặp của cùng ảnh; cần chọn một bản trước khi chia split. Nhãn Corn/Leaf Blight chung không tự đổi thành Northern Leaf Blight. Chưa dùng trong thí nghiệm này. |
| [Tomato Leaf Disease Classification Dataset in Pakistan, Malik và cộng sự, Mendeley V2](https://data.mendeley.com/datasets/3mbnb82mxd/2) | Công bố 7.200 ảnh, sáu lớp Early Blight, Late Blight, Septoria, Leaf Mold, Yellow Leaf Curl Virus, Healthy; gán nhãn theo triệu chứng quan sát. CC BY 4.0. | Cần kiểm tra ảnh nguyên gốc, ảnh augmentation và ảnh trùng PlantDoc trước khi sử dụng. Có dữ liệu local nhưng chưa dùng trong thí nghiệm này. Tên ảnh smartphone chỉ là tín hiệu nguồn, không chứng minh độc lập hay nhãn đúng. |
| [Stage-wise Late Blight Supporting Data, Zenodo 22059910](https://zenodo.org/records/22059910) | Gói ảnh khoai tây và tài liệu hỗ trợ nghiên cứu detection theo giai đoạn. | Bản local có 1.561 ảnh trong Images cùng tài liệu và CSV kết quả. Chưa xác minh được annotation lớp/hộp riêng cho từng ảnh; không tự gán một nhãn chung cho cả thư mục. Chưa dùng trong lượt train này. |
| [Irish Potato Imagery, Laizer và cộng sự, Zenodo 8286529](https://zenodo.org/records/8286529) | 58.709 ảnh, Healthy/Early Blight/Late Blight; thu tại Mbeya, Tanzania bằng Galaxy A03 Core và ODK. Ba gói tải tổng khoảng 7,2 GB. Metadata API xác nhận CC BY 4.0. | Nguồn ưu tiên cho đợt mở rộng khoai tây tiếp theo. Cần kiểm tra nhóm nông trại/phiên chụp và trùng ảnh. Chỉ tải metadata trong lượt này. [Bài dữ liệu của tác giả](https://pmc.ncbi.nlm.nih.gov/articles/PMC12020891/) mô tả quy trình xác nhận nhãn. |
| [TOM2024, Appiah và cộng sự, Mendeley V1](https://data.mendeley.com/datasets/3d4yg89rtr/1) | 25.844 ảnh raw, 12.227 ảnh có nhãn, 30 lớp trên ngô/cà chua/hành. Tác giả mô tả xác định vấn đề và kiểm tra crop bởi chuyên gia. CC BY 4.0. | Có ích cho lá khỏe, bệnh, sâu hại và ngoài phạm vi. Cần giữ crop/species ID, phân biệt ảnh raw với các bản crop/augmentation. Chưa tải; không tự ánh xạ nhãn “healthy leaf” chung vào một loài cây. |
| [Tomato-Village, Gehlot/Saxena/Gandhi, repo tác giả](https://github.com/mamta-joshi-gehlot/Tomato-Village) | Ba variant multiclass, multilabel và detection; có bệnh, sâu hại và thiếu dinh dưỡng ngoài môi trường thực tế. [Bài gốc](https://doi.org/10.1007/s00530-023-01158-y). | Ưu tiên khảo sát thêm để bổ sung cà chua khỏe và nhận biết ngoài phạm vi. Chưa tải/gộp; cần kiểm tra giấy phép dữ liệu và nhóm ảnh trùng giữa ba variant. Không đưa ảnh multilabel vào classifier single-label một cách trực tiếp. |
| [FieldPlant, Moupojou và cộng sự, bài gốc IEEE Access](https://doi.org/10.1109/ACCESS.2023.3263042) | Tác giả báo cáo 5.170 ảnh field, tập trung ngô, cà chua và sắn, có annotation cho detection và phân loại. | Có ích cho bước tìm lá và đánh giá nền thực tế. Chưa xác nhận được phiên bản tải, giấy phép và label schema từ bản phát hành chính chủ trong lượt này, nên giữ ở danh sách khảo sát, chưa train. |
| [IP102, nhóm tác giả CVPR 2019](https://github.com/xpwu95/IP102) | Hơn 75.000 ảnh, 102 lớp côn trùng; 19.000 ảnh có hộp. README cho phép dùng học thuật, mục đích khác liên hệ tác giả. | Là bộ nhận dạng côn trùng, không phải nhãn nguyên nhân của lỗ trên lá. Hữu ích cho module sâu hại riêng hoặc thiết kế nhóm ngoài miền sau này. Chưa tải/ghép vào 11 lớp bệnh lá. |

## Phân biệt nhãn và định dạng

Ảnh phân loại thường dùng nhãn từ tên thư mục hoặc CSV. Cần ghi riêng tên nguồn, nhãn gốc và nhãn SIC. Ví dụ `Healthy` trong nguồn chỉ gồm ngô mới có thể đề xuất `Corn_healthy`; chữ Healthy trong bộ nhiều loài chưa đủ thông tin.

Hộp detection có thể là `xmin,ymin,xmax,ymax`, COCO `x,y,width,height`, hoặc YOLO tọa độ tâm và kích thước chuẩn hóa. Phải kiểm kích thước ảnh và hướng EXIF trước khi crop. Hộp quanh tổn thương không mặc nhiên bao toàn bộ lá.

Mask segmentation là annotation theo pixel, cần bảng mã nền/lớp. PlantSeg hiện có trên máy dùng nền 0 và mã vùng bệnh bằng Index+1; các JSON COCO của bản tải thiếu danh sách categories. Mask này mô tả tổn thương, không phải đường viền toàn bộ lá.

Một ảnh nhiều bệnh hoặc nhiều đối tượng không phù hợp với nhãn single-label nếu chưa xác định rõ vùng cần phân loại. Nhãn tác nhân rộng, tên bệnh mơ hồ hoặc nhiều loài phải giữ ở nhóm cần xem lại. Augmentation được thực hiện sau chia split; các bản cùng ảnh gốc luôn cùng nhóm.

## Dữ liệu thật sự dùng trong thí nghiệm expanded_20261103

Đã xem bố cục 871 ảnh trên 14 contact sheet: 639 ứng viên PlantWild_v2 và 232 ảnh ngô Healthy Ecuador. Tách 123 ảnh sơ đồ, ghép nhiều khung, chữ chú thích lớn, chủ thể quả/thân/củ hoặc không đủ rõ để duyệt ở kích thước contact sheet. Đây là AI layout triage, không phải chuyên gia xác nhận bệnh.

Sau lọc trùng với bộ cũ và giữa các ảnh mới, manifest thêm 516 ảnh PlantWild_v2 (413 train, 103 validation) và 224 ảnh ngô Healthy Ecuador (train-only). Nhãn vẫn là nhãn nguồn, có ghi `label_review` trong từng record. Nguồn Ecuador không có ID lá/phiên chụp để tạo validation độc lập đáng tin cậy, vì vậy không báo accuracy field trên cách chia ngẫu nhiên nguồn này.

Holeta có 430 JPG thực tế; 12 ảnh được loại do trùng hoặc gần giống theo bộ lọc, giữ 418 ảnh làm `external_test`. Toàn bộ nguồn này không được nạp vào train/validation của thí nghiệm. Không giả định số ảnh công bố bằng số file tải thực tế.

Phạm vi độc lập của Holeta chỉ áp dụng cho checkpoint `mobilenetv3_expanded_20261103` và parent `mobilenetv3_field_continued_20261103`. Manifest của job khác, `field_mobile_v2`, có ảnh Holeta trong train/validation. Không dùng kết quả Holeta của job đó để tuyên bố kiểm thử trên nguồn chưa từng học.

Manifest tổng: 10.338 train, 2.248 validation, 1.937 test lịch sử và 418 external_test. Validation/test cũ được giữ nguyên record và hash. Lượt warm-start dùng parent V3 đã train trên dữ liệu cũ; những benchmark cũ không trở thành dữ liệu thực địa mới độc lập chỉ vì được chạy lại.

Nguồn raw và publisher metadata nằm tại `data/raw/dataset_hub_20261103/`. Manifest thí nghiệm nằm tại `data/processed/field_expanded_v2_20261103/`. Không đánh đồng các bộ dữ liệu đang có trên máy với tập ảnh mà job train này thực sự dùng.

Lượt train bốn epoch đã hoàn tất trên RTX 2050. Checkpoint được chọn theo validation loss; [hồ sơ thí nghiệm](../final_release/data_expansion_20261103/README.md) lưu cấu hình, manifest bổ sung và kết quả đánh giá.

## Quy trình gán nhãn đề xuất cho ảnh nhóm tự thu

1. Lưu bản gốc cùng source ID, crop species, ngày/phiên chụp và mã cây/lá nếu có; chưa gán bệnh chỉ từ model dự đoán.
2. Người có chuyên môn kiểm tra nhãn bệnh hoặc healthy; có lựa chọn uncertain, ngoài phạm vi và nhiều bệnh. Người duyệt ghi tên, ngày, nguồn căn cứ và mức chắc chắn.
3. Nếu cần tự crop, gán hộp bao cả lá cần phân tích. Nếu cần hiển thị tổn thương, gán mask riêng. Duyệt chéo các trường hợp khó và các nhãn xung đột.
4. Gom nhóm theo cây/lá/phiên chụp và ảnh biến đổi, loại trùng, rồi khóa train/validation/test. Giữ một nguồn độc lập cho đánh giá sau khi cấu hình và checkpoint đã cố định.
5. Chỉ dùng train để cập nhật trọng số; dùng validation để chọn checkpoint và calibration. Khi test đã được xem để quyết định cải tiến, phải ghi nhận việc đó và thu thêm holdout mới cho tuyên bố độc lập tiếp theo.

Manifest cần giữ ít nhất `path`, `sha256`, `label`, `original_label`, `source_name`, `domain`, `audit_group`, `split`, `label_review`. Trạng thái nguồn và mức duyệt nhãn phải đi cùng kết quả, kể cả khi điểm validation tăng.
