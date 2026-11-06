"""Enrich advisory_knowledge_base.json with comprehensive agronomy, soil, environment, pest vectors, and dosage data."""
import json
from pathlib import Path

KB_PATH = Path('data/advisory_knowledge_base.json')

def enrich():
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Update general metadata
    data['version'] = '2.0.0-agronomy-extended'
    data['kaggle_reference_dataset'] = {
        'title': 'New Plant Diseases Dataset (87K Images, 38 Classes)',
        'kaggle_url': 'https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset',
        'creator': 'Smit Patel / Vipooool',
        'classes_count': 38,
        'images_count': 87000,
        'nature': 'Laboratory & controlled background augmented dataset (augmented PlantVillage)',
        'field_adaptation_challenge': 'High laboratory accuracy (~99%) but susceptible to background noise in real farm environments.',
        'our_solution': 'Multi-source hybrid training (field_mobile_v2 with 20,902 records) combining PlantDoc single-leaf bounding-box crops and genuine field captures from 10 international datasets.'
    }

    # 2. Add comprehensive agronomy details for all 11 classes
    agronomy_data = {
        "Tomato_Early_blight": {
            "soil_requirements": {
                "preferred_soil": "Đất thịt nhẹ, đất cát pha giàu mùn hữu cơ, tơi xốp, tầng canh tác dày > 25cm.",
                "optimal_ph": "6.0 - 6.8",
                "soil_moisture_pct": "60% - 70% (đất ẩm đều, tuyệt đối tránh ngập úng bí khí rễ)",
                "soil_drainage": "Thoát nước nhanh, lên luống cao 25 - 30cm, phủ bạt nilon nông nghiệp để ngăn bào tử nấm từ đất bắn lên lá khi mưa."
            },
            "environmental_conditions": {
                "temperature_range_c": "Nhiệt độ thích hợp cho cây: 22 - 28°C; Nhiệt độ tối thích bùng phát nấm Alternaria: 24 - 30°C.",
                "humidity_pct": "Độ ẩm không khí cao > 85%, đặc biệt khi mưa dầm xen kẽ nắng ráo hoặc lá bị ướt sương đêm trên 8 giờ.",
                "seasonality": "Vụ Đông Xuân và Xuân Hè tại các vùng trồng trọng điểm (Đồng bằng sông Hồng, Lâm Đồng - Đà Lạt, Tây Nguyên)."
            },
            "pest_vectors_and_co_infections": [
                "Bọ trĩ (Thrips palmi): chích hút đọt non tạo vết thương hở cho bào tử nấm xâm nhiễm.",
                "Bọ phấn trắng (Bemisia tabaci): môi giới truyền virus kết hợp làm cây suy yếu nhanh chóng.",
                "Sâu đục quả cà chua (Helicoverpa armigera): tạo lỗ thủng khiến nấm xâm nhập sang quả."
            ],
            "disease_stages": {
                "initial": "Vết đốm nhỏ màu nâu đậm hoặc đen xuất hiện đầu tiên trên các lá già sát mặt đất.",
                "progressive": "Vết đốm lan rộng, hình thành các vòng tròn đồng tâm đặc trưng giống hình bia bắn (target spots), viền lá chuyển vàng úa.",
                "severe": "Các đốm liên kết thành mảng cháy lớn, lá vàng rụng hàng loạt từ dưới lên trên, cây xơ xác, cuống quả và quả có vết lõm đen."
            },
            "comprehensive_curative_solutions": {
                "cultural_practices": [
                    "Luân canh triệt để: Không trồng cà chua trên đất vụ trước đã trồng cây họ Cà (khoai tây, cà tím, ớt, thuốc lá) tối thiểu 2 - 3 năm; luân canh với lúa nước hoặc bắp.",
                    "Làm đất & xử lý đất: Bón vôi bột khử chua và diệt khuẩn (25 - 30 kg/sào Bắc Bộ hoặc 500 - 800 kg/ha) trước khi trồng 15 ngày.",
                    "Mật độ & tỉa cành: Trồng hàng cách hàng 70cm, cây cách cây 40 - 50cm; tỉa toàn bộ lá già, lá chân sát gốc cách mặt đất 20 - 25cm tạo độ thông thoáng tuyệt đối.",
                    "Chế độ tưới tiêu: Tưới rãnh thấm hoặc tưới nhỏ giọt sát gốc, tuyệt đối không tưới phun mưa lúc trời chiều tối làm ướt tán lá qua đêm."
                ],
                "biological_organic": [
                    "Nấm đối kháng Trichoderma harzianum: Trộn 2 - 3 kg/tấn phân chuồng hoai mục bón lót hoặc pha 40g/bình 16L tưới rễ định kỳ 10 - 15 ngày/lần.",
                    "Vi khuẩn đối kháng Bacillus subtilis: Phun ướt đều tán lá (30g/bình 16L) để cạnh tranh sinh học và ức chế nấm phát triển.",
                    "Dầu Neem hữu cơ (Azadirachtin 0.3%): Pha 15 - 20ml/bình 16L phun xua đuổi bọ trĩ, bọ phấn và tạo màng bảo vệ biểu bì lá."
                ],
                "chemical_prescriptions": [
                    {
                        "active_ingredient": "Azoxystrobin 200g/l + Difenoconazole 125g/l",
                        "trade_names": ["Amistar Top 325SC"],
                        "target_action": "Thuốc trừ bệnh nội hấp lưu dẫn cực mạnh, ức chế nấm nảy mầm và diệt sợi nấm bên trong mô lá.",
                        "exact_dosage": "15 - 20 ml / bình 16 lít nước (hoặc 25 - 30 ml / bình 25 lít). Lượng nước phun 400 - 500 lít/ha.",
                        "application_timing": "Phun ngay khi tỷ lệ lá chớm bệnh đạt 3 - 5%; phun nhắc lại lần 2 sau 7 - 10 ngày nếu thời tiết tiếp tục mưa ẩm.",
                        "pre_harvest_interval_days": 7
                    },
                    {
                        "active_ingredient": "Difenoconazole 150g/l + Propiconazole 150g/l",
                        "trade_names": ["Tilt Super 300EC", "Score 250EC"],
                        "target_action": "Đặc trị nấm Ascomycetes, làm khô vết bệnh nhanh chóng trong vòng 24 - 48 giờ.",
                        "exact_dosage": "10 - 12 ml / bình 16 lít nước.",
                        "application_timing": "Phun luân phiên với nhóm Strobilurin để chống hiện tượng kháng thuốc.",
                        "pre_harvest_interval_days": 7
                    },
                    {
                        "active_ingredient": "Mancozeb 800g/kg",
                        "trade_names": ["Dithane M-45 80WP", "Manzate 200"],
                        "target_action": "Thuốc tiếp xúc phổ rộng, tạo lớp màng bảo vệ mặt lá ngăn chặn bào tử nảy mầm.",
                        "exact_dosage": "50 - 60 g / bình 16 lít nước (100g / bình 25L).",
                        "application_timing": "Phun phòng ngừa định kỳ trước các đợt mưa hoặc khi độ ẩm không khí tăng cao.",
                        "pre_harvest_interval_days": 10
                    }
                ]
            },
            "weather_risk_triggers": {
                "high_risk_temp_min": 24.0,
                "high_risk_temp_max": 30.0,
                "high_risk_humidity_min": 85.0,
                "risk_warning": "CẢNH BÁO NGUY CƠ CAO BỆNH ĐỐM VÒNG CÀ CHUA: Nhiệt độ 24-30°C kết hợp độ ẩm > 85% và mưa ẩm kéo dài. Cần tỉa lá già chân gốc và phun phòng ngay."
            }
        },
        "Tomato_Late_blight": {
            "soil_requirements": {
                "preferred_soil": "Đất phù sa bãi bồi, đất thịt pha cát thoát nước cực tốt, giàu chất hữu cơ, pH 5.8 - 6.8.",
                "optimal_ph": "5.8 - 6.8",
                "soil_moisture_pct": "65% - 75% (đất ẩm nhưng rãnh luống phải khô ráo)",
                "soil_drainage": "Rãnh luống sâu 30 - 35cm, lên luống cao, không để nước đọng rãnh quá 2 giờ sau mưa."
            },
            "environmental_conditions": {
                "temperature_range_c": "Nhiệt độ tối thích bùng phát dịch: 16 - 22°C (thời tiết lạnh ẩm). Nhiệt độ > 30°C nấm ngưng phát triển.",
                "humidity_pct": "Độ ẩm không khí rất cao > 90%, trời âm u, sương mù dày đặc hoặc mưa phùn kéo dài liên tục 2 - 3 ngày.",
                "seasonality": "Vụ Đông tại miền Bắc (tháng 11 đến tháng 2 năm sau) và quanh năm tại cao nguyên Lâm Đồng."
            },
            "pest_vectors_and_co_infections": [
                "Bọ xít muỗi (Helopeltis spp.): chích hút tạo vết châm thâm đen trên cuống.",
                "Rầy mềm (Aphis gossypii): tiết dịch ngọt tạo môi trường nấm bồ hóng phát triển."
            ],
            "disease_stages": {
                "initial": "Vết úng nước hình dạng bất định màu xanh tái hoặc xám xuất hiện ở chóp và mép lá.",
                "progressive": "Vết bệnh chuyển màu nâu đen, lan nhanh vào gân lá và cuống lá; mặt dưới lá phủ lớp mốc tơ trắng mịn (bào tử noãn khuẩn).",
                "severe": "Thân cây bị thối ủng màu đen, gãy gục; toàn bộ ruộng cháy lụi như bị luộc nước sôi chỉ trong 3 - 5 ngày."
            },
            "comprehensive_curative_solutions": {
                "cultural_practices": [
                    "Lên luống cao 30 - 40cm, tạo rãnh thoát nước dốc để nước thoát ngay sau mưa.",
                    "Không trồng với mật độ quá dày; cắm cọc làm giàn chữ A kiên cố, thông thoáng gió.",
                    "Theo dõi dự báo thời tiết: Khi có đợt gió mùa Đông Bắc kèm mưa phùn hoặc sương mù dày, phải ngừng bón phân đạm (N), tăng cường Kali (K).",
                    "Thu gom toàn bộ cây và quả bị bệnh vào bao nilon kín đem ra khỏi ruộng tiêu hủy hoặc chôn rắc vôi bột."
                ],
                "biological_organic": [
                    "Chế phẩm Nano Đồng (Nano Copper 2000ppm): Pha 30 - 40ml/bình 16L phun định kỳ, ức chế màng tế bào noãn khuẩn Oomycete.",
                    "Chế phẩm vi sinh Bacillus amyloliquefaciens: Pha 30g/bình 16L phun phủ bề mặt lá để cạnh tranh dinh dưỡng."
                ],
                "chemical_prescriptions": [
                    {
                        "active_ingredient": "Metalaxyl-M 40g/kg + Mancozeb 640g/kg",
                        "trade_names": ["Ridomil Gold 68WG"],
                        "target_action": "Thuốc đặc trị sương mai số 1, lưu dẫn 2 chiều cực mạnh từ lá xuống rễ và ngược lại.",
                        "exact_dosage": "50 g / bình 16 lít nước (hoặc 80 - 100g / bình 25L). Lượng nước phun 500 - 600 lít/ha.",
                        "application_timing": "Phun ngay khi chớm có sương mù hoặc phát hiện chấm bệnh úng nước đầu tiên.",
                        "pre_harvest_interval_days": 7
                    },
                    {
                        "active_ingredient": "Dimethomorph 9% + Mancozeb 60%",
                        "trade_names": ["Acrobat MZ 90/600WP", "Insignia"],
                        "target_action": "Phá hủy vách tế bào nấm noãn khuẩn ở mọi giai đoạn sinh trưởng.",
                        "exact_dosage": "30 - 40 g / bình 16 lít nước.",
                        "application_timing": "Phun luân phiên với Ridomil Gold khi bệnh có nguy cơ bùng phát diện rộng.",
                        "pre_harvest_interval_days": 7
                    },
                    {
                        "active_ingredient": "Cymoxanil 8% + Mancozeb 64%",
                        "trade_names": ["Curzate M8 72WP"],
                        "target_action": "Thấm sâu nhanh, diệt mầm bệnh đã xâm nhập vào biểu bì lá trong vòng 24 - 48 giờ.",
                        "exact_dosage": "40 - 50 g / bình 16 lít nước.",
                        "application_timing": "Phun chặn dịch khi vết bệnh bắt đầu lan sang cuống và thân.",
                        "pre_harvest_interval_days": 7
                    }
                ]
            },
            "weather_risk_triggers": {
                "high_risk_temp_min": 15.0,
                "high_risk_temp_max": 22.0,
                "high_risk_humidity_min": 90.0,
                "risk_warning": "BÁO ĐỘNG ĐỎ DỊCH SƯƠNG MAI CÀ CHUA: Nhiệt độ lạnh 15-22°C và sương mù/độ ẩm > 90%. Nấm Phytophthora có thể tàn phá cả ruộng trong 72 giờ! Cần phun chặn ngay bằng Ridomil Gold hoặc Acrobat MZ."
            }
        },
        "Tomato_Septoria_leaf_spot": {
            "soil_requirements": {
                "preferred_soil": "Đất thịt nhẹ, đất phù sa bãi bồi, pH 6.0 - 6.8.",
                "optimal_ph": "6.0 - 6.8",
                "soil_moisture_pct": "60% - 70%",
                "soil_drainage": "Thoát nước tốt, tránh để đọng vũng nước dưới gốc cây."
            },
            "environmental_conditions": {
                "temperature_range_c": "Nhiệt độ tối thích: 20 - 26°C.",
                "humidity_pct": "Độ ẩm không khí 80% - 90%, thời tiết ẩm ướt sau các đợt mưa dông rải rác.",
                "seasonality": "Giai đoạn cây bắt đầu nuôi quả non đến thu hoạch."
            },
            "pest_vectors_and_co_infections": [
                "Sâu vẽ bùa (Liriomyza sativae): đục ngoằn ngoèo trên lá làm biểu bì suy yếu.",
                "Nhện đỏ hai chấm (Tetranychus urticae): làm lá bị vàng bạc và dễ nhiễm nấm."
            ],
            "disease_stages": {
                "initial": "Nhiều đốm nhỏ li ti hình tròn đường kính 1 - 3mm, mép nâu sẫm, tâm đốm màu xám trắng.",
                "progressive": "Ở giữa tâm đốm xuất hiện các chấm đen nhỏ li ti như hạt tiêu (thể quả pycnidia của nấm).",
                "severe": "Lá chuyển vàng đồng loạt, khô héo và rụng từ gốc lên ngọn khiến quả bị rám nắng."
            },
            "comprehensive_curative_solutions": {
                "cultural_practices": [
                    "Thu dọn và đốt sạch tàn dư lá bệnh rụng dưới gốc sau mỗi đợt tỉa cành.",
                    "Tránh làm việc hoặc lội qua vườn cà chua khi lá cây đang ướt sương để không phát tán bào tử.",
                    "Bón bổ sung phân bón lá chứa Canxi, Bo và Silic giúp tăng độ dày vách tế bào."
                ],
                "biological_organic": [
                    "Dung dịch Bordeaux (Boóc-đô 1%): Pha 100g phèn xanh (CuSO4) + 100g vôi tôi trong 10 lít nước phun phòng ngừa định kỳ.",
                    "Dịch chiết tỏi - ớt - gừng lên men hữu cơ: Pha 50ml/bình 16L phun xua đuổi côn trùng và sát khuẩn bề mặt."
                ],
                "chemical_prescriptions": [
                    {
                        "active_ingredient": "Chlorothalonil 750g/kg",
                        "trade_names": ["Daconil 75WP", "Bravo 720SC"],
                        "target_action": "Thuốc tiếp xúc bám dính cực tốt, chống rửa trôi khi trời mưa.",
                        "exact_dosage": "30 - 35 g / bình 16 lít nước.",
                        "application_timing": "Phun khi phát hiện lấm tấm đốm tròn trên lá già.",
                        "pre_harvest_interval_days": 7
                    },
                    {
                        "active_ingredient": "Copper Oxychloride 85%",
                        "trade_names": ["Coc 85WP", "Champion 77WP"],
                        "target_action": "Gốc đồng sát khuẩn phổ rộng và phòng ngừa nấm Septoria hữu hiệu.",
                        "exact_dosage": "25 - 30 g / bình 16 lít nước.",
                        "application_timing": "Phun định kỳ 10 - 14 ngày/lần.",
                        "pre_harvest_interval_days": 5
                    }
                ]
            },
            "weather_risk_triggers": {
                "high_risk_temp_min": 20.0,
                "high_risk_temp_max": 26.0,
                "high_risk_humidity_min": 80.0,
                "risk_warning": "CẢNH BÁO BỆNH ĐỐM LÁ SEPTORIA CÀ CHUA: Mưa ẩm ở nhiệt độ 20-26°C. Cần phun bảo vệ lá tầng dưới và dọn sạch lá gốc rụng."
            }
        },
        "Tomato_healthy": {
            "soil_requirements": {
                "preferred_soil": "Đất thịt nhẹ, đất cát pha, đất phù sa tơi xốp, hàm lượng chất hữu cơ > 2.5%, tầng canh tác sâu > 30cm.",
                "optimal_ph": "6.2 - 6.8",
                "soil_moisture_pct": "60% - 70%",
                "soil_drainage": "Hệ thống mương rãnh thoát nước hoàn chỉnh, mặt luống khô ráo."
            },
            "environmental_conditions": {
                "temperature_range_c": "Nhiệt độ ngày: 22 - 28°C; Nhiệt độ đêm: 15 - 18°C. Chênh lệch nhiệt độ ngày đêm 7 - 10°C giúp quả ngọt và chắc.",
                "humidity_pct": "60% - 70% (thông thoáng gió).",
                "seasonality": "Phù hợp mọi vụ nếu quản lý tốt ánh sáng và nước tưới."
            },
            "pest_vectors_and_co_infections": [],
            "disease_stages": {
                "status": "Cây sinh trưởng khỏe mạnh, tán lá xanh bóng mỡ màng, không có vết bệnh, hoa nở rộ và đậu quả đều."
            },
            "comprehensive_curative_solutions": {
                "cultural_practices": [
                    "Bón phân cân đối N-P-K theo công thức 15-15-15 hoặc 20-20-15 kết hợp phân hữu cơ hoai mục 15 - 20 tấn/ha.",
                    "Tưới nước đều đặn vào buổi sáng, duy trì độ ẩm ổn định tránh hiện tượng nứt quả do sốc nước.",
                    "Lắp bẫy dính màu vàng và xanh trong ruộng để theo dõi và bắt bọ phấn, bọ trĩ từ sớm."
                ],
                "biological_organic": [
                    "Phun bổ sung phân bón lá rong biển và axit amin hữu cơ định kỳ 10 ngày/lần giúp tăng đề kháng tự nhiên."
                ],
                "chemical_prescriptions": []
            },
            "weather_risk_triggers": {
                "high_risk_temp_min": 35.0,
                "high_risk_temp_max": 45.0,
                "high_risk_humidity_min": 95.0,
                "risk_warning": "Điều kiện hiện tại thuận lợi cho cây cà chua phát triển khỏe mạnh. Tiếp tục duy trì chế độ chăm sóc và tưới tiêu hợp lý."
            }
        },
        "Potato_Early_blight": {
            "soil_requirements": {
                "preferred_soil": "Đất cát pha, đất thịt nhẹ tơi xốp, nhiều mùn, pH 5.2 - 6.2 giúp củ lớn nhanh và nhẵn vỏ.",
                "optimal_ph": "5.2 - 6.2",
                "soil_moisture_pct": "65% - 75%",
                "soil_drainage": "Tuyệt đối không để đọng nước gây thối củ và tạo điều kiện nấm tấn công."
            },
            "environmental_conditions": {
                "temperature_range_c": "22 - 28°C; thời tiết luân phiên giữa khô hạn và mưa rào ẩm ướt.",
                "humidity_pct": "75% - 85%.",
                "seasonality": "Vụ Đông tại các tỉnh Bắc Bộ và vụ Thu Đông tại Tây Nguyên."
            },
            "pest_vectors_and_co_infections": [
                "Bọ rùa 28 chấm hại khoai tây (Henosepilachna vigintioctopunctata): ăn biểu bì lá tạo sọc thủng.",
                "Rệp đào (Myzus persicae): truyền bệnh thoái hóa củ."
            ],
            "disease_stages": {
                "initial": "Đốm nâu nhỏ góc cạnh trên lá dưới gốc.",
                "progressive": "Đốm lan thành hình tròn đồng tâm, mô lá xung quanh bị úa vàng, lá cong queo.",
                "severe": "Toàn bộ tán lá tàn lụi sớm trước khi củ kịp tích lũy tinh bột, năng suất giảm 30 - 50%."
            },
            "comprehensive_curative_solutions": {
                "cultural_practices": [
                    "Chọn củ giống sạch bệnh từ cơ sở uy tín, xử lý củ trước khi trồng bằng nước ấm hoặc chế phẩm nấm đối kháng.",
                    "Vun luống cao 2 đợt: đợt 1 khi cây cao 15-20cm, đợt 2 khi cây bắt đầu ra hoa để che kín củ.",
                    "Luân canh với lúa nước, ngô hoặc cây họ Đậu; tránh đất đã trồng cà chua hoặc ớt."
                ],
                "biological_organic": [
                    "Tưới gốc Trichoderma viride khi đặt củ giống (50g/bình 16L).",
                    "Phun Nano Bạc - Đồng nồng độ 20ppm kích kháng cây trồng."
                ],
                "chemical_prescriptions": [
                    {
                        "active_ingredient": "Difenoconazole 250g/l",
                        "trade_names": ["Score 250EC"],
                        "target_action": "Lưu dẫn nội hấp làm khô nhanh vết đốm vòng trên lá khoai tây.",
                        "exact_dosage": "10 - 15 ml / bình 16 lít nước.",
                        "application_timing": "Phun khi phát hiện vết bệnh đầu tiên thời kỳ cây chuẩn bị phình củ.",
                        "pre_harvest_interval_days": 7
                    },
                    {
                        "active_ingredient": "Propineb 700g/kg",
                        "trade_names": ["Antracol 70WP"],
                        "target_action": "Tiếp xúc bảo vệ phổ rộng, cung cấp vi lượng Kẽm (Zn) giúp bộ lá khoai tây xanh dày dẻo dai.",
                        "exact_dosage": "40 - 50 g / bình 16 lít nước.",
                        "application_timing": "Phun định kỳ 7 - 10 ngày/lần trong mùa mưa.",
                        "pre_harvest_interval_days": 7
                    }
                ]
            },
            "weather_risk_triggers": {
                "high_risk_temp_min": 22.0,
                "high_risk_temp_max": 28.0,
                "high_risk_humidity_min": 80.0,
                "risk_warning": "CẢNH BÁO ĐỐM VÒNG KHOAI TÂY: Thời tiết 22-28°C độ ẩm cao. Cần kiểm tra lá gốc và vun luống che củ."
            }
        },
        "Potato_Late_blight": {
            "soil_requirements": {
                "preferred_soil": "Đất thịt nhẹ, đất cát pha, pH 5.0 - 6.0.",
                "optimal_ph": "5.0 - 6.0",
                "soil_moisture_pct": "70% - 80%",
                "soil_drainage": "Rãnh luống sâu > 30cm, đất tơi xốp không đóng váng bề mặt."
            },
            "environmental_conditions": {
                "temperature_range_c": "15 - 20°C (nhiệt độ lạnh ẩm tối thích cho noãn khuẩn Phytophthora infestans).",
                "humidity_pct": "> 90%, sương mù dầy đặc, mưa phùn kéo dài.",
                "seasonality": "Vụ Đông chính vụ tháng 12 đến tháng 1 tại miền Bắc."
            },
            "pest_vectors_and_co_infections": [
                "Bọ nhảy sọc cong hại củ non.",
                "Sâu xám cắn đứt gốc thân non."
            ],
            "disease_stages": {
                "initial": "Vết úng nước màu xanh tối ở mép hoặc chóp lá.",
                "progressive": "Vết bệnh thâm đen nhanh chóng, mặt dưới lá có lớp phấn trắng xốp như bông khi trời ẩm.",
                "severe": "Cành lá thối nhũn bốc mùi tanh hôi đặc trưng, nấm xâm nhập vào củ qua đất làm củ thối nâu khô xơ."
            },
            "comprehensive_curative_solutions": {
                "cultural_practices": [
                    "Vun gốc cao và dày để bào tử nấm từ lá không bị nước mưa rửa trôi thấm xuống củ bên dưới.",
                    "Cắt toàn bộ thân lá trước thu hoạch 7 - 10 ngày, phơi đất ráo mặt trước khi đào củ để vỏ củ dày chắc.",
                    "Tuyệt đối không đào củ vào ngày mưa ẩm ướt."
                ],
                "biological_organic": [
                    "Bacillus subtilis 1x10^9 CFU/g: Pha 40g/bình 16L phun phòng định kỳ.",
                    "Chế phẩm Nano Đồng hữu cơ 35ml/bình 16L."
                ],
                "chemical_prescriptions": [
                    {
                        "active_ingredient": "Mandipropamid 250g/l + Difenoconazole 250g/l",
                        "trade_names": ["Revus Opti 440SC"],
                        "target_action": "Đặc trị sương mai noãn khuẩn, bám dính lớp sáp cutin lá chỉ sau 1 giờ phun.",
                        "exact_dosage": "20 - 25 ml / bình 16 lít nước.",
                        "application_timing": "Phun chặn dịch ngay khi có đợt sương muối hoặc mưa phùn lạnh.",
                        "pre_harvest_interval_days": 7
                    },
                    {
                        "active_ingredient": "Metalaxyl-M 4% + Mancozeb 64%",
                        "trade_names": ["Ridomil Gold 68WG"],
                        "target_action": "Thuốc tiêu chuẩn quốc tế lưu dẫn nội hấp mạnh trị sương mai khoai tây.",
                        "exact_dosage": "50 g / bình 16 lít nước (100g/bình 25L).",
                        "application_timing": "Phun ướt đều 2 mặt lá và thân khi chớm xuất hiện vết bệnh.",
                        "pre_harvest_interval_days": 7
                    },
                    {
                        "active_ingredient": "Fosetyl Aluminium 80%",
                        "trade_names": ["Aliette 80WP"],
                        "target_action": "Lưu dẫn hai chiều kích hoạt hệ thống miễn dịch tự nhiên của cây.",
                        "exact_dosage": "40 g / bình 16 lít nước.",
                        "application_timing": "Phun luân phiên để tránh nấm kháng thuốc Metalaxyl.",
                        "pre_harvest_interval_days": 14
                    }
                ]
            },
            "weather_risk_triggers": {
                "high_risk_temp_min": 14.0,
                "high_risk_temp_max": 20.0,
                "high_risk_humidity_min": 90.0,
                "risk_warning": "BÁO ĐỘNG ĐỎ SƯƠNG MAI KHOAI TÂY: Thời tiết rét ẩm 14-20°C sương mù dày đặc! Nguy cơ thối rụi toàn ruộng trong 48-72h. Phun khẩn cấp Ridomil Gold hoặc Revus Opti."
            }
        },
        "Potato_healthy": {
            "soil_requirements": {
                "preferred_soil": "Đất cát pha, thịt nhẹ, tơi xốp nhiều mùn, pH 5.5 - 6.5, tơi xốp sâu 30cm.",
                "optimal_ph": "5.5 - 6.5",
                "soil_moisture_pct": "65% - 75%",
                "soil_drainage": "Thoát nước cực nhanh, luống cao 30cm."
            },
            "environmental_conditions": {
                "temperature_range_c": "Nhiệt độ tối ưu: 16 - 22°C.",
                "humidity_pct": "70% - 80%.",
                "seasonality": "Vụ Đông miền Bắc và vụ quanh năm tại cao nguyên."
            },
            "pest_vectors_and_co_infections": [],
            "disease_stages": {
                "status": "Cây khoai tây phát triển xanh tốt, lá dày màu xanh đậm, thân mập mạp, củ lớn đồng đều."
            },
            "comprehensive_curative_solutions": {
                "cultural_practices": [
                    "Bón phân cân đối N-P-K theo tỷ lệ 2:1:3 (chú trọng Kali để củ to và tích lũy tinh bột cao).",
                    "Giữ ẩm đất ổn định, tưới rãnh định kỳ 7 - 10 ngày/lần đến khi cây bắt đầu xuống củ."
                ],
                "biological_organic": [
                    "Bón lót phân chuồng ủ nấm Trichoderma 10 - 15 tấn/ha."
                ],
                "chemical_prescriptions": []
            },
            "weather_risk_triggers": {
                "high_risk_temp_min": 32.0,
                "high_risk_temp_max": 42.0,
                "high_risk_humidity_min": 95.0,
                "risk_warning": "Thời tiết lý tưởng cho khoai tây sinh trưởng. Tiếp tục duy trì chế độ vun luống và bón phân nuôi củ."
            }
        },
        "Corn_Common_rust": {
            "soil_requirements": {
                "preferred_soil": "Đất phù sa, đất thịt xốp, đất đỏ bazan, pH 5.8 - 7.0.",
                "optimal_ph": "5.8 - 7.0",
                "soil_moisture_pct": "60% - 70%",
                "soil_drainage": "Thoát nước tốt, tránh để ngập úng chân ngô quá 12 giờ."
            },
            "environmental_conditions": {
                "temperature_range_c": "16 - 24°C; thời tiết mát mẻ kèm sương đọng trên lá trên 6 giờ.",
                "humidity_pct": "> 85%.",
                "seasonality": "Vụ Đông và vụ Xuân tại miền Bắc và các vùng trung du miền núi."
            },
            "pest_vectors_and_co_infections": [
                "Sâu keo mùa thu (Spodoptera frugiperda): cắn phá nõn ngô tạo điều kiện mầm bệnh xâm nhập.",
                "Sâu đục thân ngô (Ostrinia furnacalis)."
            ],
            "disease_stages": {
                "initial": "Xuất hiện các đốm nhỏ màu vàng nhạt trên cả hai mặt lá ngô.",
                "progressive": "Đốm bệnh phát triển thành các ổ mụn mủ (pustules) hình bầu dục nổi gồ lên bề mặt, chứa đầy bột màu nâu đỏ gỉ sắt.",
                "severe": "Các ổ mụn mủ vỡ tung giải phóng hàng triệu bào tử bột gỉ sắt theo gió; lá bị cháy khô từ ngọn xuống gốc, bắp ngô nhỏ hạt lép."
            },
            "comprehensive_curative_solutions": {
                "cultural_practices": [
                    "Sử dụng giống ngô lai F1 kháng bệnh gỉ sắt (các giống LVN, CP, Syngenta đạt chuẩn).",
                    "Mật độ gieo trồng hợp lý: Cây cách cây 25 - 30cm, hàng cách hàng 70cm (khoảng 55.000 - 60.000 cây/ha) tạo độ thông thoáng gió.",
                    "Bón phân cân đối N-P-K, bón lót phân lân đầy đủ, bón thúc đợt 1 (khi ngô 3-4 lá) và đợt 2 (khi ngô 7-9 lá xoáy nõn) tăng cường Kali giúp cứng cây dày lá.",
                    "Sau thu hoạch, cày vùi hoặc đốt tiêu hủy toàn bộ tàn dư thân lá ngô."
                ],
                "biological_organic": [
                    "Chế phẩm sinh học kháng nấm vi sinh gốc nấm men hoặc Bacillus subtilis (30g/bình 16L).",
                    "Nano Chitosan hữu cơ kích thích miễn dịch tự nhiên của mô lá ngô."
                ],
                "chemical_prescriptions": [
                    {
                        "active_ingredient": "Azoxystrobin 200g/l + Difenoconazole 125g/l",
                        "trade_names": ["Amistar Top 325SC"],
                        "target_action": "Thuốc phổ rộng trừ bệnh gỉ sắt ngô, thấm sâu nhanh và hiệu lực kéo dài 15 - 20 ngày.",
                        "exact_dosage": "15 ml / bình 16 lít nước (hoặc 25ml / bình 25L). Lượng nước phun 400 - 500 lít/ha.",
                        "application_timing": "Phun khi phát hiện 5% số lá chớm có mụn gỉ sắt thời kỳ ngô 7-9 lá đến trỗ cờ.",
                        "pre_harvest_interval_days": 10
                    },
                    {
                        "active_ingredient": "Tebuconazole 250g/l",
                        "trade_names": ["Folicur 250EW", "Caramba"],
                        "target_action": "Lưu dẫn ức chế sinh tổng hợp Ergosterol màng tế bào nấm gỉ sắt Puccinia sorghi.",
                        "exact_dosage": "12 - 15 ml / bình 16 lít nước.",
                        "application_timing": "Phun khi các mụn mủ bắt đầu nổi gồ lên mặt lá.",
                        "pre_harvest_interval_days": 14
                    },
                    {
                        "active_ingredient": "Hexaconazole 50g/l",
                        "trade_names": ["Anvil 5SC"],
                        "target_action": "Nội hấp trừ nấm gỉ sắt và đốm lá ngô hiệu quả kinh tế cao.",
                        "exact_dosage": "25 - 30 ml / bình 16 lít nước.",
                        "application_timing": "Phun phòng trừ ở giai đoạn sớm.",
                        "pre_harvest_interval_days": 7
                    }
                ]
            },
            "weather_risk_triggers": {
                "high_risk_temp_min": 16.0,
                "high_risk_temp_max": 24.0,
                "high_risk_humidity_min": 85.0,
                "risk_warning": "CẢNH BÁO BỆNH GỈ SẮT NGÔ: Thời tiết mát 16-24°C và độ ẩm cao > 85%, sương đêm đọng trên lá. Cần kiểm tra hai mặt lá ngô thời kỳ xoáy nõn và phun Amistar Top hoặc Anvil kịp thời."
            }
        },
        "Corn_Northern_Leaf_Blight": {
            "soil_requirements": {
                "preferred_soil": "Đất phù sa bãi bồi, đất thịt pha cát, đất đỏ bazan, pH 5.5 - 7.0.",
                "optimal_ph": "5.5 - 7.0",
                "soil_moisture_pct": "60% - 70%",
                "soil_drainage": "Thoát nước nhanh, không để ngập úng gốc."
            },
            "environmental_conditions": {
                "temperature_range_c": "18 - 27°C (thời tiết ấm ẩm vừa phải).",
                "humidity_pct": "> 90%, mưa ẩm rải rác hoặc sương mù dày.",
                "seasonality": "Phát sinh mạnh từ giai đoạn ngô trỗ cờ phun râu đến chín sữa."
            },
            "pest_vectors_and_co_infections": [
                "Rệp cờ hại ngô (Rhopalosiphum maidis).",
                "Sâu cắn lá ngô."
            ],
            "disease_stages": {
                "initial": "Vết bệnh ban đầu là những đốm nhỏ dài màu xanh xám sũng nước trên các lá bánh tẻ tầng dưới.",
                "progressive": "Vết bệnh lan rộng thành các vệt dài hình thoi lớn đặc trưng (dài 5 - 15cm, rộng 1 - 2cm) màu xám tro hoặc rơm khô.",
                "severe": "Nhiều vệt hình thoi liên kết làm rách nát toàn bộ phiến lá; cây chết khô từ lá dưới lên bắp, làm giảm 40 - 60% năng suất."
            },
            "comprehensive_curative_solutions": {
                "cultural_practices": [
                    "Cày sâu lật đất để vùi lấp tàn dư lá bệnh vụ trước xuống sâu dưới 20cm làm bất hoạt bào tử nấm.",
                    "Luân canh với các cây họ Đậu (đậu tương, đậu xanh) hoặc lúa nước ít nhất 1 vụ.",
                    "Bón đủ phân lân và tăng cường Kali thúc đợt 2 giúp cây cứng cáp, tăng đề kháng thành tế bào lá.",
                    "Tỉa bỏ bớt các lá già sát gốc bị bệnh đem đốt để cắt đứt nguồn lây lan lên tầng lá bắp."
                ],
                "biological_organic": [
                    "Chế phẩm sinh học Nano Chitosan - Bạc kích kháng biểu bì (30ml/bình 16L).",
                    "Dịch chiết lá xoan hoặc dầu neem hữu cơ phun định kỳ xua đuổi sâu hại và cản trở nấm nảy mầm."
                ],
                "chemical_prescriptions": [
                    {
                        "active_ingredient": "Pyraclostrobin 250g/l",
                        "trade_names": ["Headline 250EC"],
                        "target_action": "Thuốc trừ nấm thế hệ mới kích thích sinh trưởng cây xanh lá và diệt trừ nấm hình thoi Exserohilum turcicum.",
                        "exact_dosage": "12 - 15 ml / bình 16 lít nước.",
                        "application_timing": "Phun khi phát hiện vết hình thoi chớm xuất hiện trên tầng lá dưới bắp.",
                        "pre_harvest_interval_days": 14
                    },
                    {
                        "active_ingredient": "Propiconazole 150g/l + Difenoconazole 150g/l",
                        "trade_names": ["Tilt Super 300EC"],
                        "target_action": "Hỗn hợp 2 hoạt chất Triazole lưu dẫn mạnh, làm khô vết cháy lá nhanh sau 24h.",
                        "exact_dosage": "10 - 12 ml / bình 16 lít nước.",
                        "application_timing": "Phun phòng trị ở giai đoạn trổ cờ phun râu.",
                        "pre_harvest_interval_days": 14
                    },
                    {
                        "active_ingredient": "Mancozeb 800g/kg",
                        "trade_names": ["Dithane M-45 80WP"],
                        "target_action": "Phun bao phủ bề mặt tán lá bảo vệ khỏi bào tử bay theo gió.",
                        "exact_dosage": "50 g / bình 16 lít nước.",
                        "application_timing": "Phun phòng đón đầu đợt mưa ẩm.",
                        "pre_harvest_interval_days": 10
                    }
                ]
            },
            "weather_risk_triggers": {
                "high_risk_temp_min": 18.0,
                "high_risk_temp_max": 27.0,
                "high_risk_humidity_min": 90.0,
                "risk_warning": "CẢNH BÁO BỆNH CHÁY LÁ HÌNH THOI NGÔ: Nhiệt độ 18-27°C kết hợp độ ẩm cực cao > 90% thời kỳ trổ cờ. Nguy cơ cháy lá lớn gây giảm năng suất nặng. Cần phun ngay Tilt Super hoặc Headline."
            }
        },
        "Corn_Gray_leaf_spot": {
            "soil_requirements": {
                "preferred_soil": "Đất phù sa, đất thịt giữ ẩm, pH 5.8 - 7.0.",
                "optimal_ph": "5.8 - 7.0",
                "soil_moisture_pct": "65% - 75%",
                "soil_drainage": "Thoát nước tốt, không trồng trên đất trũng ngập."
            },
            "environmental_conditions": {
                "temperature_range_c": "25 - 32°C (thời tiết nóng ẩm nhiệt đới).",
                "humidity_pct": "> 90% (thường gặp trong mùa nồm ẩm hoặc mưa dông ngắt quãng nắng gắt).",
                "seasonality": "Vụ Hè Thu và Thu Đông tại các vùng trồng bắp thâm canh."
            },
            "pest_vectors_and_co_infections": [
                "Rệp muội hại ngô.",
                "Bọ trĩ gây hại thời kỳ cây con."
            ],
            "disease_stages": {
                "initial": "Vết đốm nhỏ hình chữ nhật màu nâu sáng xuất hiện dọc theo các gân phụ của lá.",
                "progressive": "Vết bệnh mở rộng dài ra bị giới hạn chặt chẽ bởi hai đường gân phụ song song, tạo thành hình chữ nhật dài 2 - 8cm màu xám tro đặc trưng.",
                "severe": "Toàn bộ phiến lá bị khô cháy xám xịt, gãy cuống lá ngô, bắp teo tóp và cây dễ đổ ngã."
            },
            "comprehensive_curative_solutions": {
                "cultural_practices": [
                    "Vệ sinh đồng ruộng triệt để: dọn sạch rơm rạ, gốc rạ ngô vụ trước vì nấm Cercospora tồn tại dai dẳng trên tàn dư thực vật.",
                    "Tránh gieo trồng ngô liên tục nhiều vụ trên cùng một thửa ruộng.",
                    "Không gieo quá dày làm bít gió ẩm ướt trong ruộng ngô.",
                    "Bón phân cân đối, tăng cường phân hữu cơ vi sinh và phân Kali."
                ],
                "biological_organic": [
                    "Chế phẩm vi sinh Bacillus subtilis kết hợp nấm Chaetomium kháng nấm lá (35g/bình 16L).",
                    "Nano Bạc Đồng hữu cơ 30ml/bình 16L."
                ],
                "chemical_prescriptions": [
                    {
                        "active_ingredient": "Azoxystrobin 200g/l + Cyproconazole 80g/l",
                        "trade_names": ["Priori Xtra 280SC"],
                        "target_action": "Thuốc đặc trị nấm Cercospora đốm xám, kéo dài tuổi thọ bộ lá ngô và kích thích bắp đầy hạt.",
                        "exact_dosage": "15 - 20 ml / bình 16 lít nước.",
                        "application_timing": "Phun khi phát hiện các đốm chữ nhật đầu tiên xuất hiện ở tầng lá dưới.",
                        "pre_harvest_interval_days": 14
                    },
                    {
                        "active_ingredient": "Tebuconazole 125g/l + Trifloxystrobin 250g/l",
                        "trade_names": ["Nativo 750WG"],
                        "target_action": "Hỗn hợp 2 hoạt chất diệt nấm phổ rộng, hiệu ứng xanh lá (greening effect).",
                        "exact_dosage": "6 - 8 g / bình 16 lít nước.",
                        "application_timing": "Phun luân phiên chống kháng thuốc.",
                        "pre_harvest_interval_days": 14
                    }
                ]
            },
            "weather_risk_triggers": {
                "high_risk_temp_min": 25.0,
                "high_risk_temp_max": 32.0,
                "high_risk_humidity_min": 90.0,
                "risk_warning": "CẢNH BÁO BỆNH ĐỐM LÁ ĐỐM XÁM CHỮ NHẬT NGÔ: Thời tiết nóng ẩm 25-32°C và độ ẩm cực cao > 90%. Vết bệnh hình chữ nhật lan nhanh, cần phun ngay Priori Xtra hoặc Nativo."
            }
        },
        "Corn_healthy": {
            "soil_requirements": {
                "preferred_soil": "Đất phù sa bãi bồi ven sông, đất thịt pha cát, đất đỏ bazan tầng canh tác sâu > 35cm, tơi xốp, giàu mùn, pH 6.0 - 7.0.",
                "optimal_ph": "6.0 - 7.0",
                "soil_moisture_pct": "65% - 75%",
                "soil_drainage": "Thoát nước chủ động, tưới tiêu thuận tiện."
            },
            "environmental_conditions": {
                "temperature_range_c": "Nhiệt độ tối ưu: 22 - 30°C.",
                "humidity_pct": "65% - 75%.",
                "seasonality": "Phù hợp cả vụ Xuân, vụ Hè Thu và vụ Đông."
            },
            "pest_vectors_and_co_infections": [],
            "disease_stages": {
                "status": "Cây ngô sinh trưởng khỏe mạnh, thân to mập, bộ rễ chân kiềng vững chắc, lá xanh mượt bản rộng, cờ bung phấn tốt và bắp dài đều hạt đến chóp."
            },
            "comprehensive_curative_solutions": {
                "cultural_practices": [
                    "Quy trình bón phân N-P-K theo công thức thâm canh: 150kg N : 80kg P2O5 : 100kg K2O trên 1 hecta kết hợp 10 tấn phân chuồng hoai mục.",
                    "Chia làm 3 đợt bón thúc: Đợt 1 khi ngô 3-4 lá, đợt 2 khi ngô 7-9 lá xoáy nõn, đợt 3 trước khi ngô trổ cờ 7 ngày.",
                    "Làm cỏ, xới xáo và vun gốc kết hợp các đợt bón phân giúp cây chống đổ ngã."
                ],
                "biological_organic": [
                    "Bón phân hữu cơ vi sinh định kỳ và tưới chế phẩm EM cải tạo đất."
                ],
                "chemical_prescriptions": []
            },
            "weather_risk_triggers": {
                "high_risk_temp_min": 38.0,
                "high_risk_temp_max": 45.0,
                "high_risk_humidity_min": 95.0,
                "risk_warning": "Thời tiết rất thuận lợi cho ngô phát triển khỏe mạnh. Tiếp tục duy trì chế độ nước tưới và bón phân đúng chu kỳ sinh trưởng."
            }
        }
    }

    # Merge enriched data into data['diseases']
    for d_id, enrich_dict in agronomy_data.items():
        if d_id in data['diseases']:
            data['diseases'][d_id].update(enrich_dict)

    with open(KB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Successfully enriched {len(agronomy_data)} diseases with comprehensive agronomy data in {KB_PATH}!")

if __name__ == '__main__':
    enrich()
