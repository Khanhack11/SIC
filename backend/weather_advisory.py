"""
Agricultural Weather & Disease Epidemic Early Warning Engine (Pillar 3)
Calculates Epidemiological Disease Index (EDI) based on agro-climatic conditions
(temperature, relative humidity, rainfall, leaf wetness duration) to forecast
outbreak risks for Late Blight, Early Blight, Common Rust, and Bacterial Spot.
"""

from typing import Dict, Any, List
import math
import time


REGIONS_CATALOG = [
    {
        "id": "thai_nguyen",
        "name": "Thái Nguyên (Trung du & Đồi núi phía Bắc)",
        "crops": ["Cà chua", "Khoai tây", "Ngô"],
        "station": "Trạm Khí tượng ĐH Nông Lâm Thái Nguyên",
        "current_weather": {
            "temp_c": 21.5,
            "humidity_pct": 88,
            "rain_mm": 12.0,
            "condition": "Mưa nhỏ rải rác, sương mù nhẹ",
            "wind_kmh": 8.5,
            "leaf_wetness_hours": 14
        }
    },
    {
        "id": "ha_noi",
        "name": "Hà Nội / ĐBSH (Vùng rau màu chuyên canh vụ Đông)",
        "crops": ["Cà chua", "Khoai tây"],
        "station": "Trạm Khí tượng Ba Vì - Gia Lâm",
        "current_weather": {
            "temp_c": 23.0,
            "humidity_pct": 82,
            "rain_mm": 4.5,
            "condition": "Nhiều mây, độ ẩm cao",
            "wind_kmh": 11.0,
            "leaf_wetness_hours": 9
        }
    },
    {
        "id": "da_lat",
        "name": "Lâm Đồng - Đà Lạt (Nông nghiệp Công nghệ cao)",
        "crops": ["Cà chua", "Khoai tây"],
        "station": "Trạm Khí tượng Đơn Dương - Đà Lạt",
        "current_weather": {
            "temp_c": 17.5,
            "humidity_pct": 94,
            "rain_mm": 25.0,
            "condition": "Mưa phùn, sương mù ẩm ướt",
            "wind_kmh": 6.0,
            "leaf_wetness_hours": 18
        }
    },
    {
        "id": "son_la",
        "name": "Sơn La / Tây Bắc (Vùng trồng ngô sinh khối lớn)",
        "crops": ["Ngô", "Rau màu"],
        "station": "Trạm Khí tượng Mộc Châu",
        "current_weather": {
            "temp_c": 19.0,
            "humidity_pct": 78,
            "rain_mm": 0.0,
            "condition": "Nắng dịu, se lạnh về đêm",
            "wind_kmh": 14.0,
            "leaf_wetness_hours": 6
        }
    },
    {
        "id": "tay_nguyen",
        "name": "Đắk Lắk - Tây Nguyên (Vùng ngô lai & rau củ)",
        "crops": ["Ngô", "Khoai tây"],
        "station": "Trạm Khí tượng Buôn Ma Thuột",
        "current_weather": {
            "temp_c": 26.5,
            "humidity_pct": 74,
            "rain_mm": 2.0,
            "condition": "Nắng gián đoạn, chiều tối có mưa rào",
            "wind_kmh": 12.0,
            "leaf_wetness_hours": 5
        }
    }
]


def calculate_disease_risks(weather: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Computes disease outbreak risk levels based on agronomic epidemiology criteria:
    - Late Blight (Phytophthora infestans): Optimal 15-22°C, RH > 85%, Leaf wetness >= 10h.
    - Early Blight (Alternaria solani): Optimal 24-29°C, RH > 75%, wet-dry cycles.
    - Bacterial Spot (Xanthomonas): Warm 25-30°C, heavy rains / rain splashing.
    - Common Rust (Puccinia sorghi): Cool-warm 16-24°C, RH > 80%, high morning dew.
    """
    temp = weather.get("temp_c", 22.0)
    rh = weather.get("humidity_pct", 75)
    rain = weather.get("rain_mm", 0.0)
    lwd = weather.get("leaf_wetness_hours", 6)

    risks = []

    # 1. Late Blight (Mốc sương / Mốc muộn)
    # Phytophthora sporangia germinate best at 15-20°C with high humidity
    lb_score = 0
    if 14.0 <= temp <= 22.5:
        lb_score += 40
    elif 10.0 <= temp < 14.0 or 22.5 < temp <= 25.0:
        lb_score += 20

    if rh >= 90:
        lb_score += 35
    elif rh >= 80:
        lb_score += 25
    elif rh >= 70:
        lb_score += 10

    if lwd >= 12:
        lb_score += 25
    elif lwd >= 8:
        lb_score += 15

    lb_level, lb_badge, lb_color = _grade_risk(lb_score)
    risks.append({
        "disease_id": "Tomato_Late_blight",
        "disease_name": "Bệnh Mốc sương / Cháy lá muộn (Late Blight)",
        "crops": ["Cà chua", "Khoai tây"],
        "risk_score": lb_score,
        "risk_level": lb_level,
        "badge_color": lb_color,
        "favorable_factors": f"Nhiệt độ thích hợp nấm ({temp}°C), độ ẩm bão hòa ({rh}%), đọng sương {lwd}h.",
        "preventive_action": (
            "⚠️ NGUY CƠ BÙNG PHÁT CỰC KỲ CAO! Cần phun phòng ngay bằng thuốc nội hấp Ridomil Gold 68WG hoặc Mancozeb trước đợt mưa kéo dài. Tuyệt đối không tưới phun mưa vào buổi chiều tối."
            if lb_score >= 70 else
            "Tỉa bớt lá già sát gốc, khơi thông rãnh thoát nước, theo dõi sát vết bệnh ẩm nhũn mặt dưới lá."
        )
    })

    # 2. Early Blight (Đốm vòng / Cháy lá sớm)
    eb_score = 0
    if 23.0 <= temp <= 30.0:
        eb_score += 45
    elif 20.0 <= temp < 23.0:
        eb_score += 30

    if rh >= 80:
        eb_score += 30
    elif rh >= 65:
        eb_score += 20

    if rain > 5.0 or lwd >= 8:
        eb_score += 25
    elif lwd >= 4:
        eb_score += 15

    eb_level, eb_badge, eb_color = _grade_risk(eb_score)
    risks.append({
        "disease_id": "Tomato_Early_blight",
        "disease_name": "Bệnh Đốm vòng / Cháy lá sớm (Early Blight)",
        "crops": ["Cà chua", "Khoai tây"],
        "risk_score": eb_score,
        "risk_level": eb_level,
        "badge_color": eb_color,
        "favorable_factors": f"Biên độ nhiệt ẩm ấm ({temp}°C, {rh}%), lá ẩm xen kẽ nắng ráo.",
        "preventive_action": "Phun phòng hoạt chất Azoxystrobin + Difenoconazole (Score 250EC, Amistar Top) khi xuất hiện đốm nâu đầu tiên ở tầng lá gốc."
    })

    # 3. Common Rust (Gỉ sắt ngô)
    rust_score = 0
    if 16.0 <= temp <= 25.0:
        rust_score += 45
    if rh >= 75:
        rust_score += 35
    if lwd >= 6:
        rust_score += 20

    rust_level, rust_badge, rust_color = _grade_risk(rust_score)
    risks.append({
        "disease_id": "Corn_Common_rust",
        "disease_name": "Bệnh Gỉ sắt ngô (Common Rust)",
        "crops": ["Ngô (Bắp)"],
        "risk_score": rust_score,
        "risk_level": rust_level,
        "badge_color": rust_color,
        "favorable_factors": f"Nhiệt độ ôn hòa ({temp}°C), độ ẩm cao và sương mai đọng lâu trên phiến lá.",
        "preventive_action": "Bón cân đối N-P-K, tránh bón thừa đạm (ure). Khi tỷ lệ bệnh vượt 5% lá, phun Mancozeb hoặc Tebuconazole."
    })

    # 4. Bacterial Spot (Đốm vi khuẩn)
    bac_score = 0
    if temp >= 24.0:
        bac_score += 35
    if rain >= 10.0:
        bac_score += 45
    elif rain >= 2.0:
        bac_score += 25
    if rh >= 80:
        bac_score += 20

    bac_level, bac_badge, bac_color = _grade_risk(bac_score)
    risks.append({
        "disease_id": "Tomato_Bacterial_spot",
        "disease_name": "Bệnh Đốm vi khuẩn (Bacterial Spot)",
        "crops": ["Cà chua"],
        "risk_score": bac_score,
        "risk_level": bac_level,
        "badge_color": bac_color,
        "favorable_factors": f"Mưa rào làm giọt bắn vi khuẩn phát tán ({rain}mm), thời tiết ấm ẩm.",
        "preventive_action": "Phun luân phiên thuốc gốc đồng (Copper Hydroxide) hoặc hoạt chất Kasugamycin. Ngừng làm cỏ khi lá còn ướt để tránh trầy xước tạo vết thương hở."
    })

    return risks


def _grade_risk(score: int) -> tuple:
    if score >= 75:
        return "RẤT CAO (Khẩn cấp)", "danger", "#d32f2f"
    elif score >= 50:
        return "CAO (Báo động)", "warning", "#f57c00"
    elif score >= 30:
        return "TRUNG BÌNH (Theo dõi)", "info", "#1976d2"
    else:
        return "THẤP (An toàn)", "safe", "#388e3c"


def get_agro_weather_advisory(region_id: str = "thai_nguyen") -> Dict[str, Any]:
    """
    Returns full weather data, forecast and epidemiological disease risks for selected region.
    """
    selected_region = None
    for r in REGIONS_CATALOG:
        if r["id"] == region_id:
            selected_region = r
            break

    if not selected_region:
        selected_region = REGIONS_CATALOG[0]

    weather = selected_region["current_weather"]
    risks = calculate_disease_risks(weather)

    # General recommendations based on highest risk
    max_risk = max(r["risk_score"] for r in risks)
    if max_risk >= 75:
        overall_advice = "🚨 CẢNH BÁO NÔNG VỤ ĐỎ: Điều kiện vi khí hậu cực kỳ thuận lợi cho dịch bệnh phát tán nhanh (đặc biệt là nấm mốc ẩm). Khuyến cáo bà con kiểm tra đồng ruộng hàng ngày và chủ động các biện pháp phun phòng sớm."
    elif max_risk >= 50:
        overall_advice = "⚠️ BÁO ĐỘNG VÀNG: Độ ẩm và thời gian ướt lá ở mức nguy cơ cao. Chú ý thoát nước tốt ở các rãnh luống, tỉa lá gốc thông thoáng."
    else:
        overall_advice = "✅ ĐIỀU KIỆN AN TOÀN: Thời tiết thuận lợi cho sinh trưởng cây trồng, áp lực sâu bệnh thấp."

    return {
        "region_id": selected_region["id"],
        "region_name": selected_region["name"],
        "station": selected_region["station"],
        "crops": selected_region["crops"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weather": weather,
        "disease_risks": risks,
        "overall_advisory": overall_advice,
        "available_regions": [{"id": r["id"], "name": r["name"]} for r in REGIONS_CATALOG]
    }
