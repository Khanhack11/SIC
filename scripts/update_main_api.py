"""Add agronomy detail, weather advisory, chatbot, and dataset insights endpoints to backend/main.py."""
from pathlib import Path

FILE_PATH = Path('backend/main.py')

def update():
    content = FILE_PATH.read_text(encoding='utf-8')
    if '/api/weather-advisory' in content:
        print('Endpoints already exist in main.py')
        return

    new_endpoints = '''

# Agronomy & Chatbot Request Models
class WeatherRiskRequest(BaseModel):
    temperature_c: float
    humidity_pct: float
    crop: Optional[str] = None


class ChatMessageRequest(BaseModel):
    message: str


@app.get("/api/diseases/{disease_id}/agronomy", tags=["Knowledge Base"])
def get_disease_agronomy(disease_id: str) -> Dict[str, Any]:
    """Retrieves full agronomic profile including soil, climate, pest vectors, and exact chemical dosages."""
    record = advisory_engine.get_agronomy_detail(disease_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Không tìm thấy hồ sơ nông học cho mã: '{disease_id}'"
        )
    return {
        "disease_id": disease_id,
        "agronomy_data": record,
        "disclaimer": advisory_engine.disclaimer
    }


@app.post("/api/weather-advisory", tags=["Agricultural Advisory"])
def weather_advisory(req: WeatherRiskRequest) -> Dict[str, Any]:
    """Evaluates environmental telemetry (temperature, humidity) against epidemic triggers."""
    alerts = advisory_engine.assess_weather_risk(req.temperature_c, req.humidity_pct, req.crop)
    return {
        "status": "ALERT" if alerts else "SAFE",
        "telemetry": {
            "temperature_c": req.temperature_c,
            "humidity_pct": req.humidity_pct,
            "crop_filter": req.crop or "Tất cả cây trồng"
        },
        "total_risks_detected": len(alerts),
        "alerts": alerts,
        "general_advice": (
            "Điều kiện thời tiết đang ở ngưỡng nguy cơ cao bùng phát dịch bệnh. Khuyến cáo kiểm tra đồng ruộng và chuẩn bị thuốc BVTV phòng trị kịp thời."
            if alerts else "Điều kiện thời tiết hiện tại trong ngưỡng an toàn, ít nguy cơ bùng phát dịch hại lớn."
        )
    }


@app.post("/api/chat", tags=["Agricultural Advisory"])
def farm_chat_assistant(req: ChatMessageRequest) -> Dict[str, Any]:
    """AI Agricultural Chatbot Assistant answering farmers questions about soil, crops, diseases, and dosages."""
    if not req.message.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Vui lòng nhập câu hỏi.")
    return advisory_engine.agricultural_chat(req.message)


@app.get("/api/dataset-insights", tags=["Knowledge Base"])
def get_dataset_insights() -> Dict[str, Any]:
    """Returns dataset survey comparing Kaggle New Plant Diseases Dataset vs. Real-World Field datasets."""
    kb_data = advisory_engine.data
    return {
        "kaggle_benchmark": kb_data.get("kaggle_reference_dataset", {}),
        "our_field_v2_dataset": {
            "total_records": 20902,
            "training_samples": 16225,
            "validation_samples": 2740,
            "test_samples": 1937,
            "single_leaf_crops_extracted": 3160,
            "international_data_sources": [
                "PlantVillage (Lab standard)",
                "PlantDoc (Real-world in-the-wild field conditions)",
                "CCMT (Crop-Care Multi-Country)",
                "BARI (Bangladesh Agricultural Research Institute)",
                "Holeta (Ethiopia Potato Late Blight)",
                "IIT Kharagpur (India Potato Leaf Dataset)",
                "Indonesia Potato Field Survey",
                "Pakistan Tomato Camera Captures"
            ],
            "advantages": [
                "Không bị overfitting phông nền phòng thí nghiệm",
                "Hỗ trợ crop bounding-box lá đơn ngoài thực địa",
                "Tăng trưởng F1 thực địa +39.03 pp trên PlantDoc"
            ]
        }
    }
'''

    # Insert before `if __name__ == "__main__":`
    target = 'if __name__ == "__main__":'
    if target in content:
        idx = content.index(target)
        updated = content[:idx] + new_endpoints + '\n\n' + content[idx:]
    else:
        updated = content + '\n' + new_endpoints

    FILE_PATH.write_text(updated, encoding='utf-8')
    print('Successfully updated backend/main.py with new endpoints!')

if __name__ == '__main__':
    update()
