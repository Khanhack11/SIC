"""Append agronomic, weather risk, and chat assistant methods to advisory_engine.py."""
from pathlib import Path

FILE_PATH = Path('backend/advisory_engine.py')

def update():
    content = FILE_PATH.read_text(encoding='utf-8')
    if 'get_agronomy_detail' in content:
        print('Methods already exist in advisory_engine.py')
        return

    new_methods = '''
    def get_agronomy_detail(self, disease_id: str):
        """Return full agronomy profile including soil, climate, pest vectors, and curative prescriptions."""
        raw = self.diseases.get(disease_id)
        if raw is None:
            return None
        return deepcopy(raw)

    def assess_weather_risk(self, temp_c: float, humidity_pct: float, crop=None):
        """Evaluate weather sensor data against high-risk epidemic triggers."""
        alerts = []
        for d_id, info in self.diseases.items():
            if info.get('is_healthy', False) or 'healthy' in d_id.lower():
                continue
            if crop and crop.lower() not in info.get('crop', '').lower():
                continue
            triggers = info.get('weather_risk_triggers', {})
            if not triggers:
                continue
            min_temp = triggers.get('high_risk_temp_min', -100.0)
            max_temp = triggers.get('high_risk_temp_max', 100.0)
            min_humidity = triggers.get('high_risk_humidity_min', 0.0)
            if min_temp <= temp_c <= max_temp and humidity_pct >= min_humidity:
                alerts.append({
                    'disease_id': d_id,
                    'crop': info.get('crop'),
                    'vietnamese_name': info.get('vietnamese_name'),
                    'risk_level': 'BÁO ĐỘNG ĐỎ' if 'BÁO ĐỘNG ĐỎ' in triggers.get('risk_warning', '') else 'CẢNH BÁO CAO',
                    'temperature_c': temp_c,
                    'humidity_pct': humidity_pct,
                    'warning_message': triggers.get('risk_warning'),
                    'preventive_actions': info.get('comprehensive_curative_solutions', {}).get('cultural_practices', [])[:2],
                    'recommended_prescriptions': info.get('comprehensive_curative_solutions', {}).get('chemical_prescriptions', [])[:1]
                })
        return alerts

    def agricultural_chat(self, query: str):
        """Agricultural Chatbot Knowledge Assistant answering questions about crops, soil, pests, and treatments."""
        q = query.lower().strip()
        matched_disease = None
        matched_info = None

        for d_id, info in self.diseases.items():
            names = [
                d_id.lower(),
                info.get('vietnamese_name', '').lower(),
                info.get('scientific_name', '').lower(),
                info.get('english_name', '').lower()
            ]
            if any(name in q for name in names if name):
                matched_disease = d_id
                matched_info = info
                break

        if not matched_info:
            for crop_name in ['cà chua', 'tomato', 'khoai tây', 'potato', 'ngô', 'bắp', 'corn']:
                if crop_name in q:
                    for d_id, info in self.diseases.items():
                        if crop_name in info.get('crop', '').lower():
                            matched_disease = d_id
                            matched_info = info
                            break
                    if matched_info:
                        break

        if not matched_info:
            return {
                'answer': (
                    "Xin chào bà con! Tôi là Trợ lý AI Tư vấn Nông nghiệp SIC Group 5. "
                    "Tôi có thể hỗ trợ bà con giải đáp: \\n"
                    "1. Chẩn đoán và trị dứt điểm các bệnh trên Cà chua, Khoai tây, Ngô.\\n"
                    "2. Yêu cầu đất trồng, độ pH, độ ẩm và kỹ thuật bón phân.\\n"
                    "3. Nhận diện sâu hại môi giới truyền bệnh (bọ trĩ, rầy mềm, rệp sáp).\\n"
                    "4. Liều lượng thuốc BVTV chuẩn (Amistar Top, Ridomil Gold, Tilt Super, Score, Daconil) và thời gian cách ly (PHI).\\n"
                    "5. Cảnh báo nguy cơ dịch bệnh theo thời tiết (nhiệt độ, độ ẩm).\\n"
                    "Bà con vui lòng nhập tên cây trồng hoặc triệu chứng bệnh muốn hỏi nhé!"
                ),
                'matched_disease': None,
                'references': []
            }

        crop = matched_info.get('crop', '')
        v_name = matched_info.get('vietnamese_name', '')
        soil = matched_info.get('soil_requirements', {})
        env = matched_info.get('environmental_conditions', {})
        solutions = matched_info.get('comprehensive_curative_solutions', {})
        cultural = solutions.get('cultural_practices', [])
        biological = solutions.get('biological_organic', [])
        chemical = solutions.get('chemical_prescriptions', [])

        if any(w in q for w in ['đất', 'ph', 'ẩm', 'tưới', 'thoát nước']):
            answer = (
                f"**Yêu cầu đất trồng & môi trường cho {crop} ({v_name}):**\\n"
                f"- **Loại đất phù hợp:** {soil.get('preferred_soil', 'Đất thịt nhẹ, tơi xốp')}\\n"
                f"- **Độ pH tối ưu:** {soil.get('optimal_ph', '6.0 - 6.8')}\\n"
                f"- **Độ ẩm đất lý tưởng:** {soil.get('soil_moisture_pct', '60% - 70%')}\\n"
                f"- **Chế độ thoát nước:** {soil.get('soil_drainage', 'Cần lên luống cao, thoát nước tốt sau mưa.')}\\n"
                f"- **Môi trường & mùa vụ:** {env.get('seasonality', 'Vụ Đông Xuân và Xuân Hè.')}"
            )
        elif any(w in q for w in ['thuốc', 'liều lượng', 'phun', 'hoạt chất', 'chữa', 'trị', 'phi', 'cách ly']):
            chem_text = ''
            for idx, c in enumerate(chemical, 1):
                chem_text += (
                    f"\\n**{idx}. Hoạt chất:** {c.get('active_ingredient')} "
                    f"*(Thương phẩm: {', '.join(c.get('trade_names', []))})*\\n"
                    f"   - **Liều lượng pha:** {c.get('exact_dosage')}\\n"
                    f"   - **Thời điểm xử lý:** {c.get('application_timing')}\\n"
                    f"   - **Thời gian cách ly (PHI):** {c.get('pre_harvest_interval_days')} ngày.\\n"
                )
            answer = (
                f"**Phác đồ thuốc điều trị & liều lượng chuẩn cho {v_name}:**\\n"
                f"{chem_text or 'Hiện chưa ghi nhận phác đồ hóa học.'}\\n"
                f"**Lưu ý an toàn:** Tuân thủ nguyên tắc 4 đúng (đúng thuốc, đúng liều, đúng lúc, đúng cách) và mang đồ bảo hộ khi phun."
            )
        elif any(w in q for w in ['sâu', 'rầy', 'rệp', 'bọ', 'côn trùng', 'môi giới']):
            pests = matched_info.get('pest_vectors_and_co_infections', [])
            answer = (
                f"**Sâu hại môi giới và bệnh đi kèm trên {crop} ({v_name}):**\\n"
                + "\\n".join(f"- {p}" for p in pests)
                + "\\n\\n**Khuyến nghị:** Cần phòng trừ triệt để sâu chích hút từ giai đoạn cây con để ngăn nấm và virus lây lan."
            )
        else:
            answer = (
                f"**Thông tin chuyên sâu về {v_name} trên {crop}:**\\n\\n"
                f"1. **Tác nhân:** {matched_info.get('scientific_name')} ({matched_info.get('pathogen_type')}).\\n"
                f"2. **Đất & Môi trường:** Đất pH {soil.get('optimal_ph', '6.0-6.8')}, nhiệt độ thích hợp {env.get('temperature_range_c', 'ấm ẩm')}.\\n"
                f"3. **Biện pháp canh tác triệt để:**\\n"
                + "\\n".join(f"   - {c}" for c in cultural[:2]) + "\\n"
                f"4. **Phòng trừ sinh học:**\\n"
                + "\\n".join(f"   - {b}" for b in biological[:2]) + "\\n"
                f"5. **Phác đồ hóa học chính:** "
                + (f"{chemical[0].get('active_ingredient')} ({', '.join(chemical[0].get('trade_names', []))}), liều: {chemical[0].get('exact_dosage')}" if chemical else "Ưu tiên biện pháp sinh học.")
            )

        return {
            'answer': answer,
            'matched_disease': matched_disease,
            'crop': crop,
            'references': matched_info.get('sources', [])
        }
'''
    FILE_PATH.write_text(content.rstrip() + '\n' + new_methods, encoding='utf-8')
    print('Successfully updated backend/advisory_engine.py')

if __name__ == '__main__':
    update()
