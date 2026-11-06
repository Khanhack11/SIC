"""Refine agricultural_chat in backend/advisory_engine.py with precise keyword prioritization."""
from pathlib import Path

FILE_PATH = Path('backend/advisory_engine.py')

def update():
    content = FILE_PATH.read_text(encoding='utf-8')
    
    old_method_start = '    def agricultural_chat(self, query: str):'
    assert old_method_start in content
    idx = content.find(old_method_start)
    
    refined_method = '''    def agricultural_chat(self, query: str):
        """Agricultural Chatbot Knowledge Assistant answering questions about crops, soil, pests, and treatments."""
        q = query.lower().strip()
        q_clean = strip_accents(q)

        disease_keywords = {
            'Tomato_Late_blight': ['suong mai ca chua', 'late blight tomato', 'suong mai', 'moc suong', 'phytophthora infestans', 'tomato_late_blight'],
            'Potato_Late_blight': ['suong mai khoai tay', 'late blight potato', 'potato_late_blight'],
            'Tomato_Early_blight': ['dom vong ca chua', 'chay la som ca chua', 'early blight tomato', 'dom vong', 'chay la som', 'alternaria solani', 'tomato_early_blight'],
            'Potato_Early_blight': ['dom vong khoai tay', 'chay la som khoai tay', 'early blight potato', 'potato_early_blight'],
            'Tomato_Septoria_leaf_spot': ['septoria', 'dom la septoria', 'tomato_septoria_leaf_spot'],
            'Corn_Common_rust': ['gi sat', 'ri sat', 'rust', 'puccinia sorghi', 'corn_common_rust'],
            'Corn_Northern_Leaf_Blight': ['chay la hinh thoi', 'dom la lon ngo', 'dom la lon', 'northern leaf blight', 'exserohilum turcicum', 'corn_northern_leaf_blight'],
            'Corn_Gray_leaf_spot': ['dom xam', 'gray leaf spot', 'cercospora', 'dom chu nhat', 'corn_gray_leaf_spot'],
            'Tomato_healthy': ['ca chua khoe', 'tomato healthy', 'cham soc ca chua', 'tomato_healthy'],
            'Potato_healthy': ['khoai tay khoe', 'potato healthy', 'cham soc khoai tay', 'potato_healthy'],
            'Corn_healthy': ['ngo khoe', 'bap khoe', 'corn healthy', 'cham soc ngo', 'corn_healthy']
        }

        matched_disease = None
        matched_info = None

        # 1. Match specific disease by high-priority keywords (longest phrase first)
        best_match = None
        best_match_len = 0
        for d_id, kws in disease_keywords.items():
            for kw in kws:
                kw_clean = strip_accents(kw)
                if kw_clean in q_clean:
                    # check crop context if ambiguous (e.g. suong mai)
                    if kw_clean in ['suong mai', 'moc suong', 'late blight']:
                        if any(c in q_clean for c in ['khoai tay', 'potato']):
                            d_id = 'Potato_Late_blight'
                        elif any(c in q_clean for c in ['ca chua', 'tomato']):
                            d_id = 'Tomato_Late_blight'
                    elif kw_clean in ['dom vong', 'chay la som', 'early blight']:
                        if any(c in q_clean for c in ['khoai tay', 'potato']):
                            d_id = 'Potato_Early_blight'
                        elif any(c in q_clean for c in ['ca chua', 'tomato']):
                            d_id = 'Tomato_Early_blight'
                    if len(kw_clean) > best_match_len:
                        best_match = d_id
                        best_match_len = len(kw_clean)

        if best_match:
            matched_disease = best_match
            matched_info = self.diseases.get(best_match)

        # 2. Fallback to crop search if no specific disease was identified
        if not matched_info:
            crop_map = {
                'ca chua': 'Tomato_Early_blight',
                'tomato': 'Tomato_Early_blight',
                'khoai tay': 'Potato_Early_blight',
                'potato': 'Potato_Early_blight',
                'ngo': 'Corn_Common_rust',
                'bap': 'Corn_Common_rust',
                'corn': 'Corn_Common_rust'
            }
            for c_term, default_d in crop_map.items():
                if c_term in q_clean:
                    matched_disease = default_d
                    matched_info = self.diseases.get(default_d)
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

        if any(w in q_clean for w in ['dat', 'ph', 'am', 'tuoi', 'thoat nuoc']):
            answer = (
                f"**Yêu cầu đất trồng & môi trường cho {crop} ({v_name}):**\\n"
                f"- **Loại đất phù hợp:** {soil.get('preferred_soil', 'Đất thịt nhẹ, tơi xốp')}\\n"
                f"- **Độ pH tối ưu:** {soil.get('optimal_ph', '6.0 - 6.8')}\\n"
                f"- **Độ ẩm đất lý tưởng:** {soil.get('soil_moisture_pct', '60% - 70%')}\\n"
                f"- **Chế độ thoát nước:** {soil.get('soil_drainage', 'Cần lên luống cao, thoát nước tốt sau mưa.')}\\n"
                f"- **Môi trường & mùa vụ:** {env.get('seasonality', 'Vụ Đông Xuân và Xuân Hè.')}"
            )
        elif any(w in q_clean for w in ['thuoc', 'lieu luong', 'phun', 'hoat chat', 'chua', 'tri', 'phi', 'cach ly']):
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
        elif any(w in q_clean for w in ['sau', 'ray', 'rep', 'bo', 'con trung', 'moi gioi']):
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
    content = content[:idx] + refined_method
    FILE_PATH.write_text(content, encoding='utf-8')
    print('Successfully refined agricultural_chat keyword matching!')

if __name__ == '__main__':
    update()
