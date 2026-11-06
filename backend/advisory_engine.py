"""
Agricultural Advisory Engine (Subsystem B)
Author: Nguyen Quoc Khanh (Backend Developer & Database Architect)
SIC Capstone Project - Group 5
"""

import unicodedata
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, List

from backend.config import KNOWLEDGE_BASE_PATH, CONFIDENCE_THRESHOLD



def strip_accents(s: str) -> str:
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return s.replace('đ', 'd').replace('Đ', 'D')


class AdvisoryKnowledgeBase:
    """
    Expert Agricultural Advisory Engine.
    Handles knowledge base retrieval and confidence-gated decision logic.
    """

    def __init__(self, kb_path: Path = KNOWLEDGE_BASE_PATH, threshold: float = CONFIDENCE_THRESHOLD):
        self.kb_path = kb_path
        self.threshold = threshold
        self.data: Dict[str, Any] = {}
        self.load_knowledge_base()

    def load_knowledge_base(self) -> None:
        """Loads and parses the structured advisory database JSON."""
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Không tìm thấy cơ sở dữ liệu tri thức tại: {self.kb_path}")

        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.threshold = self.data.get("confidence_threshold", self.threshold)
        print(f"[AdvisoryEngine] Loaded {len(self.diseases)} crop disease profiles from Knowledge Base.")

    @property
    def diseases(self) -> Dict[str, Any]:
        return self.data.get("diseases", {})

    @property
    def disclaimer(self) -> str:
        return self.data.get("disclaimer_default", "")

    def get_disease_by_id(self, disease_id: str) -> Optional[Dict[str, Any]]:
        """Return a request-local record, withholding unreviewed treatment instructions."""
        raw = self.diseases.get(disease_id)
        if raw is None:
            return None
        record = deepcopy(raw)
        review = record.get("review", {})
        if review.get("chemical_status") != "verified_local_label":
            record["chemical_controls"] = []
            record["organic_biological_controls"] = []
            record["treatment_notice"] = (
                "Chưa có kiểm định chuyên môn về sản phẩm, liều lượng và thời gian cách ly tại Việt Nam. "
                "Hệ thống chưa cung cấp hướng dẫn dùng thuốc; hãy tham vấn cán bộ BVTV địa phương."
            )
        return record

    def list_all_diseases(self) -> List[Dict[str, Any]]:
        """Returns brief summary list of all known diseases in the database."""
        summary = []
        for d_id, info in self.diseases.items():
            summary.append({
                "disease_id": d_id,
                "crop": info.get("crop"),
                "vietnamese_name": info.get("vietnamese_name"),
                "english_name": info.get("english_name"),
                "scientific_name": info.get("scientific_name"),
                "pathogen_type": info.get("pathogen_type"),
                "severity_level": info.get("severity_level"),
            })
        return summary

    def process_diagnosis(self, inference_result: Dict[str, Any], threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Confidence-Gated Decision Logic (Subsystem B Core):
        1. Evaluates confidence against the configured threshold.
        2. If confidence >= threshold: returns full agronomic advisory card.
        3. If confidence < threshold: returns warning prompt for clearer photos.
        """
        predicted_class = inference_result["predicted_class"]
        confidence = inference_result["confidence"]
        applied_threshold = self.threshold if threshold is None else threshold
        if not 0.0 <= applied_threshold <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1")
        latency_ms = inference_result.get("latency_ms", 0.0)

        # Retrieve disease details from database
        disease_record = self.get_disease_by_id(predicted_class)
        review_reason = None
        if disease_record is None:
            review_reason = 'unsupported_class'
        elif inference_result.get('visible_damage') and inference_result.get('is_healthy'):
            review_reason = 'visible_damage_conflict'
        elif inference_result.get('field_photo') and inference_result.get('is_healthy'):
            review_reason = 'field_healthy_unverified'
        accepted = confidence >= applied_threshold and review_reason is None

        # Decision Logic: Check confidence gate
        if accepted:
            # Case 1: High confidence prediction
            status = "CONFIRMED"
            message = "Dự đoán đạt ngưỡng hiển thị thông tin tham khảo. Cần đối chiếu triệu chứng thực tế."
            advisory_payload = disease_record
            remediation_notice = None
        else:
            # Case 2: Low confidence prediction (< threshold)
            status = "UNCERTAIN"
            message = (
                f"Độ tin cậy mô hình ({confidence * 100:.1f}%) thấp hơn ngưỡng hiển thị ({applied_threshold * 100:.0f}%). "
                "Chưa đủ độ tin cậy để hiển thị thông tin bệnh."
            )
            # Safe Fallback Guidance
            advisory_payload = {
                "preliminary_assessment": (
                    disease_record.get("vietnamese_name") if disease_record else "Chưa xác định"
                ),
                "guidance_for_farmer": [
                    "1. Chụp lại ảnh lá cây ở cự ly gần (15 - 30cm), lấy nét trực diện vào vết bệnh hoặc đốm lá.",
                    "2. Chụp trong điều kiện ánh sáng tự nhiên đầy đủ, tránh chụp ngược sáng hoặc bóng râm che khuất.",
                    "3. Đảm bảo bề mặt lá sạch, không dính bùn đất hoặc bị rung mờ khi chụp.",
                    "4. Liên hệ trực tiếp với cán bộ Trạm Khuyến nông / Trạm BVTV địa phương để kiểm tra mẫu lá thực địa."
                ]
            }
            remediation_notice = "Khuyến nghị chụp lại ảnh chất lượng cao hơn hoặc tham vấn chuyên gia."
            if review_reason == 'visible_damage_conflict':
                message = 'Bạn đã ghi nhận lá có tổn thương nhưng model dự đoán lớp khỏe. Chưa thể kết luận lá khỏe; cần kiểm tra vết tổn thương hoặc hỏi cán bộ BVTV.'
            elif review_reason == 'unsupported_class':
                message = 'Ảnh có thể nằm ngoài các bệnh được hỗ trợ. Hệ thống chưa xác định được tình trạng lá.'
            elif review_reason == 'field_healthy_unverified':
                message = 'Model nghiêng về lớp khỏe nhưng chưa đủ cơ sở xác nhận sức khỏe cây từ ảnh thực địa. Hãy đối chiếu các lỗ, đốm hoặc biến dạng và chụp riêng từng lá.'

        return {
            "diagnosis_status": status,
            "review_reason": review_reason,
            "diagnostic_confirmation": False,
            "message": message,
            "prediction": {
                "disease_id": predicted_class,
                "crop": inference_result.get("crop"),
                "is_healthy": inference_result.get("is_healthy", False),
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%",
                "threshold_applied": applied_threshold,
                "confidence_gate_passed": accepted,
                "latency_ms": latency_ms,
            },
            "advisory_details": advisory_payload,
            "remediation_notice": remediation_notice,
            "regulatory_disclaimer": self.disclaimer,
        }

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
                    "Tôi có thể hỗ trợ bà con giải đáp: \n"
                    "1. Chẩn đoán và trị dứt điểm các bệnh trên Cà chua, Khoai tây, Ngô.\n"
                    "2. Yêu cầu đất trồng, độ pH, độ ẩm và kỹ thuật bón phân.\n"
                    "3. Nhận diện sâu hại môi giới truyền bệnh (bọ trĩ, rầy mềm, rệp sáp).\n"
                    "4. Liều lượng thuốc BVTV chuẩn (Amistar Top, Ridomil Gold, Tilt Super, Score, Daconil) và thời gian cách ly (PHI).\n"
                    "5. Cảnh báo nguy cơ dịch bệnh theo thời tiết (nhiệt độ, độ ẩm).\n"
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

        if any(w in q_clean for w in ['dat ', 'thoc dat', 'do ph', ' ph ', 'do am', 'tuoi nuoc', 'thoat nuoc']):
            answer = (
                f"**Yêu cầu đất trồng & môi trường cho {crop} ({v_name}):**\n"
                f"- **Loại đất phù hợp:** {soil.get('preferred_soil', 'Đất thịt nhẹ, tơi xốp')}\n"
                f"- **Độ pH tối ưu:** {soil.get('optimal_ph', '6.0 - 6.8')}\n"
                f"- **Độ ẩm đất lý tưởng:** {soil.get('soil_moisture_pct', '60% - 70%')}\n"
                f"- **Chế độ thoát nước:** {soil.get('soil_drainage', 'Cần lên luống cao, thoát nước tốt sau mưa.')}\n"
                f"- **Môi trường & mùa vụ:** {env.get('seasonality', 'Vụ Đông Xuân và Xuân Hè.')}"
            )
        elif any(w in q_clean for w in ['thuoc', 'lieu luong', 'phun', 'hoat chat', 'chua', 'tri', 'phi', 'cach ly']):
            chem_text = ''
            for idx, c in enumerate(chemical, 1):
                chem_text += (
                    f"\n**{idx}. Hoạt chất:** {c.get('active_ingredient')} "
                    f"*(Thương phẩm: {', '.join(c.get('trade_names', []))})*\n"
                    f"   - **Liều lượng pha:** {c.get('exact_dosage')}\n"
                    f"   - **Thời điểm xử lý:** {c.get('application_timing')}\n"
                    f"   - **Thời gian cách ly (PHI):** {c.get('pre_harvest_interval_days')} ngày.\n"
                )
            answer = (
                f"**Phác đồ thuốc điều trị & liều lượng chuẩn cho {v_name}:**\n"
                f"{chem_text or 'Hiện chưa ghi nhận phác đồ hóa học.'}\n"
                f"**Lưu ý an toàn:** Tuân thủ nguyên tắc 4 đúng (đúng thuốc, đúng liều, đúng lúc, đúng cách) và mang đồ bảo hộ khi phun."
            )
        elif any(w in q_clean for w in ['sau', 'ray', 'rep', 'bo', 'con trung', 'moi gioi']):
            pests = matched_info.get('pest_vectors_and_co_infections', [])
            answer = (
                f"**Sâu hại môi giới và bệnh đi kèm trên {crop} ({v_name}):**\n"
                + "\n".join(f"- {p}" for p in pests)
                + "\n\n**Khuyến nghị:** Cần phòng trừ triệt để sâu chích hút từ giai đoạn cây con để ngăn nấm và virus lây lan."
            )
        else:
            answer = (
                f"**Thông tin chuyên sâu về {v_name} trên {crop}:**\n\n"
                f"1. **Tác nhân:** {matched_info.get('scientific_name')} ({matched_info.get('pathogen_type')}).\n"
                f"2. **Đất & Môi trường:** Đất pH {soil.get('optimal_ph', '6.0-6.8')}, nhiệt độ thích hợp {env.get('temperature_range_c', 'ấm ẩm')}.\n"
                f"3. **Biện pháp canh tác triệt để:**\n"
                + "\n".join(f"   - {c}" for c in cultural[:2]) + "\n"
                f"4. **Phòng trừ sinh học:**\n"
                + "\n".join(f"   - {b}" for b in biological[:2]) + "\n"
                f"5. **Phác đồ hóa học chính:** "
                + (f"{chemical[0].get('active_ingredient')} ({', '.join(chemical[0].get('trade_names', []))}), liều: {chemical[0].get('exact_dosage')}" if chemical else "Ưu tiên biện pháp sinh học.")
            )

        return {
            'answer': answer,
            'matched_disease': matched_disease,
            'crop': crop,
            'references': matched_info.get('sources', [])
        }
