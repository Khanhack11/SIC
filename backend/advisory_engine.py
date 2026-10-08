"""
Agricultural Advisory Engine (Subsystem B)
Author: Nguyen Quoc Khanh (Backend Developer & Database Architect)
SIC Capstone Project - Group 5
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from backend.config import KNOWLEDGE_BASE_PATH, CONFIDENCE_THRESHOLD


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
        """Retrieves raw agronomic advisory record for a given disease ID."""
        return self.diseases.get(disease_id)

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

    def process_diagnosis(self, inference_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Confidence-Gated Decision Logic (Subsystem B Core):
        1. Evaluates confidence against the configured threshold.
        2. If confidence >= threshold: returns full agronomic advisory card.
        3. If confidence < threshold: returns warning prompt for clearer photos.
        """
        predicted_class = inference_result["predicted_class"]
        confidence = inference_result["confidence"]
        latency_ms = inference_result.get("latency_ms", 0.0)

        # Retrieve disease details from database
        disease_record = self.get_disease_by_id(predicted_class)

        # Decision Logic: Check confidence gate
        if confidence >= self.threshold:
            # Case 1: High confidence prediction
            status = "CONFIRMED"
            message = "Độ tin cậy đạt chuẩn. Đã trích xuất phác đồ tư vấn nông nghiệp chính xác."
            advisory_payload = disease_record
            remediation_notice = None
        else:
            # Case 2: Low confidence prediction (< threshold)
            status = "UNCERTAIN"
            message = (
                f"Độ tin cậy mô hình ({confidence * 100:.1f}%) thấp hơn ngưỡng an toàn ({self.threshold * 100:.0f}%). "
                "Hệ thống tạm giữ phác đồ hóa chất để tránh can thiệp sai lầm."
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

        return {
            "diagnosis_status": status,
            "message": message,
            "prediction": {
                "disease_id": predicted_class,
                "crop": inference_result.get("crop"),
                "is_healthy": inference_result.get("is_healthy", False),
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%",
                "threshold_applied": self.threshold,
                "confidence_gate_passed": confidence >= self.threshold,
                "latency_ms": latency_ms,
            },
            "advisory_details": advisory_payload,
            "remediation_notice": remediation_notice,
            "regulatory_disclaimer": self.disclaimer,
        }
