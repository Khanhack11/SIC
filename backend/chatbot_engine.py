"""
Agronomy AI Chatbot Engine (Pillar 2)
Integrates Google Gemini API (Generative AI) with Structured Local Agronomic Knowledge Base
to provide expert consultation on crop diseases, pest vectors, soil & environmental conditions,
and chemical & organic treatment protocols with precise dosages.
"""

from typing import Dict, Any, List, Optional
import os
import json
import logging
import urllib.request
import urllib.error

from backend.advisory_engine import AdvisoryKnowledgeBase
from backend.config import KNOWLEDGE_BASE_PATH, GEMINI_API_KEY, GEMINI_MODEL_CANDIDATES

logger = logging.getLogger(__name__)


class AgronomyChatbot:
    """
    Hybrid Agronomy Chatbot:
    Combines Google Gemini Generative AI with the local expert Knowledge Base.
    Ensures high fidelity, empathetic, and scientifically validated agricultural guidance.
    """

    def __init__(self, advisory_kb: Optional[AdvisoryKnowledgeBase] = None, api_key: str = GEMINI_API_KEY):
        if advisory_kb is None:
            self.kb = AdvisoryKnowledgeBase(KNOWLEDGE_BASE_PATH)
        else:
            self.kb = advisory_kb
        self.api_key = api_key or GEMINI_API_KEY

    def _call_gemini(self, query: str, kb_context: str = "") -> Optional[Dict[str, str]]:
        """
        Calls Google Gemini generateContent API with candidate fallbacks.
        """
        if not self.api_key:
            return None

        system_instruction = (
            "Bạn là Trợ lý Chuyên gia Nông học AI cao cấp của dự án Samsung Innovation Campus (SIC) - Group 5. "
            "Nhiệm vụ của bạn là tư vấn cho nông dân và kỹ sư nông nghiệp về cách chăm sóc cây trồng (Cà chua, Khoai tây, Ngô), "
            "chẩn đoán bệnh, yêu cầu đất trồng, bón phân, phòng trừ sinh học và phác đồ thuốc BVTV đặc trị kèm liều lượng chuẩn xác. "
            "Hãy trả lời ân cần, khoa học, định dạng Markdown rõ ràng, nhấn mạnh nguyên tắc 4 đúng và thời gian cách ly (PHI)."
        )

        user_content = query
        if kb_context:
            user_content = f"Dữ liệu chuyên môn nông học SIC tham khảo:\n{kb_context}\n\nCâu hỏi từ người dùng:\n{query}"

        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": user_content}]}
            ],
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 1024
            }
        }
        data_bytes = json.dumps(payload).encode("utf-8")

        for model_name in GEMINI_MODEL_CANDIDATES:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"}
            )
            try:
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    if resp.status == 200:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        candidates = res_json.get("candidates", [])
                        if candidates:
                            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                            if text:
                                return {
                                    "model": model_name,
                                    "text": text.strip()
                                }
            except Exception as e:
                logger.warning("Gemini model %s failed: %s, falling back to next model candidate...", model_name, e)
                continue

        return None

    def chat(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processes query via Google Gemini AI with local knowledge base grounding.
        """
        raw_query = query.strip()
        if not raw_query:
            return {
                "answer": "Xin chào! Tôi là Trợ lý Nông học AI của dự án SIC kết nối mô hình Google Gemini. Bạn có thể hỏi tôi về cách nhận diện bệnh, thuốc đặc trị, liều lượng phun, hoặc cách chăm sóc cây cà chua, khoai tây và ngô.",
                "reply": "Xin chào! Tôi là Trợ lý Nông học AI của dự án SIC kết nối mô hình Google Gemini. Bạn có thể hỏi tôi về cách nhận diện bệnh, thuốc đặc trị, liều lượng phun, hoặc cách chăm sóc cây cà chua, khoai tây và ngô.",
                "disease_id": None,
                "matched_disease": None,
                "engine": "local_kb",
                "suggestions": [
                    "Bệnh mốc sương trên cà chua dùng thuốc gì và liều lượng bao nhiêu?",
                    "Thuốc Ridomil Gold phun liều lượng thế nào?",
                    "Cây ngô bị bệnh gỉ sắt",
                    "Thời tiết ẩm ướt phòng bệnh gì cho cà chua?"
                ]
            }

        # 1. Fetch matching structured domain knowledge from local KB
        kb_result = self.kb.agricultural_chat(raw_query)
        kb_answer = kb_result.get("answer", "")
        disease_id = kb_result.get("matched_disease")
        crop = kb_result.get("crop")
        references = kb_result.get("references", [])

        # 2. Call Google Gemini with KB grounded context if enabled and not in test mode
        use_gemini = bool(self.api_key) and not bool(os.environ.get("SIC_TEST_MODE"))
        gemini_res = self._call_gemini(raw_query, kb_answer) if use_gemini else None

        if gemini_res:
            final_text = gemini_res["text"]
            model_used = f"Google Gemini AI ({gemini_res['model']})"
        else:
            final_text = kb_answer
            model_used = "SIC Agronomy Knowledge Base (Local)"

        # Contextual suggestions
        suggestions = [
            "Liều lượng phun Ridomil Gold cho cà chua",
            "Thuốc sinh học Trichoderma trị nấm lá",
            "Thời tiết mưa ẩm cần phòng bệnh gì?",
            "Thời gian cách ly an toàn trước thu hoạch"
        ]

        return {
            "answer": final_text,
            "reply": final_text,
            "disease_id": disease_id,
            "matched_disease": disease_id,
            "crop": crop,
            "references": references,
            "engine": model_used,
            "suggestions": suggestions
        }
