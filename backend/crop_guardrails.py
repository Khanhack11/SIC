"""
Crop Scope Guardrails for SIC Agronomy AI System
Enforces strict operational boundaries: The chatbot is restricted exclusively
to the 3 trained target crops:
1. Tomato (Cà chua - Solanum lycopersicum)
2. Potato (Khoai tây - Solanum tuberosum)
3. Corn / Maize (Ngô / Bắp - Zea mays)

Complies with Agricultural AI Safety & Bio-security Standards:
Rejects out-of-scope crops to prevent phytotoxicity, off-label pesticide usage,
or misleading agronomic prescriptions.
"""

from typing import Dict, Any, List, Optional, Tuple
import unicodedata
import re

# Metadata of the 3 officially trained and supported crops
SUPPORTED_CROPS: Dict[str, Dict[str, Any]] = {
    "tomato": {
        "id": "tomato",
        "name_vi": "Cà chua",
        "name_en": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "family": "Solanaceae (Họ Cà)",
        "supported_diseases": [
            "Bệnh mốc sương / sương mai (Phytophthora infestans)",
            "Bệnh đốm vòng / cháy lá sớm (Alternaria solani)",
            "Bệnh đốm lá Septoria (Septoria lycopersici)",
            "Cây khỏe mạnh (Dinh dưỡng & Đất trồng)"
        ],
        "keywords": [
            "ca chua", "tomato", "lycopersicon", "solanum lycopersicum",
            "qua ca chua", "cay ca chua", "la ca chua"
        ]
    },
    "potato": {
        "id": "potato",
        "name_vi": "Khoai tây",
        "name_en": "Potato",
        "scientific_name": "Solanum tuberosum",
        "family": "Solanaceae (Họ Cà)",
        "supported_diseases": [
            "Bệnh mốc sương / sương mai tàn lụi (Phytophthora infestans)",
            "Bệnh đốm vòng / cháy lá sớm (Alternaria solani)",
            "Cây khỏe mạnh (Đất cát pha & pH)"
        ],
        "keywords": [
            "khoai tay", "potato", "solanum tuberosum",
            "cu khoai tay", "cay khoai tay", "la khoai tay"
        ]
    },
    "corn": {
        "id": "corn",
        "name_vi": "Ngô (Bắp)",
        "name_en": "Corn / Maize",
        "scientific_name": "Zea mays",
        "family": "Poaceae (Họ Hòa thảo)",
        "supported_diseases": [
            "Bệnh gỉ sắt hại ngô (Puccinia sorghi)",
            "Bệnh cháy lá hình thoi lớn (Exserohilum turcicum)",
            "Bệnh đốm xám chữ nhật (Cercospora zeae-maydis)",
            "Cây khỏe mạnh (Kỹ thuật bón phân N-P-K)"
        ],
        "keywords": [
            "ngo", "bap", "corn", "maize", "zea mays",
            "bap ngo", "cay ngo", "cay bap", "la ngo", "la bap"
        ]
    }
}

# Known diseases and pathogens specific to the 3 crops (in-scope triggers)
SUPPORTED_DISEASE_KEYWORDS = [
    "suong mai", "moc suong", "phytophthora", "late blight",
    "dom vong", "chay la som", "alternaria", "early blight",
    "septoria", "dom la septoria",
    "gi sat", "ri sat", "puccinia", "common rust",
    "chay la hinh thoi", "dom la lon", "exserohilum", "northern leaf blight",
    "dom xam", "cercospora", "gray leaf spot"
]

# Patterns for out-of-scope agricultural crops
OUT_OF_SCOPE_CROP_PATTERNS: Dict[str, List[str]] = {
    "Lúa nước (Gạo)": ["lua", "gao", "rice", "paddy", "dao on", "kho van", "ray nau"],
    "Sầu riêng": ["sau rieng", "durian", "ri6", "mon thong", "xì mủ", "thoi re sau rieng"],
    "Cà phê": ["ca phe", "coffee", "ri sat ca phe", "kho canh ca phe"],
    "Hồ tiêu (Tiêu)": ["tieu", "ho tieu", "pepper", "chet nhanh", "chet cham"],
    "Cây có múi (Cam, Bưởi, Chanh, Quýt)": [
        "cam", "buoi", "chanh", "quyt", "citrus", "orange", "pomelo",
        "grapefruit", "lemon", "lime", "mandarin", "vang la greeing", "loet cam"
    ],
    "Thanh long": ["thanh long", "dragon fruit", "dom trang thanh long", "nam bo hong thanh long"],
    "Xoài": ["xoai", "mango", "than thu xoai"],
    "Nhãn / Vải": ["nhan", "vai", "longan", "lychee", "choi rong nhan"],
    "Chuối": ["chuoi", "banana", "panama chuoi", "sigatoka"],
    "Đu đủ": ["du du", "papaya"],
    "Ổi": ["oi", "guava"],
    "Nho": ["nho", "grape"],
    "Táo": ["tao", "apple"],
    "Dưa hấu": ["dua hau", "watermelon"],
    "Dưa leo / Dưa chuột": ["dua leo", "dua chuot", "cucumber"],
    "Dưa lưới": ["dua luoi", "cantaloupe", "melon"],
    "Bí đao / Bí đỏ / Bầu / Mướp": ["bi dao", "bi do", "pumpkin", "squash", "bau", "muop", "gourd"],
    "Ớt": ["ot", "chili", "chilli", "than thu ot"],
    "Cà tím": ["ca tim", "eggplant"],
    "Chè (Trà)": ["che", "tra", "tea", "ray xanh che"],
    "Cao su": ["cao su", "rubber"],
    "Điều": ["dieu", "hat dieu", "cashew"],
    "Mía": ["mia", "sugar cane"],
    "Khoai lang": ["khoai lang", "sweet potato"],
    "Sắn (Khoai mì)": ["san", "khoai mi", "cassava", "kham la san"],
    "Rau họ cải (Bắp cải, Su hào, Súp lơ, Cải xanh)": [
        "bap cai", "su hao", "sup lo", "cai ngot", "cai xanh",
        "cabbage", "cauliflower", "broccoli", "sau to bap cai"
    ],
    "Hoa kiểng (Hoa hồng, Hoa lan, Hoa cúc)": ["hoa hong", "hoa lan", "hoa cuc", "rose", "orchid"]
}

# Non-agricultural domains for prompt classification
NON_AGRI_KEYWORDS = [
    "lap trinh", "python", "javascript", "code", "html", "css",
    "phuong trinh", "toan hoc", "vat ly", "hoa hoc",
    "chung khoan", "tien ao", "bitcoin", "crypto",
    "thoi trang", "dien vien", "ca si", "bong da", "the thao",
    "chinh tri", "lich su", "dia ly"
]


def normalize_text(text: str) -> str:
    """Normalize and strip Vietnamese accents for deterministic keyword matching."""
    s = unicodedata.normalize("NFD", text)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("đ", "d").replace("Đ", "d")
    s = s.lower().strip()
    return re.sub(r"\s+", " ", s)


def detect_crop_scope(query: str) -> Dict[str, Any]:
    """
    Evaluates whether the user's query falls within the 3 trained crops.

    Returns:
        dict:
            is_in_scope (bool): True if allowed, False if out-of-scope.
            detected_crop (str or None): Name of detected out-of-scope crop (if any).
            matched_supported_crop (str or None): "tomato", "potato", "corn" if matched.
            reason (str): Explanation code ('supported_crop', 'out_of_scope_crop', 'non_agricultural', 'general_greeting').
            refusal_message (str or None): Standard response if out-of-scope.
    """
    clean_q = normalize_text(query)

    # 1. Check for supported crops in query
    matched_supported = None
    for crop_id, meta in SUPPORTED_CROPS.items():
        for kw in meta["keywords"]:
            if normalize_text(kw) in clean_q:
                matched_supported = crop_id
                break
        if matched_supported:
            break

    # 2. Check for supported disease keywords
    has_supported_disease = any(normalize_text(dkw) in clean_q for dkw in SUPPORTED_DISEASE_KEYWORDS)

    # 3. Check for explicit out-of-scope crop patterns
    detected_out_of_scope = None
    for crop_name, patterns in OUT_OF_SCOPE_CROP_PATTERNS.items():
        for p in patterns:
            pattern_clean = normalize_text(p)
            # Match whole word / phrase boundaries
            regex = rf"\b{re.escape(pattern_clean)}\b"
            if re.search(regex, clean_q):
                detected_out_of_scope = crop_name
                break
        if detected_out_of_scope:
            break

    # Decision Matrix:
    # A) Out-of-scope crop explicitly mentioned without supported crop context
    if detected_out_of_scope and not matched_supported:
        return {
            "is_in_scope": False,
            "detected_crop": detected_out_of_scope,
            "matched_supported_crop": None,
            "reason": "out_of_scope_crop",
            "refusal_message": format_scope_refusal(detected_out_of_scope)
        }

    # B) Non-agricultural queries
    is_non_agri = any(re.search(rf"\b{re.escape(normalize_text(w))}\b", clean_q) for w in NON_AGRI_KEYWORDS)
    if is_non_agri and not matched_supported and not has_supported_disease:
        return {
            "is_in_scope": False,
            "detected_crop": None,
            "matched_supported_crop": None,
            "reason": "non_agricultural",
            "refusal_message": format_non_agri_refusal()
        }

    # C) In-scope: either supported crop, supported disease, or general agronomic greeting
    return {
        "is_in_scope": True,
        "detected_crop": None,
        "matched_supported_crop": matched_supported,
        "reason": "supported_crop" if (matched_supported or has_supported_disease) else "general_agri",
        "refusal_message": None
    }


def format_scope_refusal(crop_name: str) -> str:
    """Standardized polite refusal message for out-of-scope crops."""
    return (
        f"⚠️ **TỪ CHỐI TƯ VẤN: CÂY TRỒNG NGOÀI DANH MỤC THẨM ĐỊNH (SIC GROUP 5)**\n\n"
        f"Kính chào bà con! Hệ thống AI Nông nghiệp SIC Group 5 hiện tại **CHỈ ĐƯỢC HUẤN LUYỆN MÔ HÌNH VÀ THẨM ĐỊNH AN TOÀN SINH HỌC CHO 3 LOẠI CÂY TRỒNG MỤC TIÊU**:\n\n"
        f"1. 🍅 **Cà chua** (*Tomato* - *Solanum lycopersicum*)\n"
        f"2. 🥔 **Khoai tây** (*Potato* - *Solanum tuberosum*)\n"
        f"3. 🌽 **Ngô / Bắp** (*Corn / Maize* - *Zea mays*)\n\n"
        f"❌ **Cây trồng bà con vừa hỏi:** **{crop_name}** *(Chưa nằm trong danh mục mô hình AI được huấn luyện)*.\n\n"
        f"🔒 **Lý do hệ thống từ chối trả lời:**\n"
        f"• **An toàn sinh học & Ngăn ngừa ngộ độc:** Mỗi loại cây trồng có ngưỡng mẫn cảm thuốc bảo vệ thực vật và đặc điểm sinh lý hoàn toàn khác nhau. Việc tự ý áp dụng đơn thuốc ngoài danh mục có thể gây cháy lá, rụng hoa trái hoặc tồn dư hóa chất vượt ngưỡng an toàn (MRL).\n"
        f"• **Tránh chẩn đoán sai lệch:** Mô hình thị giác máy tính và cơ sở dữ liệu chuyên gia của đề tài chưa được kiểm chứng lâm sàng đối với cây {crop_name}.\n\n"
        f"💡 **Khuyến nghị dành cho bà con:**\n"
        f"• Đối với cây **{crop_name}**, xin bà con vui lòng liên hệ trực tiếp **Cán bộ Khuyến nông** hoặc **Trạm Trồng trọt & Bảo vệ Thực vật địa phương** để nhận phác đồ hướng dẫn chính xác.\n"
        f"• Tại hệ thống này, bà con có thể hỏi về các loại sâu bệnh, kỹ thuật đất trồng (độ pH, ẩm độ) và liều lượng thuốc đặc trị cho **Cà chua, Khoai tây và Ngô**."
    )


def format_non_agri_refusal() -> str:
    """Standardized response for non-agricultural queries."""
    return (
        "⚠️ **NGOÀI PHẠM VI CHUYÊN MÔN NÔNG NGHIỆP (SIC GROUP 5)**\n\n"
        "Tôi là **Trợ lý Chuyên gia Nông học AI của dự án Samsung Innovation Campus (Group 5)**. "
        "Tôi được thiết lập chuyên trách và giới hạn trong phạm vi nông nghiệp kỹ thuật số cho **3 loại cây trồng chính**:\n\n"
        "• 🍅 **Cà chua**: Bệnh mốc sương, đốm vòng, đốm lá Septoria, liều thuốc BVTV, đất trồng.\n"
        "• 🥔 **Khoai tây**: Bệnh mốc sương tàn lụi, đốm vòng sớm, kỹ thuật đất cát pha, pH đất.\n"
        "• 🌽 **Ngô (Bắp)**: Bệnh gỉ sắt, cháy lá hình thoi, đốm xám, sâu keo mùa thu, phân bón N-P-K.\n\n"
        "Vui lòng đặt câu hỏi liên quan đến kỹ thuật canh tác, sâu bệnh hoặc phân thuốc cho 3 loại cây trồng trên để được hỗ trợ tốt nhất!"
    )


# Strict System Instruction for Google Gemini
GEMINI_SCOPE_GUARDRAIL_INSTRUCTION = (
    "Bạn là Trợ lý Chuyên gia Nông học AI của dự án Samsung Innovation Campus (SIC) - Group 5.\n"
    "========================================================================================\n"
    "BỘ QUY TẮC PHẠM VI CỐT LÕI (STRICT SCOPE GUARDRAILS - BẮT BUỘC TUÂN THỦ 100%):\n"
    "1. PHẠM VI HỖ TRỢ ĐỘC QUYỀN:\n"
    "   Bạn CHỈ ĐƯỢC PHÉP tư vấn, chẩn đoán bệnh, phân tích đất trồng và kê đơn thuốc cho ĐÚNG BA (03) LOẠI CÂY TRỒNG ĐÃ ĐƯỢC HUẤN LUYỆN:\n"
    "   - 🍅 Cà chua (Tomato - Solanum lycopersicum)\n"
    "   - 🥔 Khoai tây (Potato - Solanum tuberosum)\n"
    "   - 🌽 Ngô / Bắp (Corn / Maize - Zea mays)\n\n"
    "2. QUY TẮC TỪ CHỐI CÂY TRỒNG NGOÀI PHẠM VI:\n"
    "   Nếu người dùng hỏi về BẤT KỲ cây trồng nào khác (ví dụ: lúa, sầu riêng, cà phê, cam, bưởi, chanh, xoài, thanh long, chuối, dưa hấu, ớt, rau cải, cây ăn trái khác...):\n"
    "   - BẠN BẮT BUỘC PHẢI TỪ CHỐI LỊCH SỰ VÀ DỨT KHOÁT.\n"
    "   - TUYỆT ĐỐI KHÔNG kê đơn hoạt chất hóa học, KHÔNG đưa liều lượng thuốc BVTV cho các cây ngoài 3 cây này.\n"
    "   - GIẢI THÍCH LÝ DO: Hệ thống AI SIC Group 5 chỉ được huấn luyện mô hình và thẩm định an toàn sinh học cho 3 cây trồng (Cà chua, Khoai tây, Ngô). Kê đơn cho cây trồng ngoài danh mục có thể gây ngộ độc cháy lá hoặc dư lượng độc hại.\n"
    "   - KHUYÊN NÔNG DÂN liên hệ cán bộ Khuyến nông hoặc Trạm Trồng trọt & BVTV địa phương.\n\n"
    "3. QUY TẮC NGOÀI NÔNG NGHIỆP:\n"
    "   Nếu câu hỏi không thuộc lĩnh vực nông nghiệp (lập trình, giải toán, tài chính, giải trí...): Từ chối lịch sự và hướng dẫn quay lại chủ đề Cà chua, Khoai tây, Ngô.\n\n"
    "4. QUY TẮC TƯ VẤN CÀ CHUA, KHOAI TÂY, NGÔ:\n"
    "   Luôn trả lời ân cần, khoa học, định dạng Markdown rõ ràng. Nêu rõ phác đồ 3 lớp (Canh tác - Sinh học - Hóa học), liều lượng chuẩn xác (ml/g trên bình 16L), nguyên tắc 4 đúng và thời gian cách ly (PHI) bắt buộc.\n"
    "========================================================================================"
)
