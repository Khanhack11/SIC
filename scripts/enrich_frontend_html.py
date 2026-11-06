"""Enrich frontend/frontend_demo.html with Agronomy Explorer, Weather Risk Advisory, Chatbot, and Kaggle Benchmark sections."""
from pathlib import Path

HTML_PATH = Path('frontend/frontend_demo.html')

def update():
    content = HTML_PATH.read_text(encoding='utf-8')
    
    extra_styles = '''
    /* Extra agronomy, weather, and chatbot styling */
    .section-tabs { display: flex; flex-wrap: wrap; gap: 8px; margin: 30px 0 16px 0; }
    .section-tab-btn { padding: 10px 18px; border-radius: 20px; border: 1px solid var(--border); background: var(--card-bg); color: var(--text-main); font-weight: 600; cursor: pointer; transition: all 0.2s; font-size: 15px; }
    .section-tab-btn.active { background: var(--primary); color: white; border-color: var(--primary); }
    .agro-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px; }
    .agro-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 14px; }
    .agro-subcard { background: #f9fbf9; border: 1px solid #dceede; border-radius: 10px; padding: 14px; }
    .agro-subcard h4 { color: #1b5e20; font-size: 15px; margin-bottom: 8px; display: flex; align-items: center; gap: 6px; }
    .weather-form { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)) auto; gap: 12px; align-items: flex-end; margin: 16px 0; }
    .weather-input { display: flex; flex-col; gap: 4px; }
    .weather-input label { font-size: 13px; font-weight: 600; color: var(--text-muted); }
    .weather-input input, .weather-input select { min-height: 40px; padding: 6px 12px; border: 1px solid var(--border); border-radius: 8px; font-size: 15px; }
    .btn-weather { background: #2e7d32; color: white; border: none; border-radius: 8px; padding: 0 20px; min-height: 40px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
    .btn-weather:hover { background: #1b5e20; }
    .weather-alert-card { border-left: 4px solid #d32f2f; background: #fff5f5; padding: 14px; border-radius: 0 8px 8px 0; margin-top: 10px; }
    .weather-safe-card { border-left: 4px solid #2e7d32; background: #f1f8e9; padding: 14px; border-radius: 0 8px 8px 0; margin-top: 10px; }
    .chat-container { border: 1px solid var(--border); border-radius: 12px; background: #ffffff; display: flex; flex-direction: column; height: 420px; overflow: hidden; }
    .chat-messages { flex: 1; padding: 16px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; background: #fdfdfd; }
    .chat-msg { max-width: 82%; padding: 10px 14px; border-radius: 14px; font-size: 14.5px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
    .chat-msg.bot { background: #e8f5e9; color: #1b5e20; align-self: flex-start; border-bottom-left-radius: 2px; }
    .chat-msg.user { background: #2e7d32; color: white; align-self: flex-end; border-bottom-right-radius: 2px; }
    .chat-input-row { display: flex; gap: 8px; padding: 10px 14px; border-top: 1px solid var(--border); background: #ffffff; }
    .chat-input-row input { flex: 1; min-height: 42px; border: 1px solid var(--border); border-radius: 20px; padding: 0 16px; font-size: 14.5px; }
    .chat-input-row button { background: var(--primary); color: white; border: none; border-radius: 20px; padding: 0 20px; font-weight: 600; cursor: pointer; }
    .chat-quick-tags { display: flex; flex-wrap: wrap; gap: 6px; padding: 8px 14px; background: #f8faf8; border-top: 1px solid #e8f0e8; }
    .chat-tag-btn { background: #ffffff; border: 1px solid #c8e6c9; border-radius: 14px; padding: 4px 10px; font-size: 12.5px; color: #1b5e20; cursor: pointer; transition: all 0.15s; }
    .chat-tag-btn:hover { background: #e8f5e9; }
    .pill-badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600; }
    .pill-red { background: #ffebee; color: #c62828; }
    .pill-green { background: #e8f5e9; color: #2e7d32; }
    .pill-blue { background: #e3f2fd; color: #1565c0; }
    '''

    # Insert extra styles before </style>
    if '.agro-card' not in content:
        content = content.replace('</style>', extra_styles + '\n  </style>')

    # Extra HTML sections to insert right before `</main>`
    extra_sections = '''
    <!-- SUB-MODULES: AGRONOMY, WEATHER, CHATBOT, AND DATASET INSIGHTS -->
    <div class="section-tabs" role="tablist">
      <button class="section-tab-btn active" onclick="switchSection('sec-chat')" id="tab-btn-chat">Trợ lý AI Nông nghiệp (Chatbot)</button>
      <button class="section-tab-btn" onclick="switchSection('sec-weather')" id="tab-btn-weather">Cảnh báo Dịch bệnh Thời tiết</button>
      <button class="section-tab-btn" onclick="switchSection('sec-agronomy')" id="tab-btn-agronomy">Hồ sơ Đất trồng & Liều lượng Thuốc</button>
      <button class="section-tab-btn" onclick="switchSection('sec-dataset')" id="tab-btn-dataset">Khảo sát Dữ liệu Kaggle vs Thực tế</button>
    </div>

    <!-- 1. CHATBOT SECTION -->
    <section id="sec-chat" class="agro-card" aria-labelledby="chat-heading">
      <h2 id="chat-heading">Trợ lý AI Nông nghiệp & Tư vấn Chăm sóc Cây trồng</h2>
      <p class="hint">Hỏi đáp trực tiếp về liều lượng thuốc, cách xử lý đất, sâu bệnh môi giới và các biện pháp điều trị dứt điểm cho Cà chua, Khoai tây, Ngô.</p>
      
      <div class="chat-container">
        <div id="chat-box" class="chat-messages">
          <div class="chat-msg bot">Xin chào bà con! Tôi là Trợ lý Nông nghiệp AI SIC Group 5. Bà con có thể hỏi tôi về:
• Liều lượng và cách phun thuốc (Ridomil Gold, Amistar Top, Tilt Super, Score, Daconil...)
• Yêu cầu đất trồng, độ pH và cách bón phân cân đối
• Sâu hại môi giới truyền bệnh và cách phòng trừ sinh học (Trichoderma, dầu neem)
• Cảnh báo dịch bệnh theo thời tiết thực tế</div>
        </div>
        <div class="chat-quick-tags">
          <span style="font-size:12px; color:#555; align-self:center;">Gợi ý:</span>
          <button type="button" class="chat-tag-btn" onclick="askChat('Liều lượng thuốc trị sương mai cà chua là bao nhiêu?')">Liều thuốc sương mai cà chua?</button>
          <button type="button" class="chat-tag-btn" onclick="askChat('Đất trồng khoai tây cần độ pH và độ ẩm bao nhiêu?')">Đất trồng & pH khoai tây?</button>
          <button type="button" class="chat-tag-btn" onclick="askChat('Cách trị dứt điểm bệnh gỉ sắt trên ngô?')">Trị dứt điểm gỉ sắt ngô?</button>
          <button type="button" class="chat-tag-btn" onclick="askChat('Sâu hại nào truyền bệnh đốm vòng cà chua?')">Sâu hại truyền bệnh?</button>
        </div>
        <form id="chat-form" class="chat-input-row" onsubmit="handleChatSubmit(event)">
          <input type="text" id="chat-input" placeholder="Nhập câu hỏi về cây trồng, bệnh hại hoặc phân thuốc..." autocomplete="off">
          <button type="submit" id="btn-chat-send">Gửi</button>
        </form>
      </div>
    </section>

    <!-- 2. WEATHER ADVISORY SECTION -->
    <section id="sec-weather" class="agro-card" style="display:none;" aria-labelledby="weather-heading">
      <h2 id="weather-heading">Dự báo & Cảnh báo Nguy cơ Bùng phát Dịch bệnh theo Thời tiết</h2>
      <p class="hint">Dựa trên mô hình dịch tễ học nông nghiệp: Nhiệt độ và độ ẩm không khí là 2 yếu tố quyết định tốc độ nảy mầm của bào tử nấm (Phytophthora, Alternaria, Puccinia...).</p>
      
      <form id="weather-form" class="weather-form" onsubmit="handleWeatherCheck(event)">
        <div class="weather-input">
          <label for="w-temp">Nhiệt độ (°C):</label>
          <input type="number" id="w-temp" step="0.5" value="18.0" min="0" max="50" required>
        </div>
        <div class="weather-input">
          <label for="w-humid">Độ ẩm không khí (%):</label>
          <input type="number" id="w-humid" step="1" value="92" min="10" max="100" required>
        </div>
        <div class="weather-input">
          <label for="w-crop">Cây trồng quan tâm:</label>
          <select id="w-crop">
            <option value="">Tất cả cây trồng</option>
            <option value="tomato">Cà chua (Tomato)</option>
            <option value="potato">Khoai tây (Potato)</option>
            <option value="corn">Ngô / Bắp (Corn)</option>
          </select>
        </div>
        <button type="submit" class="btn-weather">Kiểm tra Nguy cơ</button>
      </form>

      <div style="margin-top: 8px; display:flex; gap:6px; flex-wrap:wrap;">
        <span style="font-size:12px; color:#555; align-self:center;">Kịch bản mẫu:</span>
        <button type="button" class="chat-tag-btn" onclick="setWeatherPreset(18, 94, 'tomato')">Rét ẩm sương mù (18°C, 94%)</button>
        <button type="button" class="chat-tag-btn" onclick="setWeatherPreset(27, 88, 'tomato')">Ấm ẩm mưa dầm (27°C, 88%)</button>
        <button type="button" class="chat-tag-btn" onclick="setWeatherPreset(20, 92, 'corn')">Đông xuân mát mẻ (20°C, 92%)</button>
        <button type="button" class="chat-tag-btn" onclick="setWeatherPreset(32, 45, '')">Nắng nóng khô hanh (32°C, 45%)</button>
      </div>

      <div id="weather-results" style="margin-top: 14px;"></div>
    </section>

    <!-- 3. AGRONOMY & DOSAGE EXPLORER SECTION -->
    <section id="sec-agronomy" class="agro-card" style="display:none;" aria-labelledby="agro-heading">
      <h2 id="agro-heading">Tra cứu Hồ sơ Đất trồng, Môi trường & Liều lượng Thuốc Chuẩn</h2>
      <p class="hint">Thông tin đối chiếu từ Viện Bảo vệ Thực vật Việt Nam và Đại học Minnesota. Cung cấp giải pháp triệt để 3 lớp: Canh tác cơ học, Sinh học hữu cơ và Hoạt chất hóa học kèm liều lượng.</p>
      
      <div style="display:flex; gap:12px; align-items:center; margin-bottom:14px; flex-wrap:wrap;">
        <label for="agro-select" style="font-weight:600; font-size:14px;">Chọn đối tượng bệnh cây:</label>
        <select id="agro-select" onchange="loadAgronomyDetail(this.value)" style="padding:8px 12px; border-radius:8px; border:1px solid var(--border); font-size:14px;">
          <option value="Tomato_Early_blight">Cà chua - Đốm vòng / Cháy lá sớm (Alternaria solani)</option>
          <option value="Tomato_Late_blight">Cà chua - Mốc sương / Sương mai (Phytophthora infestans)</option>
          <option value="Tomato_Septoria_leaf_spot">Cà chua - Đốm lá Septoria (Septoria lycopersici)</option>
          <option value="Tomato_healthy">Cà chua - Cây khỏe mạnh (Đất trồng & dinh dưỡng)</option>
          <option value="Potato_Early_blight">Khoai tây - Đốm vòng sớm (Alternaria solani)</option>
          <option value="Potato_Late_blight">Khoai tây - Sương mai tàn lụi (Phytophthora infestans)</option>
          <option value="Potato_healthy">Khoai tây - Cây khỏe mạnh (Đất cát pha & pH)</option>
          <option value="Corn_Common_rust">Ngô - Bệnh gỉ sắt (Puccinia sorghi)</option>
          <option value="Corn_Northern_Leaf_Blight">Ngô - Bệnh cháy lá hình thoi lớn (Exserohilum turcicum)</option>
          <option value="Corn_Gray_leaf_spot">Ngô - Bệnh đốm xám chữ nhật (Cercospora zeae-maydis)</option>
          <option value="Corn_healthy">Ngô - Cây khỏe mạnh (Kỹ thuật bón phân N-P-K)</option>
        </select>
      </div>

      <div id="agro-detail-display">
        <p class="hint">Đang tải hồ sơ nông học...</p>
      </div>
    </section>

    <!-- 4. KAGGLE DATASET INSIGHTS SECTION -->
    <section id="sec-dataset" class="agro-card" style="display:none;" aria-labelledby="data-heading">
      <h2 id="data-heading">Khảo sát Bộ dữ liệu: Kaggle New Plant Diseases Dataset vs. Dữ liệu Thực địa</h2>
      
      <div class="agro-grid">
        <div class="agro-subcard">
          <h4>Kaggle New Plant Diseases Dataset (87K Ảnh)</h4>
          <p style="font-size:13.5px; color:#444;">
            • <strong>Quy mô:</strong> 87.000+ ảnh, 38 lớp cây trồng (Táo, Cà chua, Ngô, Khoai tây, Nho, Cam, Đào, Việt quất...).<br>
            • <strong>Đặc điểm:</strong> Ảnh chụp lá cắt rời trong phòng thí nghiệm trên nền xám/trắng đồng nhất.<br>
            • <strong>Thách thức thực tế:</strong> Mô hình chỉ train trên tập này đạt độ chính xác lab >99%, nhưng khi nông dân chụp ảnh ngoài đồng (có nền đất, cỏ rác, nhiều lá lộn xộn) dễ bị sụt giảm độ chính xác và phán đoán sai lệch.
          </p>
        </div>

        <div class="agro-subcard" style="border-color:#bbf7d0; background:#f0fdf4;">
          <h4 style="color:#166534;">Dữ liệu Đa nguồn Thực địa field_mobile_v2 (SIC Group 5)</h4>
          <p style="font-size:13.5px; color:#14532d;">
            • <strong>Quy mô:</strong> 20.902 records tuyển chọn kỹ lưỡng, khóa mã băm SHA-256.<br>
            • <strong>Đột phá kỹ thuật:</strong> Trích xuất <strong>3.160 hộp bao crop lá đơn</strong> từ PlantDoc và bổ sung 5 tập dữ liệu thực địa quốc tế (Bangladesh, Ethiopia, Indonesia, Pakistan, IIT KGP).<br>
            • <strong>Hiệu quả:</strong> Tăng trưởng F1 thực địa ngoạn mục <strong>+39.03 điểm phần trăm</strong> trên PlantDoc (từ 15.95% lên 54.98%), đồng thời giữ nguyên 98.17% trên PlantVillage!
          </p>
        </div>
      </div>

      <div style="margin-top:14px; padding:12px; background:#f5f5f5; border-radius:8px; font-size:13px; color:#555;">
        <strong>Ý nghĩa cốt lõi:</strong> Tính năng khoanh vùng lá (Bounding-Box Crop Canvas) tích hợp trên giao diện này cho phép người nông dân chủ động loại bỏ phông nền rác ngoài đồng, đưa ảnh thực tế về tiêu chuẩn tối ưu giúp mô hình AI phân loại với độ tin cậy vượt trội (tăng từ 51.2% lên 98.3%).
      </div>
    </section>
'''

    if 'sec-chat' not in content:
        content = content.replace('</main>', extra_sections + '\n  </main>')
        HTML_PATH.write_text(content, encoding='utf-8')
        print('Successfully updated frontend/frontend_demo.html with new modules!')
    else:
        print('Sections already present in HTML.')

if __name__ == '__main__':
    update()
