"""Append agronomy, weather, and chatbot JavaScript logic to frontend/frontend.js."""
from pathlib import Path

JS_PATH = Path('frontend/frontend.js')

def update():
    content = JS_PATH.read_text(encoding='utf-8')
    if 'switchSection' in content:
        print('Methods already exist in frontend.js')
        return

    extra_js = '''

/* === AGRONOMY, WEATHER, AND CHATBOT EXTENSION FUNCTIONS === */
function switchSection(sectionId) {
  const sections = ['sec-chat', 'sec-weather', 'sec-agronomy', 'sec-dataset'];
  const tabBtns = ['tab-btn-chat', 'tab-btn-weather', 'tab-btn-agronomy', 'tab-btn-dataset'];
  sections.forEach(id => {
    const el = byId(id);
    if (el) el.style.display = (id === sectionId) ? 'block' : 'none';
  });
  tabBtns.forEach(id => {
    const btn = byId(id);
    if (btn) {
      if (id === 'tab-btn-' + sectionId.replace('sec-', '')) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    }
  });
  if (sectionId === 'sec-agronomy') {
    const select = byId('agro-select');
    if (select) loadAgronomyDetail(select.value);
  }
}

async function askChat(text) {
  const input = byId('chat-input');
  if (input) {
    input.value = text;
    handleChatSubmit(new Event('submit'));
  }
}

async function handleChatSubmit(e) {
  if (e && e.preventDefault) e.preventDefault();
  const input = byId('chat-input');
  const box = byId('chat-box');
  if (!input || !box) return;
  const msg = input.value.trim();
  if (!msg) return;

  // Append user message
  const userDiv = document.createElement('div');
  userDiv.className = 'chat-msg user';
  userDiv.textContent = msg;
  box.appendChild(userDiv);
  input.value = '';
  box.scrollTop = box.scrollHeight;

  // Bot loading placeholder
  const botDiv = document.createElement('div');
  botDiv.className = 'chat-msg bot';
  botDiv.textContent = 'Đang tra cứu cơ sở tri thức nông nghiệp…';
  box.appendChild(botDiv);
  box.scrollTop = box.scrollHeight;

  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg }),
      signal: requestTimeout(8000)
    });
    if (!response.ok) throw new Error('Chat API error');
    const data = await response.json();
    botDiv.innerHTML = data.answer.replace(/\\n/g, '<br>').replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
  } catch (err) {
    botDiv.textContent = 'Xin lỗi bà con, hệ thống đang bận. Vui lòng thử lại sau ít giây!';
  }
  box.scrollTop = box.scrollHeight;
}

function setWeatherPreset(temp, humid, crop) {
  if (byId('w-temp')) byId('w-temp').value = temp;
  if (byId('w-humid')) byId('w-humid').value = humid;
  if (byId('w-crop')) byId('w-crop').value = crop;
  handleWeatherCheck(new Event('submit'));
}

async function handleWeatherCheck(e) {
  if (e && e.preventDefault) e.preventDefault();
  const resultsDiv = byId('weather-results');
  if (!resultsDiv) return;
  const temp = parseFloat(byId('w-temp')?.value || 25);
  const humid = parseFloat(byId('w-humid')?.value || 80);
  const crop = byId('w-crop')?.value || null;

  resultsDiv.innerHTML = '<p class="hint">Đang phân tích dữ liệu khí tượng và kích hoạt mô hình dịch tễ học...</p>';

  try {
    const res = await fetch(`${API_BASE}/api/weather-advisory`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ temperature_c: temp, humidity_pct: humid, crop: crop }),
      signal: requestTimeout(6000)
    });
    if (!res.ok) throw new Error('Weather API error');
    const data = await res.json();

    if (data.status === 'ALERT' && data.alerts.length > 0) {
      let html = `<div class="weather-alert-card">
        <h3 style="color:#c62828; margin-bottom:6px;">⚠️ Phát hiện ${data.total_risks_detected} Nguy cơ Dịch bệnh Bùng phát!</h3>
        <p style="font-size:13.5px; color:#555;">${data.general_advice}</p>`;
      
      data.alerts.forEach(alert => {
        html += `<div style="margin-top:10px; padding:10px; background:#fff; border-radius:6px; border:1px solid #ffcdd2;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong>${alert.vietnamese_name} (${alert.crop})</strong>
            <span class="pill-badge pill-red">${alert.risk_level}</span>
          </div>
          <p style="font-size:13px; color:#b71c1c; margin:4px 0;">${alert.warning_message}</p>
          <div style="font-size:12.5px; color:#333; margin-top:4px;">
            <strong>Khuyến nghị khẩn:</strong> ${alert.preventive_actions.join('; ')}
          </div>
        </div>`;
      });
      html += '</div>';
      resultsDiv.innerHTML = html;
    } else {
      resultsDiv.innerHTML = `<div class="weather-safe-card">
        <h3 style="color:#2e7d32; margin-bottom:4px;">✅ An toàn - Chưa ghi nhận Nguy cơ Bùng phát Dịch lớn</h3>
        <p style="font-size:13.5px; color:#33691e;">Với nhiệt độ ${temp}°C và độ ẩm ${humid}%, điều kiện môi trường hiện tại không nằm trong vùng nhiệt ẩm tối thích của các bào tử nấm sương mai, đốm vòng hay gỉ sắt.</p>
      </div>`;
    }
  } catch (err) {
    resultsDiv.innerHTML = '<p class="hint" style="color:#c62828;">Không thể kết nối máy chủ phân tích khí tượng.</p>';
  }
}

async function loadAgronomyDetail(diseaseId) {
  const display = byId('agro-detail-display');
  if (!display || !diseaseId) return;
  display.innerHTML = '<p class="hint">Đang tải chi tiết hồ sơ nông học...</p>';

  try {
    const res = await fetch(`${API_BASE}/api/diseases/${diseaseId}/agronomy`, { signal: requestTimeout(6000) });
    if (!res.ok) throw new Error('Agronomy API error');
    const data = await res.json();
    const agro = data.agronomy_data;
    const soil = agro.soil_requirements || {};
    const env = agro.environmental_conditions || {};
    const pests = agro.pest_vectors_and_co_infections || [];
    const stages = agro.disease_stages || {};
    const solutions = agro.comprehensive_curative_solutions || {};
    const cultural = solutions.cultural_practices || [];
    const biological = solutions.biological_organic || [];
    const chemical = solutions.chemical_prescriptions || [];

    let chemHtml = '';
    chemical.forEach((c, idx) => {
      chemHtml += `<div style="margin-top:8px; padding:10px; background:#ffffff; border-radius:8px; border:1px solid #dcdcdc;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
          <strong style="color:#1b5e20;">${idx+1}. ${c.active_ingredient}</strong>
          <span class="pill-badge pill-blue">PHI: ${c.pre_harvest_interval_days} ngày</span>
        </div>
        <p style="font-size:13px; color:#444; margin:4px 0;"><strong>Thương phẩm tại VN:</strong> ${(c.trade_names||[]).join(', ')}</p>
        <p style="font-size:13px; color:#2e7d32; margin:2px 0;"><strong>Liều lượng pha:</strong> ${c.exact_dosage}</p>
        <p style="font-size:12.5px; color:#666; margin:2px 0;"><strong>Thời điểm phun:</strong> ${c.application_timing}</p>
      </div>`;
    });

    display.innerHTML = `
      <div style="border-top:2px solid #e8f5e9; padding-top:14px;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:12px;">
          <div>
            <h3 style="color:#1b5e20; font-size:20px; font-weight:700;">${agro.vietnamese_name}</h3>
            <p style="font-size:14px; font-style:italic; color:#555;">Tác nhân: ${agro.scientific_name} (${agro.pathogen_type}) · Cây trồng: ${agro.crop}</p>
          </div>
          <span class="pill-badge pill-green">${agro.severity_level || 'Cần can thiệp'}</span>
        </div>

        <div class="agro-grid">
          <!-- Soil Card -->
          <div class="agro-subcard">
            <h4>🌱 Yêu cầu Đất trồng & Thoát nước</h4>
            <p style="font-size:13px; margin-bottom:4px;"><strong>Loại đất:</strong> ${soil.preferred_soil || 'Đất thịt nhẹ tơi xốp'}</p>
            <p style="font-size:13px; margin-bottom:4px;"><strong>Độ pH tối ưu:</strong> <span class="pill-badge pill-green">${soil.optimal_ph || '6.0 - 6.8'}</span></p>
            <p style="font-size:13px; margin-bottom:4px;"><strong>Độ ẩm đất:</strong> ${soil.soil_moisture_pct || '60% - 70%'}</p>
            <p style="font-size:12.5px; color:#555;"><strong>Thoát nước:</strong> ${soil.soil_drainage || 'Lên luống cao, không để ngập úng'}</p>
          </div>

          <!-- Environment Card -->
          <div class="agro-subcard">
            <h4>🌦️ Điều kiện Khí hậu & Sâu hại Môi giới</h4>
            <p style="font-size:13px; margin-bottom:4px;"><strong>Nhiệt độ thích hợp:</strong> ${env.temperature_range_c || 'Ấm ẩm'}</p>
            <p style="font-size:13px; margin-bottom:4px;"><strong>Độ ẩm không khí:</strong> ${env.humidity_pct || '> 85%'}</p>
            <p style="font-size:13px; margin-bottom:4px;"><strong>Mùa vụ phát sinh:</strong> ${env.seasonality || 'Vụ Đông Xuân'}</p>
            <p style="font-size:12.5px; color:#c62828;"><strong>Sâu môi giới:</strong> ${(pests.length>0 ? pests.join('; ') : 'Chưa ghi nhận')}</p>
          </div>
        </div>

        <!-- Solutions Section -->
        <div style="margin-top:16px;">
          <h4 style="color:#1b5e20; font-size:16px; margin-bottom:8px;">🛡️ Giải pháp Quản lý & Phòng trị Triệt để</h4>
          
          <div class="agro-grid">
            <div class="agro-subcard">
              <h5 style="color:#2e7d32; font-weight:700; margin-bottom:6px;">1. Biện pháp Canh tác Cơ học:</h5>
              <ul style="font-size:13px; padding-left:18px; color:#333;">
                ${cultural.map(c => `<li>${c}</li>`).join('')}
              </ul>
            </div>

            <div class="agro-subcard">
              <h5 style="color:#2e7d32; font-weight:700; margin-bottom:6px;">2. Phòng trừ Sinh học & Hữu cơ:</h5>
              <ul style="font-size:13px; padding-left:18px; color:#333;">
                ${biological.map(b => `<li>${b}</li>`).join('')}
              </ul>
            </div>
          </div>

          <div style="margin-top:14px; background:#f4f9f4; border:1px solid #c8e6c9; border-radius:10px; padding:14px;">
            <h5 style="color:#1b5e20; font-weight:700; margin-bottom:6px;">3. Phác đồ Hóa học & Liều lượng Pha chuẩn (Khoa học & Đúng bệnh):</h5>
            ${chemHtml || '<p style="font-size:13px; color:#666;">Cây khỏe hoặc chưa ghi nhận thuốc hóa học đặc hiệu.</p>'}
            <p style="font-size:12px; color:#666; margin-top:8px;">* Tuân thủ quy tắc 4 đúng (đúng thuốc, đúng liều, đúng lúc, đúng cách) và thời gian cách ly PHI trước khi thu hoạch.</p>
          </div>
        </div>
      </div>
    `;
  } catch (err) {
    display.innerHTML = '<p class="hint" style="color:#c62828;">Không thể tải dữ liệu nông học lúc này.</p>';
  }
}
'''

    content = content.rstrip() + '\n' + extra_js
    JS_PATH.write_text(content, encoding='utf-8')
    print('Successfully appended extension functions to frontend/frontend.js!')

if __name__ == '__main__':
    update()
