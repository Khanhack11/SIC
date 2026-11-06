/* Shared-origin demo client; no model predictions or clinical claims are hard-coded. */
const API_BASE = (location.protocol === 'file:' || location.port === '8080')
  ? 'http://127.0.0.1:8000' : location.origin;
const byId = (id) => document.getElementById(id);
function requestTimeout(milliseconds) {
  if (typeof AbortSignal !== 'undefined' && typeof AbortSignal.timeout === 'function') return AbortSignal.timeout(milliseconds);
  const controller = new AbortController();
  setTimeout(() => controller.abort(), milliseconds);
  return controller.signal;
}
let selectedFile = null;
let previewUrl = null;
let selectionVersion = 0;
let modelReady = false;
let busy = false;
let cameraStream = null;
let cameraOpening = false;
let cameraVersion = 0;
let fieldPhoto = true;

function updateSubmit() {
  byId('btn-submit').disabled = busy || !modelReady || !selectedFile;
  byId('btn-submit').textContent = busy ? 'Đang phân tích…' : 'Phân tích ảnh';
  byId('input-controls').disabled = busy;
  byId('result-card').setAttribute('aria-busy', String(busy));
}

function inputError(message = '') {
  byId('input-error').textContent = message;
  byId('input-error').hidden = !message;
}

async function checkServerHealth() {
  const status = byId('server-status');
  try {
    const response = await fetch(`${API_BASE}/api/health`, { signal: requestTimeout(5000) });
    if (!response.ok) throw new Error('health');
    const health = await response.json();
    byId('model-scope').textContent = health.model_architecture?.startsWith('MobileNetV3')
      ? 'V3 thử nghiệm · Cà chua, khoai tây, ngô. Độ chính xác ngoài thực địa chưa được xác nhận.'
      : 'MobileNetV2 · Cà chua, khoai tây, ngô · 11 lớp lá';
    if (health.device) byId('model-scope').textContent += health.device.startsWith('cuda') ? ' · Máy chủ GPU CUDA' : ' · Máy chủ CPU';
    modelReady = health.status === 'HEALTHY' && health.weights_loaded === true;
    status.textContent = modelReady ? 'Model sẵn sàng' : 'Máy chủ đã chạy, model chưa sẵn sàng';
    status.className = `badge ${modelReady ? 'badge-online' : 'badge-offline'}`;
  } catch {
    modelReady = false;
    status.textContent = 'Chưa kết nối được máy chủ. Kiểm tra server rồi thử lại.';
    status.className = 'badge badge-offline';
  }
  updateSubmit();
}

function stopCamera() {
  cameraVersion++;
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  byId('camera-video').srcObject = null;
  byId('camera-panel').hidden = true;
}

async function selectFile(file, { fromCrop = false, isDemo = false } = {}) {
  if (busy || !file) return;
  const version = ++selectionVersion;
  if (!fromCrop) fieldPhoto = !isDemo;
  selectedFile = null;
  inputError();
  byId('result-box').hidden = true;
  byId('empty-state').hidden = false;
  byId('empty-state').textContent = 'Chọn ảnh để xem dự đoán và thông tin chăm sóc tham khảo.';
  byId('image-preview').style.display = 'none';
  byId('file-info').textContent = '';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  if (!fromCrop) window.leafCrop.clear();
  updateSubmit();
  if (!file.size) return inputError('Ảnh đang rỗng. Hãy chọn file khác.');
  if (file.size > 10 * 1024 * 1024) return inputError('Ảnh vượt 10 MB. Hãy giảm kích thước hoặc chọn ảnh khác.');
  if (!/\.(jpe?g|png)$/i.test(file.name) || (file.type && !['image/jpeg', 'image/png'].includes(file.type))) {
    return inputError('Chỉ hỗ trợ JPG và PNG. Hãy chọn lại ảnh.');
  }
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    if (version !== selectionVersion) { URL.revokeObjectURL(url); return; }
    if (Math.min(image.naturalWidth, image.naturalHeight) < 32 || image.naturalWidth * image.naturalHeight > 20000000) {
      throw new Error('Ảnh cần có mỗi cạnh ít nhất 32 px và không quá 20 megapixel.');
    }
    previewUrl = url;
    selectedFile = file;
    byId('image-preview').src = url;
    byId('image-preview').style.display = 'block';
    if (!fromCrop) window.leafCrop.setImage(image, file);
    byId('file-info').textContent = `${file.name} · ${image.naturalWidth} × ${image.naturalHeight} px · ${(file.size / 1024).toFixed(0)} KB`;
  } catch (error) {
    URL.revokeObjectURL(url);
    if (version === selectionVersion) inputError(error.message.includes('32 px') ? error.message : 'Không đọc được ảnh. Hãy chọn JPG hoặc PNG hợp lệ.');
  }
  updateSubmit();
}

function fillList(id, values = []) {
  const list = byId(id);
  list.replaceChildren();
  for (const text of values) {
    const item = document.createElement('li');
    item.textContent = text;
    list.append(item);
  }
}

function renderResult(data, requestMs) {
  const prediction = data.prediction;
  const confirmed = data.diagnosis_status === 'CONFIRMED';
  const advice = data.advisory_details || {};
  byId('empty-state').hidden = true;
  byId('result-box').hidden = false;
  byId('disease-name').textContent = confirmed ? (prediction.is_healthy ? 'Dự đoán: lá thuộc lớp khỏe' : advice.vietnamese_name || prediction.disease_id) : (data.review_reason === 'visible_damage_conflict' ? 'Cần kiểm tra tổn thương' : data.review_reason === 'field_healthy_unverified' ? 'Chưa thể xác nhận lá khỏe' : data.review_reason ? 'Chưa xác định được tình trạng lá' : 'Chưa đủ độ tin cậy');
  byId('scientific-name').textContent = prediction.is_healthy && data.review_reason ? 'Model nghiêng về lớp khỏe, cần kiểm tra lại.' : confirmed ? advice.scientific_name || '' : `Dự đoán sơ bộ: ${advice.preliminary_assessment || 'chưa xác định'}`;
  byId('status-badge').textContent = confirmed ? (prediction.is_healthy ? 'Chưa xác nhận lá khỏe' : 'Đạt ngưỡng tham khảo') : 'Cần chụp lại';
  byId('status-badge').className = `badge ${confirmed ? 'badge-online' : 'badge-offline'}`;
  byId('metric-conf').textContent = prediction.confidence >= .99995 ? '≥99,99%' : `${(prediction.confidence * 100).toLocaleString('vi-VN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
  byId('metric-latency').textContent = `${prediction.latency_ms.toFixed(1)} ms`;
  byId('confidence-meter').value = prediction.confidence * 100;
  byId('request-time').textContent = `Toàn bộ yêu cầu: ${requestMs.toFixed(0)} ms. Ngưỡng tư vấn: ${(prediction.threshold_applied * 100).toFixed(0)}%.`;
  byId('result-message').textContent = confirmed && prediction.is_healthy ? `${data.message} Kết quả này chưa xác nhận cây khỏe. Các dạng tổn thương khác có thể chưa nằm trong bộ lớp đã học.` : data.message;
  byId('advisory-panels').hidden = !confirmed;
  byId('uncertain-guidance').hidden = confirmed;
  fillList('guidance-list', advice.guidance_for_farmer);
  byId('symptoms-text').textContent = advice.visual_symptoms || '';
  byId('climate-text').textContent = advice.environmental_triggers || '';
  fillList('cultural-list', advice.cultural_preventions);
  fillList('safety-list', advice.safety_precautions);
  byId('treatment-notice').textContent = prediction.is_healthy
    ? 'Không có khuyến nghị dùng thuốc từ kết quả này. Tiếp tục theo dõi cây.' : advice.treatment_notice || '';
  fillList('chemical-list', (advice.chemical_controls || []).map((item) => `${item.active_ingredient}: ${item.dosage}. Thời gian cách ly: ${item.pre_harvest_interval_days} ngày.`));
  byId('sources').replaceChildren();
  for (const source of advice.sources || []) {
    if (!source.url.startsWith('https://')) continue;
    const item = document.createElement('li');
    const link = document.createElement('a');
    link.href = source.url; link.target = '_blank'; link.rel = 'noopener noreferrer';
    link.textContent = `${source.publisher}: ${source.title}`;
    item.append(link); byId('sources').append(item);
  }
  byId('review-note').textContent = advice.review?.scope || '';
  byId('disclaimer-text').textContent = data.regulatory_disclaimer || '';
}

async function submitDiagnosis() {
  if (busy || !selectedFile || !modelReady) return;
  busy = true;
  stopCamera();
  inputError();
  updateSubmit();
  byId('result-box').hidden = true;
  byId('empty-state').hidden = false;
  byId('empty-state').textContent = 'Đang phân tích ảnh…';
  const started = performance.now();
  try {
    const form = new FormData();
    form.append('file', selectedFile);
    const query = `?field_photo=${fieldPhoto}&visible_damage=${byId('visible-damage').checked}`;
    const response = await fetch(`${API_BASE}/api/diagnose${query}`, { method: 'POST', body: form, signal: requestTimeout(45000) });
    const data = await response.json();
    if (!response.ok) {
      if (response.status === 503) modelReady = false;
      throw new Error(typeof data.detail === 'string' ? data.detail : 'Máy chủ chưa xử lý được ảnh. Hãy thử lại.');
    }
    renderResult(data, performance.now() - started);
  } catch (error) {
    inputError(['TimeoutError', 'AbortError'].includes(error.name) ? 'Yêu cầu quá lâu. Kiểm tra kết nối rồi thử lại.' : error.message === 'Failed to fetch' ? 'Mất kết nối máy chủ. Hãy kiểm tra server rồi thử lại.' : error.message);
    byId('empty-state').textContent = 'Chưa có kết quả. Bạn có thể thử lại với ảnh đã chọn.';
  } finally {
    busy = false;
    updateSubmit();
    checkServerHealth();
  }
}

async function openCamera() {
  if (cameraOpening || busy) return;
  inputError();
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    byId('camera-file').click();
    return;
  }
  cameraOpening = true;
  byId('open-camera').disabled = true;
  try {
    stopCamera();
    const version = cameraVersion;
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: 'environment' } }, audio: false });
    if (version !== cameraVersion || busy) { stream.getTracks().forEach(track => track.stop()); return; }
    cameraStream = stream;
    byId('camera-video').srcObject = stream;
    byId('camera-panel').hidden = false;
    await byId('camera-video').play();
  } catch {
    stopCamera();
    inputError('Chưa mở được camera. Hãy cho phép camera trong trình duyệt hoặc dùng nút chọn ảnh.');
  } finally {
    cameraOpening = false;
    byId('open-camera').disabled = false;
  }
}

async function captureImage() {
  if (busy) return;
  const video = byId('camera-video');
  if (!video.videoWidth) return inputError('Camera chưa sẵn sàng. Đợi hình ảnh xuất hiện rồi chụp lại.');
  const selection = selectionVersion;
  const camera = cameraVersion;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', .92));
  if (selection !== selectionVersion || camera !== cameraVersion || busy) return;
  stopCamera();
  if (blob) await selectFile(new File([blob], 'camera_leaf.jpg', { type: 'image/jpeg' }));
}

async function loadSamples() {
  try {
    const response = await fetch(`${API_BASE}/demo/manifest.json`);
    if (!response.ok) return;
    const samples = await response.json();
    for (const sample of samples.images) {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'btn-sample'; button.textContent = sample.caption;
      button.addEventListener('click', async () => {
        if (busy) return;
        const version = ++selectionVersion;
        selectedFile = null;
        updateSubmit();
        try {
          const image = await fetch(`${API_BASE}/demo/${encodeURIComponent(sample.filename)}`, { signal: requestTimeout(10000) });
          if (!image.ok) throw new Error();
          const blob = await image.blob();
          if (version === selectionVersion) await selectFile(new File([blob], sample.filename, { type: 'image/jpeg' }), {isDemo:true});
        } catch {
          if (version === selectionVersion) inputError('Không tải được ảnh mẫu. Hãy chọn ảnh từ máy.');
        }
      });
      byId('sample-buttons').append(button);
    }
  } catch { /* Optional samples do not block local image uploads. */ }
}

byId('file-input').addEventListener('change', (event) => { stopCamera(); selectFile(event.target.files[0]); event.target.value = ''; });
byId('camera-file').addEventListener('change', (event) => selectFile(event.target.files[0]));
byId('btn-submit').addEventListener('click', submitDiagnosis);
byId('open-camera').addEventListener('click', openCamera);
byId('close-camera').addEventListener('click', stopCamera);
byId('camera-options').addEventListener('toggle', event => { if (!event.target.open) stopCamera(); });
byId('capture').addEventListener('click', captureImage);
window.addEventListener('pagehide', () => { stopCamera(); if (previewUrl) URL.revokeObjectURL(previewUrl); });
checkServerHealth();
loadSamples();
setInterval(checkServerHealth, 15000);


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
  botDiv.textContent = 'Trợ lý Google Gemini AI đang phân tích dữ liệu nông học…';
  box.appendChild(botDiv);
  box.scrollTop = box.scrollHeight;

  try {
    const response = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg }),
      signal: requestTimeout(15000)
    });
    if (!response.ok) throw new Error('Chat API error');
    const data = await response.json();
    const replyText = data.answer || data.reply || '';
    const engineBadge = data.engine ? `<div style="font-size:11.5px; color:#1b5e20; margin-top:8px; border-top:1px dashed #a5d6a7; padding-top:4px; font-weight:600;">✨ Nguồn phản hồi: ${data.engine}</div>` : '';
    botDiv.innerHTML = replyText.replace(/\n/g, '<br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') + engineBadge;
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

async function loadRegionalWeather(regionId = 'thai_nguyen') {
  const resultsDiv = byId('weather-results');
  if (!resultsDiv) return;
  resultsDiv.innerHTML = '<p class="hint">Đang kết nối trạm khí tượng nông nghiệp và tính toán chỉ số dịch tễ học EDI...</p>';

  try {
    const res = await fetch(`${API_BASE}/api/weather/advisory?region=${regionId}`, { signal: requestTimeout(6000) });
    if (!res.ok) throw new Error('Không thể tải dữ liệu khí tượng vùng.');
    const data = await res.json();
    const w = data.weather;
    const risks = data.disease_risks || [];

    if (byId('w-temp')) byId('w-temp').value = w.temp_c;
    if (byId('w-humid')) byId('w-humid').value = w.humidity_pct;

    let html = `
      <div style="background:#fff; border:1px solid #c8e6c9; border-radius:10px; padding:16px; margin-bottom:12px; box-shadow:0 2px 6px rgba(0,0,0,0.04);">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
          <div>
            <h3 style="color:#1b5e20; font-size:18px; margin-bottom:2px;">🌦️ ${data.region_name}</h3>
            <p style="font-size:13px; color:#555;">${data.station} · Cập nhật: ${data.timestamp}</p>
          </div>
          <span class="pill-badge pill-blue" style="font-size:13px; padding:4px 10px;">${w.condition}</span>
        </div>

        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:10px; margin-top:12px;">
          <div style="background:#f9fbf9; padding:8px 12px; border-radius:8px; border:1px solid #e0ede0;">
            <div style="font-size:12px; color:#666;">Nhiệt độ</div>
            <div style="font-size:18px; font-weight:700; color:#1b5e20;">${w.temp_c} °C</div>
          </div>
          <div style="background:#f9fbf9; padding:8px 12px; border-radius:8px; border:1px solid #e0ede0;">
            <div style="font-size:12px; color:#666;">Độ ẩm không khí</div>
            <div style="font-size:18px; font-weight:700; color:#1565c0;">${w.humidity_pct} %</div>
          </div>
          <div style="background:#f9fbf9; padding:8px 12px; border-radius:8px; border:1px solid #e0ede0;">
            <div style="font-size:12px; color:#666;">Lượng mưa</div>
            <div style="font-size:18px; font-weight:700; color:#0288d1;">${w.rain_mm} mm</div>
          </div>
          <div style="background:#f9fbf9; padding:8px 12px; border-radius:8px; border:1px solid #e0ede0;">
            <div style="font-size:12px; color:#666;">Thời gian ướt lá</div>
            <div style="font-size:18px; font-weight:700; color:#d81b60;">${w.leaf_wetness_hours} giờ</div>
          </div>
        </div>

        <div style="margin-top:12px; padding:10px; background:#fafafa; border-radius:8px; font-size:13.5px; border-left:4px solid #2e7d32;">
          <strong>Nhận định tổng quan:</strong> ${data.overall_advisory}
        </div>
      </div>

      <div style="margin-top:14px;">
        <h4 style="color:#1b5e20; margin-bottom:8px;">📊 Ma trận Đánh giá Nguy cơ Dịch tễ Học (EDI):</h4>
    `;

    risks.forEach(r => {
      html += `
        <div style="margin-bottom:10px; padding:12px; background:#fff; border-radius:8px; border:1px solid #e0e0e0; border-left:5px solid ${r.badge_color};">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px;">
            <strong style="font-size:15px; color:#212121;">${r.disease_name}</strong>
            <span class="pill-badge" style="background:${r.badge_color}22; color:${r.badge_color}; font-weight:700; border:1px solid ${r.badge_color};">${r.risk_level} (Điểm: ${r.risk_score}/100)</span>
          </div>
          <p style="font-size:13px; color:#555; margin:4px 0;"><strong>Yếu tố kích hoạt:</strong> ${r.favorable_factors}</p>
          <p style="font-size:13px; color:#2e7d32; margin-top:4px;"><strong>Khuyến nghị phòng ngừa:</strong> ${r.preventive_action}</p>
        </div>
      `;
    });

    html += '</div>';
    resultsDiv.innerHTML = html;
  } catch (err) {
    resultsDiv.innerHTML = `<p class="hint" style="color:#c62828;">Lỗi tải dữ liệu khí tượng vùng: ${err.message}</p>`;
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
