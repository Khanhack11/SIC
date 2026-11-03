/* Shared-origin demo client; no model predictions or clinical claims are hard-coded. */
const API_BASE = (location.protocol === 'file:' || location.port === '8080')
  ? 'http://127.0.0.1:8000' : location.origin;
const byId = (id) => document.getElementById(id);
let selectedFile = null;
let previewUrl = null;
let selectionVersion = 0;
let modelReady = false;
let busy = false;
let cameraStream = null;
let cameraOpening = false;
let cameraVersion = 0;

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
    const response = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(5000) });
    if (!response.ok) throw new Error('health');
    const health = await response.json();
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

async function selectFile(file) {
  if (busy || !file) return;
  const version = ++selectionVersion;
  selectedFile = null;
  inputError();
  byId('result-box').hidden = true;
  byId('empty-state').hidden = false;
  byId('empty-state').textContent = 'Chọn ảnh để xem dự đoán và thông tin chăm sóc tham khảo.';
  byId('image-preview').style.display = 'none';
  byId('file-info').textContent = '';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
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
  byId('disease-name').textContent = confirmed ? (prediction.is_healthy ? 'Ảnh lá thuộc lớp khỏe' : advice.vietnamese_name || prediction.disease_id) : 'Chưa đủ độ tin cậy';
  byId('scientific-name').textContent = confirmed ? advice.scientific_name || '' : `Dự đoán sơ bộ: ${advice.preliminary_assessment || 'chưa xác định'}`;
  byId('status-badge').textContent = confirmed ? (prediction.is_healthy ? 'Lớp lá khỏe' : 'Đạt ngưỡng tham khảo') : 'Cần chụp lại';
  byId('status-badge').className = `badge ${confirmed ? 'badge-online' : 'badge-offline'}`;
  byId('metric-conf').textContent = `${(prediction.confidence * 100).toLocaleString('vi-VN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}%`;
  byId('metric-latency').textContent = `${prediction.latency_ms.toFixed(1)} ms`;
  byId('confidence-meter').value = prediction.confidence * 100;
  byId('request-time').textContent = `Toàn bộ yêu cầu: ${requestMs.toFixed(0)} ms. Ngưỡng tư vấn: ${(prediction.threshold_applied * 100).toFixed(0)}%.`;
  byId('result-message').textContent = data.message;
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
    const response = await fetch(`${API_BASE}/api/diagnose`, { method: 'POST', body: form, signal: AbortSignal.timeout(45000) });
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
  const video = byId('camera-video');
  if (!video.videoWidth) return inputError('Camera chưa sẵn sàng. Đợi hình ảnh xuất hiện rồi chụp lại.');
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', .92));
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
          const image = await fetch(`${API_BASE}/demo/${encodeURIComponent(sample.filename)}`);
          if (!image.ok) throw new Error();
          const blob = await image.blob();
          if (version === selectionVersion) await selectFile(new File([blob], sample.filename, { type: 'image/jpeg' }));
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
byId('capture').addEventListener('click', captureImage);
window.addEventListener('pagehide', () => { stopCamera(); if (previewUrl) URL.revokeObjectURL(previewUrl); });
checkServerHealth();
loadSamples();
setInterval(checkServerHealth, 15000);
