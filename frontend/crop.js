/* Automatic & Manual Reversible Crop with AI Leaf/Lesion Detection (Magenta Bounding Boxes) */
(() => {
  const el = id => document.getElementById(id);
  const canvas = el('crop-canvas');
  const context = canvas.getContext('2d');
  let source = null;
  let original = null;
  let anchor = null;
  let pointerId = null;
  let rectangle = [0, 0, 1, 1];
  let detectedBoxes = [];
  const fields = ['crop-x', 'crop-y', 'crop-width', 'crop-height'];

  function draw() {
    if (!source) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(source, 0, 0, canvas.width, canvas.height);

    // 1. Draw all AI-detected leaf bounding boxes in Magenta (#e91e63) like sample image
    if (detectedBoxes && detectedBoxes.length > 0) {
      detectedBoxes.forEach((box, idx) => {
        const [nx, ny, nw, nh] = box.norm_box;
        const bx = nx * canvas.width;
        const by = ny * canvas.height;
        const bw = nw * canvas.width;
        const bh = nh * canvas.height;

        context.save();
        context.strokeStyle = '#e91e63'; // Magenta/pink border
        context.lineWidth = 3;
        context.strokeRect(bx, by, bw, bh);

        // Label tag
        const tagText = `Lá ${idx + 1}${idx === 0 ? ' (Bệnh chính)' : ''}`;
        context.font = 'bold 12px Inter, sans-serif';
        const textWidth = context.measureText(tagText).width;
        context.fillStyle = '#e91e63';
        context.fillRect(bx, Math.max(0, by - 20), textWidth + 8, 20);
        context.fillStyle = '#ffffff';
        context.fillText(tagText, bx + 4, Math.max(14, by - 5));
        context.restore();
      });
    }

    // 2. Draw user/selected active crop rectangle with dark overlay
    const [x, y, w, h] = rectangle.map((value, index) => value * (index % 2 ? canvas.height : canvas.width));
    context.fillStyle = '#102d1588';
    context.beginPath();
    context.rect(0, 0, canvas.width, canvas.height);
    context.rect(x + w, y, -w, h);
    context.fill('evenodd');

    context.strokeStyle = '#00e676';
    context.lineWidth = 2.5;
    context.setLineDash([6, 4]);
    context.strokeRect(x, y, w, h);
    context.setLineDash([]);

    fields.forEach((id, index) => { el(id).value = +(rectangle[index] * 100).toFixed(1); });
  }

  function position(event) {
    const bounds = canvas.getBoundingClientRect();
    return [
      Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width)),
      Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height))
    ];
  }

  canvas.addEventListener('pointerdown', event => {
    if (!source || busy || pointerId !== null) return;
    pointerId = event.pointerId;
    anchor = position(event);
    canvas.setPointerCapture(pointerId);
  });

  canvas.addEventListener('pointermove', event => {
    if (!anchor || event.pointerId !== pointerId) return;
    const point = position(event);
    rectangle = [
      Math.min(anchor[0], point[0]),
      Math.min(anchor[1], point[1]),
      Math.abs(anchor[0] - point[0]),
      Math.abs(anchor[1] - point[1])
    ];
    draw();
  });

  for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
    canvas.addEventListener(name, () => {
      anchor = null;
      pointerId = null;
    });
  }

  fields.forEach(id => el(id).addEventListener('input', () => {
    const values = fields.map(id => Number(el(id).value) / 100);
    if (!values.every(Number.isFinite)) return;
    values[0] = Math.max(0, Math.min(.99, values[0]));
    values[1] = Math.max(0, Math.min(.99, values[1]));
    values[2] = Math.max(.01, Math.min(1 - values[0], values[2]));
    values[3] = Math.max(.01, Math.min(1 - values[1], values[3]));
    rectangle = values;
    draw();
  }));

  el('apply-crop').addEventListener('click', async () => {
    if (!source || busy) return;
    const version = selectionVersion;
    const [x, y, w, h] = rectangle;
    const left = Math.round(source.naturalWidth * x), top = Math.round(source.naturalHeight * y);
    const right = Math.min(source.naturalWidth, Math.round(source.naturalWidth * (x + w)));
    const bottom = Math.min(source.naturalHeight, Math.round(source.naturalHeight * (y + h)));
    const width = right - left, height = bottom - top;
    if (Math.min(width, height) < 32) return inputError('Vùng cắt cần có mỗi cạnh ít nhất 32 px. Hãy chọn vùng lớn hơn.');
    const output = document.createElement('canvas');
    output.width = width; output.height = height;
    output.getContext('2d').drawImage(source, left, top, width, height, 0, 0, width, height);
    const blob = await new Promise(resolve => output.toBlob(resolve, 'image/jpeg', .95));
    if (!blob || version !== selectionVersion || busy) return;
    await selectFile(new File([blob], 'leaf_crop.jpg', { type: 'image/jpeg' }), { fromCrop: true });
    el('crop-status').textContent = `Đã cắt vùng ${width} × ${height} px. Ảnh xem trước là ảnh gửi phân tích.`;
    el('crop-editor').open = false;
  });

  el('restore-image').addEventListener('click', async () => {
    if (!original || busy) return;
    detectedBoxes = [];
    renderBoxButtons([]);
    await selectFile(original);
    el('crop-status').textContent = 'Đã khôi phục ảnh gốc.';
  });

  // AUTO DETECT LEAF REGIONS (Excess Green & Lesion Saliency)
  async function autoDetectLeaves() {
    if (!original || busy) return;
    const btn = el('btn-auto-detect');
    const origText = btn.innerHTML;
    try {
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Đang nhận diện lá...';
      el('crop-status').textContent = 'Thuật toán ExG + Lesion Saliency đang phân tích cấu trúc tán lá...';

      const formData = new FormData();
      formData.append('file', original);

      const resp = await fetch(`${API_BASE}/api/detect-leaves`, {
        method: 'POST',
        body: formData,
        signal: requestTimeout(10000)
      });
      if (!resp.ok) throw new Error('Không thể nhận diện lá tự động.');
      const data = await resp.json();

      detectedBoxes = data.boxes || [];
      if (detectedBoxes.length === 0) {
        el('crop-status').textContent = 'Không tìm thấy cụm lá rõ nét. Vui lòng tự kéo khung thủ công.';
        return;
      }

      // Pick primary diseased crop box
      if (data.primary_crop_str) {
        const parts = data.primary_crop_str.split(',').map(Number);
        if (parts.length === 4) {
          rectangle = parts;
        }
      } else if (detectedBoxes[0]) {
        rectangle = [...detectedBoxes[0].norm_box];
      }

      el('crop-editor').open = true;
      renderBoxButtons(detectedBoxes);
      draw();
      el('crop-status').textContent = `✨ AI tự động phát hiện ${data.count} cụm lá & vùng tổn thương (Hộp màu hồng tím). Đã tự động chọn lá bệnh nổi bật.`;
    } catch (err) {
      el('crop-status').textContent = 'Lỗi nhận diện tự động: ' + err.message;
    } finally {
      btn.disabled = false;
      btn.innerHTML = origText;
    }
  }

  function renderBoxButtons(boxes) {
    let container = el('detected-leaf-chips');
    if (!container) return;
    container.innerHTML = '';
    if (!boxes || boxes.length === 0) {
      container.hidden = true;
      return;
    }
    container.hidden = false;
    boxes.forEach((box, idx) => {
      const bBtn = document.createElement('button');
      bBtn.type = 'button';
      bBtn.className = 'chat-tag-btn';
      bBtn.style.borderColor = '#e91e63';
      bBtn.style.color = '#c2185b';
      bBtn.style.fontWeight = '600';
      const areaPct = box.area_ratio !== undefined ? Math.round(box.area_ratio * 100) : (box.norm_box ? Math.round(box.norm_box[2] * box.norm_box[3] * 100) : 25);
      bBtn.innerHTML = `🌿 Hộp ${idx + 1} (${areaPct}% diện tích)`;
      bBtn.onclick = () => {
        rectangle = [...box.norm_box];
        draw();
        el('crop-status').textContent = `Đã chọn vùng Hộp ${idx + 1}. Bấm "Cắt vùng đã chọn" để áp dụng.`;
      };
      container.appendChild(bBtn);
    });
  }

  // 1-CLICK AUTO DIAGNOSE
  async function autoDiagnose() {
    if (!original || busy || !modelReady) return;
    const btn = el('btn-auto-diagnose');
    const origText = btn.innerHTML;
    try {
      busy = true;
      updateSubmit();
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span> Đang tự động crop & chẩn đoán...';
      el('empty-state').hidden = false;
      el('empty-state').textContent = 'Đang tự động nhận diện vùng lá bệnh và phân tích phác đồ...';
      el('result-box').hidden = true;

      const formData = new FormData();
      formData.append('file', original);
      formData.append('visible_damage', String(el('visible-damage').checked));

      const started = performance.now();
      const resp = await fetch(`${API_BASE}/api/auto-diagnose`, {
        method: 'POST',
        body: formData,
        signal: requestTimeout(15000)
      });
      if (!resp.ok) throw new Error('Không thể chẩn đoán tự động.');
      const data = await resp.json();
      const requestMs = performance.now() - started;

      if (data.detection_info && data.detection_info.boxes) {
        detectedBoxes = data.detection_info.boxes;
        if (data.detection_info.primary_crop_str) {
          rectangle = data.detection_info.primary_crop_str.split(',').map(Number);
        }
        renderBoxButtons(detectedBoxes);
        draw();
      }

      renderResult(data, requestMs);
    } catch (err) {
      inputError(err.message || 'Lỗi xử lý tự động.');
    } finally {
      busy = false;
      updateSubmit();
      btn.disabled = false;
      btn.innerHTML = origText;
    }
  }

  if (el('btn-auto-detect')) {
    el('btn-auto-detect').addEventListener('click', autoDetectLeaves);
  }
  if (el('btn-auto-diagnose')) {
    el('btn-auto-diagnose').addEventListener('click', autoDiagnose);
  }

  window.leafCrop = {
    clear() {
      source = null; original = null; anchor = null; pointerId = null;
      detectedBoxes = []; renderBoxButtons([]);
      el('crop-panel').hidden = true;
    },
    setImage(image, file) {
      source = image; original = file; rectangle = [0, 0, 1, 1];
      detectedBoxes = []; renderBoxButtons([]);
      canvas.width = Math.min(800, image.naturalWidth);
      canvas.height = Math.max(1, Math.round(canvas.width * image.naturalHeight / image.naturalWidth));
      el('crop-panel').hidden = false;
      el('crop-status').textContent = 'Chưa cắt ảnh. Bạn có thể tự kéo khung hoặc dùng nút "✨ Tự động nhận diện lá & đốm bệnh".';
      draw();
    },
    autoDetect: autoDetectLeaves,
    autoDiagnose: autoDiagnose
  };
})();
