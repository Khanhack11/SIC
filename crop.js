/* Manual, reversible crop. The preview and uploaded file contain the same pixels. */
(() => {
  const el = id => document.getElementById(id);
  const canvas = el('crop-canvas');
  const context = canvas.getContext('2d');
  let source = null;
  let original = null;
  let anchor = null;
  let pointerId = null;
  let rectangle = [0, 0, 1, 1];
  const fields = ['crop-x', 'crop-y', 'crop-width', 'crop-height'];

  function draw() {
    if (!source) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    const [x,y,w,h] = rectangle.map((value,index) => value * (index%2 ? canvas.height : canvas.width));
    context.fillStyle = '#102d1599';
    context.beginPath();
    context.rect(0,0,canvas.width,canvas.height);
    context.rect(x+w,y,-w,h);
    context.fill('evenodd');
    context.strokeStyle = '#ffffff'; context.lineWidth = 3;
    context.strokeRect(x,y,w,h);
    fields.forEach((id,index) => { el(id).value = +(rectangle[index]*100).toFixed(1); });
  }
  function position(event) {
    const bounds = canvas.getBoundingClientRect();
    return [Math.max(0,Math.min(1,(event.clientX-bounds.left)/bounds.width)),
      Math.max(0,Math.min(1,(event.clientY-bounds.top)/bounds.height))];
  }
  canvas.addEventListener('pointerdown',event => {
    if (!source || busy || pointerId !== null) return;
    pointerId=event.pointerId; anchor=position(event);
    canvas.setPointerCapture(pointerId);
  });
  canvas.addEventListener('pointermove',event => {
    if (!anchor || event.pointerId!==pointerId) return;
    const point=position(event);
    rectangle=[Math.min(anchor[0],point[0]),Math.min(anchor[1],point[1]),Math.abs(anchor[0]-point[0]),Math.abs(anchor[1]-point[1])];
    draw();
  });
  for (const name of ['pointerup','pointercancel','lostpointercapture']) canvas.addEventListener(name,() => {
    anchor=null; pointerId=null;
  });
  fields.forEach(id => el(id).addEventListener('input',() => {
    const values=fields.map(id => Number(el(id).value)/100);
    if (!values.every(Number.isFinite)) return;
    values[0]=Math.max(0,Math.min(.99,values[0]));
    values[1]=Math.max(0,Math.min(.99,values[1]));
    values[2]=Math.max(.01,Math.min(1-values[0],values[2]));
    values[3]=Math.max(.01,Math.min(1-values[1],values[3]));
    rectangle=values; draw();
  }));
  el('apply-crop').addEventListener('click',async () => {
    if (!source || busy) return;
    const version=selectionVersion;
    const [x,y,w,h]=rectangle;
    const width=Math.round(source.naturalWidth*w), height=Math.round(source.naturalHeight*h);
    if (Math.min(width,height)<32) return inputError('Vùng cắt cần có mỗi cạnh ít nhất 32 px. Hãy chọn vùng lớn hơn.');
    const output=document.createElement('canvas');
    output.width=width; output.height=height;
    output.getContext('2d').drawImage(source,Math.round(x*source.naturalWidth),Math.round(y*source.naturalHeight),width,height,0,0,width,height);
    const blob=await new Promise(resolve => output.toBlob(resolve,'image/jpeg',.95));
    if (!blob || version!==selectionVersion || busy) return;
    await selectFile(new File([blob],'leaf_crop.jpg',{type:'image/jpeg'}),{fromCrop:true});
    el('crop-status').textContent=`Đã cắt vùng ${width} × ${height} px. Ảnh xem trước là ảnh gửi phân tích.`;
    el('crop-editor').open=false;
  });
  el('restore-image').addEventListener('click',async () => {
    if (!original || busy) return;
    await selectFile(original);
    el('crop-status').textContent='Đã khôi phục ảnh gốc.';
  });
  window.leafCrop={
    clear() { source=null; original=null; anchor=null; pointerId=null; el('crop-panel').hidden=true; },
    setImage(image,file) {
      source=image; original=file; rectangle=[0,0,1,1];
      canvas.width=Math.min(800,image.naturalWidth);
      canvas.height=Math.max(1,Math.round(canvas.width*image.naturalHeight/image.naturalWidth));
      el('crop-panel').hidden=false; el('crop-status').textContent='Chưa cắt ảnh.'; draw();
    }
  };
})();
