const {chromium} = require('playwright');
const fs = require('node:fs/promises');
const assert = require('node:assert/strict');
const path = require('node:path');
process.chdir(path.resolve(__dirname, '..'));
const outputDir = process.env.SIC_TEST_OUTPUT || 'tests/results/browser';
let activeBrowser;
let server;
let serverPid;
function stopServer() {
  if (serverPid) { try { process.kill(serverPid); } catch {} }
  server?.kill();
}

(async () => {
  let listening = false;
  let startupError = '';
  server=require('node:child_process').spawn('.venv/Scripts/python.exe',['-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8001'],{cwd:process.cwd(),windowsHide:true});
  server.on('error',error=>{startupError=error.message;});
  server.stderr.on('data',data=>{
    const text=String(data);
    const match=text.match(/Started server process \[(\d+)\]/);
    if(match)serverPid=Number(match[1]);
    if(text.includes('Uvicorn running on'))listening=true;
    if(text.includes('ERROR:'))startupError=text;
  });
  let healthy = false;
  for(let attempt=0;attempt<60;attempt++) {
    if(startupError || server.exitCode!==null)throw new Error(`Test server failed: ${startupError || server.exitCode}`);
    if(listening) {
      try { const health=await fetch('http://127.0.0.1:8001/api/health'); if(health.ok) {healthy=true;break;} } catch {}
    }
    await new Promise(resolve=>setTimeout(resolve,500));
  }
  assert(healthy,'Test server did not become ready on port 8001');
  const browser = await chromium.launch({executablePath:process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true,args:['--disable-gpu','--use-fake-device-for-media-stream']});
  activeBrowser=browser;
  const page = await browser.newPage({viewport:{width:1280,height:900}});
  const errors=[];
  page.on('pageerror', error=>errors.push(error.message));
  await page.goto('http://127.0.0.1:8001/app');
  await page.getByText('Model sẵn sàng',{exact:true}).waitFor();
  await fs.mkdir(outputDir,{recursive:true});
  await page.screenshot({path:path.join(outputDir,'desktop-empty.png'),fullPage:true});
  await page.context().grantPermissions(['camera'],{origin:'http://127.0.0.1:8001'});
  await page.locator('#open-camera').click();
  await page.locator('#camera-panel:not([hidden])').waitFor();
  await page.waitForFunction(()=>document.querySelector('#camera-video').videoWidth>0);
  await page.locator('#capture').click();
  await page.waitForFunction(()=>document.querySelector('#file-info').textContent.includes('camera_leaf.jpg'));
  assert(await page.locator('#camera-panel').isHidden());
  await page.evaluate(()=>{navigator.mediaDevices.getUserMedia=async()=>{throw new DOMException('Denied','NotAllowedError');};});
  await page.locator('#open-camera').click();
  await page.getByText('Chưa mở được camera. Hãy cho phép camera trong trình duyệt hoặc dùng nút chọn ảnh.').waitFor();
  await page.locator('#file-input').setInputFiles({name:'broken.jpg',mimeType:'image/jpeg',buffer:Buffer.from('invalid image')});
  await page.getByText('Không đọc được ảnh. Hãy chọn JPG hoặc PNG hợp lệ.').waitFor();
  assert(await page.locator('#btn-submit').isDisabled());
  await page.locator('#file-input').setInputFiles({name:'too-big.jpg',mimeType:'image/jpeg',buffer:Buffer.alloc(10*1024*1024+1)});
  await page.getByText('Ảnh vượt 10 MB. Hãy giảm kích thước hoặc chọn ảnh khác.').waitFor();
  await page.getByText('Thử với ảnh lá mẫu',{exact:true}).click();
  await page.getByRole('button',{name:'Ngô, gỉ sắt',exact:true}).click();
  await page.locator('#btn-submit:not([disabled])').waitFor();
  let posts=0;
  page.on('request',request=>{if(request.url().endsWith('/api/diagnose') && request.method()==='POST') posts++;});
  await page.locator('#btn-submit').evaluate(button=>{button.click();button.click();});
  await page.getByText('Đạt ngưỡng tham khảo',{exact:true}).waitFor();
  assert.equal(posts,1);
  assert(await page.locator('#advisory-panels').isVisible());
  assert.equal(await page.locator('#chemical-list li').count(),0);
  await page.getByText('Lưu ý và nguồn tham khảo',{exact:true}).click();
  assert((await page.locator('#sources a').count())>=1);
  await page.screenshot({path:path.join(outputDir,'desktop-disease.png'),fullPage:true});
  await page.getByRole('button',{name:'Ảnh khó, độ tin cậy thấp',exact:true}).click();
  await page.locator('#btn-submit:not([disabled])').waitFor();
  await page.locator('#btn-submit').click();
  await page.getByText('Cần chụp lại',{exact:true}).waitFor();
  assert(await page.locator('#advisory-panels').isHidden());
  assert(await page.locator('#uncertain-guidance').isVisible());
  await page.screenshot({path:path.join(outputDir,'desktop-uncertain.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.getByRole('button',{name:'Khoai tây, lá khỏe',exact:true}).click();
  await page.locator('#btn-submit:not([disabled])').waitFor();
  await page.locator('#btn-submit').click();
  await page.getByText('Lớp lá khỏe',{exact:true}).waitFor();
  assert(await page.evaluate(()=>document.documentElement.scrollWidth <= document.documentElement.clientWidth));
  await page.screenshot({path:path.join(outputDir,'mobile-healthy.png'),fullPage:true});
  // A slow earlier sample must never replace the user's newer local selection.
  let releaseSample;
  const delayedSample = new Promise(resolve => { releaseSample = resolve; });
  let sampleStarted;
  const samplePending = new Promise(resolve => { sampleStarted = resolve; });
  await page.route('**/demo/Corn_Common_rust.jpg',async route=>{
    sampleStarted();
    await delayedSample;
    await route.continue();
  });
  await page.getByRole('button',{name:'Ngô, gỉ sắt',exact:true}).click();
  await samplePending;
  assert(await page.locator('#btn-submit').isDisabled(), 'Pending sample must disable analysis of the previous image');
  await page.locator('#file-input').setInputFiles('sample_images/demo/Potato_healthy.jpg');
  await page.waitForFunction(()=>document.querySelector('#file-info').textContent.includes('Potato_healthy.jpg'));
  releaseSample();
  await page.waitForResponse(response=>response.url().endsWith('/demo/Corn_Common_rust.jpg'));
  await page.waitForTimeout(300);
  assert.match(await page.locator('#file-info').textContent(), /Potato_healthy.jpg/, 'Slow old sample replaced newer local image');
  await page.unroute('**/demo/Corn_Common_rust.jpg');
  // HTTP errors and a broken connection must restore controls for retry.
  for (const status of [400,500,503]) {
    await page.route('**/api/diagnose',route=>route.fulfill({status,json:{detail:`Demo error ${status}`}}));
    await page.locator('#btn-submit:not([disabled])').waitFor();
    await page.locator('#btn-submit').click();
    await page.getByText(`Demo error ${status}`,{exact:true}).waitFor();
    await page.locator('#btn-submit:not([disabled])').waitFor();
    await page.unroute('**/api/diagnose');
  }
  await page.route('**/api/diagnose',route=>route.abort());
  await page.locator('#btn-submit').click();
  await page.getByText('Mất kết nối máy chủ. Hãy kiểm tra server rồi thử lại.',{exact:true}).waitFor();
  await page.locator('#btn-submit:not([disabled])').waitFor();
  await page.unroute('**/api/diagnose');
  await page.locator('#btn-submit').click();
  await page.getByText('Lớp lá khỏe',{exact:true}).waitFor();
  await page.route('**/api/health',route=>route.fulfill({json:{status:'MODEL_NOT_READY',weights_loaded:false}}));
  await page.evaluate(()=>checkServerHealth());
  await page.getByText('Máy chủ đã chạy, model chưa sẵn sàng',{exact:true}).waitFor();
  assert(await page.locator('#btn-submit').isDisabled());
  await page.screenshot({path:path.join(outputDir,'mobile-model-not-ready.png'),fullPage:true});
  await page.unroute('**/api/health');
  await page.route('**/api/health',route=>route.abort());
  await page.evaluate(()=>checkServerHealth());
  await page.getByText('Chưa kết nối được máy chủ. Kiểm tra server rồi thử lại.',{exact:true}).waitFor();
  assert(await page.locator('#btn-submit').isDisabled());
  assert.deepEqual(errors,[]);
  await fs.writeFile(path.join(outputDir,'frontend_verification.json'),JSON.stringify({passed:true,checks:['simulated camera capture and stream cleanup','camera permission denied','corrupt image','oversize image','disease inference','duplicate submission prevention','sources','unreviewed treatments withheld','uncertain guidance','healthy leaf','mobile overflow','model not ready','offline server','slow sample does not replace newer local file','HTTP 400 recovery','HTTP 500 recovery','HTTP 503 recovery','network failure recovery','successful retry'],page_errors:errors},null,2));
  await browser.close();
  stopServer();
  console.log('Frontend verification passed: 19 checks, no JavaScript page errors. Camera used a simulated device.');
})().catch(async error=>{console.error(error);if(activeBrowser)await activeBrowser.close();stopServer();process.exitCode=1;});
