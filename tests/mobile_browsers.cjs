const {chromium,webkit} = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const {spawn} = require('node:child_process');
process.chdir(path.resolve(__dirname,'..'));
const output = process.env.SIC_TEST_OUTPUT || 'tests/results/mobile_browsers';
let server,serverPid,active;
(async()=>{
  let ready=false,error='';
  server=spawn('.venv/Scripts/python.exe',['-m','uvicorn','backend.main:app','--host','127.0.0.1','--port','8002'],{windowsHide:true});
  server.stderr.on('data',bytes=>{
    const text=String(bytes),match=text.match(/Started server process \[(\d+)\]/);
    if(match)serverPid=Number(match[1]);
    if(text.includes('Uvicorn running on'))ready=true;
    if(text.includes('ERROR:'))error=text;
  });
  server.on('error',e=>{error=e.message;});
  for(let i=0;i<100 && !ready;i++){
    if(error || server.exitCode!==null)throw new Error(error || 'Test server exited');
    await new Promise(resolve=>setTimeout(resolve,500));
  }
  assert(ready,'Server did not become ready');
  await fs.mkdir(output,{recursive:true});
  const results=[];
  for(const engine of ['chromium','webkit']){
    active=engine==='chromium'
      ? await chromium.launch({executablePath:process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true})
      : await webkit.launch({headless:true});
    const context=await active.newContext({viewport:{width:390,height:844},deviceScaleFactor:2,isMobile:true,hasTouch:true});
    const page=await context.newPage();const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto('http://127.0.0.1:8002/app');
    await page.getByText('Model sẵn sàng',{exact:true}).waitFor();
    await page.locator('#file-input').setInputFiles('sample_images/demo/Potato_healthy.jpg');
    await page.locator('#crop-panel:not([hidden])').waitFor();
    await page.getByText('Chọn vùng lá để phân tích',{exact:true}).tap();
    await page.locator('#crop-x').fill('25');
    await page.locator('#crop-width').fill('50');
    await page.locator('#apply-crop').tap();
    await page.waitForFunction(()=>document.querySelector('#file-info').textContent.includes('leaf_crop.jpg'));
    assert.equal(await page.locator('#image-preview').evaluate(e=>e.naturalWidth),128);
    await page.getByText('Chọn vùng lá để phân tích',{exact:true}).tap();
    await page.locator('#restore-image').tap();
    await page.waitForFunction(()=>document.querySelector('#file-info').textContent.includes('Potato_healthy.jpg'));
    await page.locator('#btn-submit:not([disabled])').waitFor();
    await page.locator('#btn-submit').tap();
    await page.getByText('Chưa thể xác nhận lá khỏe',{exact:true}).waitFor();
    assert(await page.locator('#advisory-panels').isHidden());
    await page.locator('#visible-damage').check();
    await page.locator('#btn-submit:not([disabled])').waitFor();
    await page.locator('#btn-submit').tap();
    await page.getByText('Cần kiểm tra tổn thương',{exact:true}).waitFor();
    assert(await page.locator('#advisory-panels').isHidden());
    for(const width of [360,390,412,844]){
      await page.setViewportSize({width,height:width===844?390:844});
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=document.documentElement.clientWidth));
      assert(parseFloat(await page.locator('#result-card').evaluate(e=>getComputedStyle(e).fontSize))>=18);
      await page.screenshot({path:path.join(output,`${engine}-${width}.png`),fullPage:true});
    }
    assert.deepEqual(errors,[]);
    results.push({engine,passed:true,viewports:[360,390,412,844],touch_controls:true,crop:true,restore:true,damage_review:true,page_errors:errors});
    await active.close();active=null;
  }
  await fs.writeFile(path.join(output,'verification.json'),JSON.stringify({passed:true,results,
    limitation:'Desktop Windows engines with mobile viewport/touch emulation. Physical Android/iPhone and physical camera not tested.'},null,2));
  console.log('Chromium and WebKit mobile checks passed. Physical devices not tested.');
})().catch(error=>{console.error(error);process.exitCode=1;}).finally(async()=>{
  if(active)await active.close();
  if(serverPid){try{process.kill(serverPid);}catch{}}
  server?.kill();
});
