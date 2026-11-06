/* Record full screen demo of the new interface using Playwright */
const { chromium } = require('playwright');
const path = require('node:path');
const fs = require('node:fs/promises');
const child_process = require('node:child_process');

const root = path.resolve(__dirname, '..');
const outDir = path.resolve(root, '..', '..', 'Reports', 'demo_new_ui');

(async () => {
  await fs.mkdir(outDir, { recursive: true });
  console.log('Output directory:', outDir);

  console.log('Starting demo server on port 8002...');
  const server = child_process.spawn(
    path.join(root, '.venv', 'Scripts', 'python.exe'),
    ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8002'],
    { cwd: root, windowsHide: true }
  );

  let serverPid;
  server.stderr.on('data', data => {
    const text = String(data);
    const m = text.match(/Started server process \[(\d+)\]/);
    if (m) serverPid = Number(m[1]);
  });

  // Wait for health
  let healthy = false;
  for (let i = 0; i < 40; i++) {
    try {
      const res = await fetch('http://127.0.0.1:8002/api/health');
      if (res.ok) { healthy = true; break; }
    } catch {}
    await new Promise(r => setTimeout(r, 500));
  }

  if (!healthy) {
    if (serverPid) process.kill(serverPid);
    server.kill();
    throw new Error('Server failed to start on port 8002');
  }
  console.log('Server is healthy! Launching browser with video recording...');

  const browser = await chromium.launch({
    executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true,
    args: ['--disable-gpu', '--use-fake-device-for-media-stream']
  });

  const context = await browser.newContext({
    viewport: { width: 1280, height: 900 },
    recordVideo: {
      dir: outDir,
      size: { width: 1280, height: 900 }
    }
  });

  const page = await context.newPage();

  console.log('Navigating to http://127.0.0.1:8002/app...');
  await page.goto('http://127.0.0.1:8002/app', { waitUntil: 'networkidle', timeout: 30000 });
  await page.waitForTimeout(2000);

  // 1. Select sample image
  console.log('Selecting demo image: Corn_Common_rust.jpg...');
  const samplePath = path.join(root, 'sample_images', 'demo', 'Corn_Common_rust.jpg');
  await page.locator('#file-input').setInputFiles(samplePath);
  await page.waitForTimeout(1500);

  // 2. Click analyze
  console.log('Analyzing image...');
  await page.locator('#btn-submit').click();
  await page.waitForResponse(r => r.url().includes('/api/diagnose'), { timeout: 15000 });
  await page.locator('#result-box:not([hidden])').waitFor({ timeout: 10000 });
  await page.waitForTimeout(2500);

  // 3. Expand accordion panels
  console.log('Expanding advisory panels...');
  const details = await page.locator('details').all();
  for (const d of details) {
    await d.evaluate(el => el.open = true).catch(() => {});
  }
  await page.waitForTimeout(2000);

  // 4. Smooth scroll down to demonstrate new UI features
  console.log('Scrolling through new UI modules...');
  const totalHeight = await page.evaluate(() => document.body.scrollHeight);
  const scrollSteps = 6;
  for (let s = 1; s <= scrollSteps; s++) {
    const y = Math.floor((totalHeight / scrollSteps) * s);
    await page.evaluate(top => window.scrollTo({ top, behavior: 'smooth' }), y);
    await page.waitForTimeout(2000);
  }

  // Scroll back to top
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
  await page.waitForTimeout(2000);

  // Take crisp full-page screenshot and viewport screenshot for poster frame
  await page.screenshot({ path: path.join(outDir, 'new_ui_hero.png'), fullPage: false });
  await page.screenshot({ path: path.join(outDir, 'new_ui_fullpage.png'), fullPage: true });

  console.log('Closing browser to finalize video file...');
  await page.close();
  await context.close();
  await browser.close();

  // Stop server
  if (serverPid) {
    try { process.kill(serverPid); } catch {}
  }
  server.kill();

  // Find the recorded video in outDir
  const files = await fs.readdir(outDir);
  const videoFile = files.find(f => f.endsWith('.webm'));
  if (videoFile) {
    const src = path.join(outDir, videoFile);
    const dst = path.join(outDir, 'SIC_Demo_New_UI.webm');
    await fs.copyFile(src, dst);
    console.log(`Demo video recorded successfully -> ${dst}`);
  }

  console.log('All recording tasks finished!');
})().catch(err => {
  console.error('Recording error:', err);
  process.exit(1);
});
