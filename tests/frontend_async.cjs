/* Deterministic camera-encoding race, using real frontend code and a simulated camera. */
const { chromium } = require('playwright');
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
    headless: true, args: ['--disable-gpu', '--use-fake-device-for-media-stream'],
  });
  const checks = [];
  try {
    const page = await browser.newPage();
    await page.route('http://localhost:8005/**', async route => {
      const pathname = new URL(route.request().url()).pathname;
      if (pathname === '/api/health') return route.fulfill({ json: { status: 'HEALTHY', weights_loaded: true } });
      if (pathname === '/demo/manifest.json') return route.fulfill({ json: { images: [] } });
      const file = { '/app': 'frontend_demo.html', '/frontend.js': 'frontend.js', '/crop.js': 'crop.js' }[pathname];
      if (file) return route.fulfill({ body: await fs.readFile(path.join(root, 'frontend', file)), contentType: file.endsWith('.js') ? 'text/javascript' : 'text/html' });
      return route.fulfill({ status: 404 });
    });
    await page.context().grantPermissions(['camera'], { origin: 'http://localhost:8005' });
    await page.goto('http://localhost:8005/app');
    await page.getByText('Model sẵn sàng', { exact: true }).waitFor();
    await page.locator('#file-input').setInputFiles(path.join(root, 'sample_images/demo/Potato_healthy.jpg'));
    await page.waitForFunction(() => document.querySelector('#file-info').textContent.includes('Potato_healthy.jpg'));

    for (const action of ['new file', 'close camera', 'close options']) {
      if (!await page.locator('#camera-options').evaluate(details=>details.open)) await page.locator('#camera-options > summary').click();
      await page.locator('#open-camera').click();
      await page.waitForFunction(() => document.querySelector('#camera-video').videoWidth > 0);
      await page.evaluate(() => {
        const native = HTMLCanvasElement.prototype.toBlob;
        HTMLCanvasElement.prototype.toBlob = function (callback, ...args) {
          HTMLCanvasElement.prototype.toBlob = native;
          native.call(this, blob => { window.releaseCameraBlob = () => callback(blob); }, ...args);
        };
        window.releaseCameraBlob = null;
        window.captureTracks = cameraStream.getTracks();
        window.captureJob = captureImage();
      });
      await page.waitForFunction(() => window.releaseCameraBlob !== null);
      if (action === 'new file') {
        await page.locator('#file-input').setInputFiles(path.join(root, 'sample_images/demo/Tomato_Early_blight.jpg'));
        await page.waitForFunction(() => document.querySelector('#file-info').textContent.includes('Tomato_Early_blight.jpg'));
      } else if (action === 'close camera') {
        await page.locator('#close-camera').click();
      } else {
        await page.locator('#camera-options > summary').click();
        await page.locator('#camera-panel').waitFor({state:'hidden'});
      }
      const expected = await page.locator('#file-info').textContent();
      await page.evaluate(() => window.releaseCameraBlob());
      await page.evaluate(() => window.captureJob);
      assert.equal(await page.locator('#file-info').textContent(), expected,
        `A slow capture replaced the image after: ${action}`);
      assert(await page.locator('#camera-panel').isHidden());
      assert(await page.evaluate(()=>window.captureTracks.every(track=>track.readyState==='ended')));
      checks.push(action);
    }
    // A crop touching the right edge of an odd-width image must not add a black column.
    const fixture = await page.evaluate(() => {
      const canvas = document.createElement('canvas'); canvas.width = 95; canvas.height = 97;
      const context = canvas.getContext('2d'); context.fillStyle = 'rgb(70,170,80)';
      context.fillRect(0, 0, 95, 97); return canvas.toDataURL('image/png').split(',')[1];
    });
    await page.locator('#file-input').setInputFiles({ name: 'odd-width.png', mimeType: 'image/png', buffer: Buffer.from(fixture, 'base64') });
    await page.waitForFunction(() => document.querySelector('#file-info').textContent.includes('odd-width.png'));
    await page.locator('#crop-editor').evaluate(details => { details.open = true; });
    await page.locator('#crop-x').fill('50');
    await page.locator('#crop-width').fill('50');
    await page.locator('#apply-crop').click();
    await page.waitForFunction(() => document.querySelector('#file-info').textContent.includes('leaf_crop.jpg'));
    assert.equal(await page.locator('#image-preview').evaluate(image => image.naturalWidth), 47, 'Crop extended beyond the 95px source');
    assert(await page.locator('#image-preview').evaluate(image => {
      const canvas = document.createElement('canvas'); canvas.width = image.naturalWidth; canvas.height = image.naturalHeight;
      const context = canvas.getContext('2d'); context.drawImage(image, 0, 0);
      return context.getImageData(canvas.width - 1, 40, 1, 1).data[1] > 100;
    }), 'Crop introduced a dark edge outside the source image');
    checks.push('odd-width crop preserves source boundary');
    await page.route('**/api/health', route => route.fulfill({ json: {
      status: 'HEALTHY', weights_loaded: true, model_architecture: 'MobileNetV3-Large field (224x224 RGB letterbox)',
    } }));
    await page.evaluate(() => checkServerHealth());
    assert.match(await page.locator('#model-scope').textContent(), /V3 thử nghiệm/);
    await page.setViewportSize({ width: 390, height: 844 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    const output = process.env.SIC_TEST_OUTPUT || path.join(root, 'tests/results/frontend_async');
    await fs.mkdir(output, { recursive: true });
    await page.screenshot({ path: path.join(output, 'experimental-v3-mobile.png'), fullPage: true });
    await fs.writeFile(path.join(output, 'frontend_async.json'), JSON.stringify({ passed: true, checks, experimental_model_notice: true, camera: 'simulated', backend: 'mock health; no model inference needed for this race' }, null, 2));
    console.log('Camera encoding race checks passed:', checks.join(', '));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
