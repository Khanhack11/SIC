/* Diagnose the running user server, without creating or stopping a server. */
const { chromium, webkit } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const url = process.env.SIC_DEMO_URL || 'http://127.0.0.1:8000/app';
const output = process.env.SIC_TEST_OUTPUT || path.join(root, 'tests/results/live_demo_20261103');

(async () => {
  await fs.mkdir(output, { recursive: true });
  const results = [];
  for (const name of ['chromium', 'webkit']) {
    const browser = name === 'chromium'
      ? await chromium.launch({ executablePath: process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true })
      : await webkit.launch({ headless: true });
    const page = await browser.newPage();
    const errors = [], failedRequests = [], responses = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => failedRequests.push({ path: new URL(request.url()).pathname, error: request.failure()?.errorText }));
    page.on('response', response => { if (response.status() >= 400) responses.push({ path: new URL(response.url()).pathname, status: response.status() }); });
    const record = { browser: name, origin: new URL(url).origin, passed: false, errors, failedRequests, httpErrors: responses };
    try {
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 15000 });
      await page.getByText('Model sẵn sàng', { exact: true }).waitFor({ timeout: 12000 });
      await page.locator('#file-input').setInputFiles(path.join(root, 'sample_images/demo/Corn_Common_rust.jpg'));
      await page.locator('#btn-submit:not([disabled])').waitFor({ timeout: 10000 });
      const responsePromise = page.waitForResponse(response => response.url().includes('/api/diagnose') && response.request().method() === 'POST', { timeout: 20000 });
      await page.locator('#btn-submit').click();
      const response = await responsePromise;
      assert.equal(response.status(), 200);
      await page.locator('#result-box:not([hidden])').waitFor({ timeout: 10000 });
      assert.equal(await page.locator('#input-error').textContent(), '');
      assert(!/load\s*failed|failed to fetch/i.test(await page.locator('#server-status').textContent()));
      assert.deepEqual(errors, []);
      assert.deepEqual(failedRequests, []);
      assert.deepEqual(responses, []);
      record.passed = true;
      record.result = await page.locator('#disease-name').textContent();
    } catch (error) {
      record.error = error.message;
      record.status = await page.locator('#server-status').textContent().catch(() => 'page unavailable');
    } finally {
      await page.screenshot({ path: path.join(output, `${name}.png`), fullPage: true }).catch(() => {});
      results.push(record);
      await browser.close();
    }
  }
  const report = { passed: results.every(result => result.passed), results };
  await fs.writeFile(path.join(output, 'verification.json'), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report));
  if (!report.passed) process.exitCode = 1;
})().catch(error => { console.error(error); process.exitCode = 1; });
