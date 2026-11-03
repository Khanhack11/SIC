/* Independent rendering check. This does not replace Archify's own receipt. */
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

(async () => {
  const folder = path.resolve(process.argv[2]);
  const evidence = path.resolve('tests/results/architecture_diagrams');
  fs.mkdirSync(evidence, { recursive: true });
  const browser = await chromium.launch({ executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true });
  const results = [];
  try {
    for (const file of ['he-thong.html', 'huan-luyen.html', 'suy-luan.html']) {
      for (const width of [1440, 1920]) {
        const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.goto(pathToFileURL(path.join(folder, file)).href);
        await page.locator('svg').first().waitFor();
        await page.evaluate(() => document.fonts.ready);
        const result = await page.evaluate(() => ({
          title: document.title,
          horizontalOverflow: document.documentElement.scrollWidth > innerWidth + 1,
          diagramCount: document.querySelectorAll('svg').length,
          labels: [...document.querySelectorAll('svg text')].map(el => el.textContent),
        }));
        await page.screenshot({ path: path.join(evidence, `${file}-${width}.png`), fullPage: true });
        results.push({ file, width, ...result, errors });
        if (result.horizontalOverflow || errors.length || !result.labels.length) throw Error(JSON.stringify(results.at(-1)));
        await page.close();
      }
    }
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(evidence, 'render-check.json'), JSON.stringify({
      scope: 'Independent Playwright Chrome rendering only; Archify browser-check failed at process cleanup.', results,
    }, null, 2));
  }
  console.log(`Rendered ${results.length} diagram/viewports without horizontal overflow or JavaScript errors.`);
})().catch(error => { console.error(error); process.exitCode = 1; });
