import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const tid = process.argv[2];
assert.match(tid || '', /^[0-9a-f]{32}$/);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const base = 'http://127.0.0.1:8765';
const response = await fetch(`${base}/api/tasks/${tid}`);
assert.ok(response.ok);
const task = await response.json();
assert.equal(task.status, 'completed', task.error || 'Task must finish before verification');
assert.ok(task.report.verified);
const out = path.join(root, 'data', 'verification');
fs.mkdirSync(out, { recursive: true });
const profile = fs.mkdtempSync(path.join(root, 'cache', 'live-task-browser-'));
const browser = await chromium.launchPersistentContext(profile, {
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  viewport: { width: 1600, height: 1000 }, acceptDownloads: true,
  downloadsPath: path.join(root, 'tmp'),
  args: ['--no-first-run', '--disable-extensions', '--disable-background-networking',
    '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
});
try {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: /资产库/ }).click();
  await page.getByRole('textbox', { name: '搜索资产' }).fill(task.title);
  await page.locator('.library-grid .asset-card').first().waitFor();
  assert.equal(await page.locator('.library-grid .asset-card').count(), 1);
  const glbLoaded = page.waitForResponse(r => r.url().includes(`/api/tasks/${tid}/files/glb`) && r.ok());
  await page.locator('.library-grid .asset-card').first().click();
  await glbLoaded;
  await page.locator('.viewer-notice').waitFor({ state: 'hidden' });
  await page.waitForFunction(title => document.querySelector('.model-caption')?.textContent.includes(title), task.title);
  await page.waitForTimeout(1500);
  await page.screenshot({ path: path.join(out, `live-${tid}.png`), fullPage: true });
  const downloading = page.waitForEvent('download', { timeout: 30000 });
  await page.getByRole('link', { name: '下载资产包' }).click();
  const download = await downloading;
  const destination = path.join(out, `live-${tid}.zip`);
  await download.saveAs(destination);
  assert.equal(await download.failure(), null);
  const sha = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  assert.equal(sha(destination), sha(path.join(root, 'data', 'assets', tid, 'asset.zip')));
  assert.deepEqual(errors, []);
  const result = { task_id: tid, title: task.title, status: task.status, model: task.model,
    parts: task.part_count, triangles: task.report.triangles, materials: task.report.materials,
    elapsed_seconds: task.elapsed_seconds, formats: task.formats, independently_verified: task.report.verified,
    browser_errors: errors, viewer_loaded: true, download_matches_source: true,
    download_bytes: fs.statSync(destination).size };
  fs.writeFileSync(path.join(out, `live-${tid}.json`), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
} finally {
  await browser.close();
}
