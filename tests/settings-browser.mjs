import { chromium } from '../frontend/node_modules/playwright/index.mjs';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'data', 'verification');
fs.mkdirSync(out, { recursive: true });
const profile = fs.mkdtempSync(path.join(root, 'cache', 'settings-browser-'));
const browser = await chromium.launchPersistentContext(profile, {
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  headless: true, viewport: { width: 1440, height: 1000 },
  downloadsPath: path.join(root, 'tmp'),
  args: ['--no-first-run', '--disable-extensions', '--disable-background-networking',
    '--use-angle=swiftshader', '--enable-unsafe-swiftshader'],
});
const errors = [];
const checked = [];
try {
  const page = await browser.newPage();
  page.on('pageerror', error => errors.push(error.message));
  let offline = true, failSave = false, saved = null;
  // Abort requests to reproduce a stopped backend. Saving and cloud testing are
  // intercepted so no real credentials or user settings are modified.
  await page.route('**/api/**', async route => {
    const request = route.request();
    const pathname = new URL(request.url()).pathname;
    if (offline) return route.abort('connectionrefused');
    if (pathname === '/api/settings/test') {
      return route.fulfill({ status: 400, json: { detail: '模型接口返回 HTTP 401。请检查地址、模型名、额度和授权。' } });
    }
    if (pathname === '/api/settings' && request.method() === 'PUT') {
      if (failSave) return route.abort('connectionrefused');
      const { api_key, ...settings } = request.postDataJSON();
      assert.equal(api_key, 'browser-test-key');
      saved = { ...settings, has_api_key: true, connection_status: 'failed' };
      return route.fulfill({ json: saved });
    }
    if (pathname === '/api/settings' && saved) return route.fulfill({ json: saved });
    return route.continue();
  });
  await page.goto('http://127.0.0.1:8765/?settings=1');
  const dialog = page.getByRole('dialog');
  await dialog.getByText('暂时无法读取 API 设置', { exact: true }).waitFor();
  assert.equal(await dialog.locator('.spin').count(), 0);
  assert.ok((await dialog.innerText()).includes('Start-Studio.cmd'));
  await page.screenshot({ path: path.join(out, 'settings-offline.png') });
  checked.push('Stopped backend shows an actionable error without an endless spinner');

  offline = false;
  await dialog.getByRole('button', { name: '重新连接', exact: true }).click();
  await dialog.getByLabel('API 地址', { exact: true }).waitFor();
  await page.locator('.service-banner').waitFor({ state: 'hidden', timeout: 15000 });
  checked.push('Retry restores the real settings form and polling clears the offline banner');

  await dialog.getByRole('combobox').selectOption('openai-compatible');
  await dialog.getByLabel('API 地址', { exact: true }).fill('https://test.invalid/v1');
  await dialog.getByLabel('模型名称', { exact: true }).fill('browser-test-model');
  await dialog.getByLabel('API Key', { exact: true }).fill('browser-test-key');
  failSave = true;
  await dialog.getByRole('button', { name: '保存设置', exact: true }).click();
  await dialog.locator('.form-message').filter({ hasText: '无法连接本地服务' }).waitFor();
  assert.equal(await dialog.getByLabel('API Key', { exact: true }).inputValue(), 'browser-test-key');
  assert.equal(await dialog.getByLabel('模型名称', { exact: true }).inputValue(), 'browser-test-model');
  checked.push('A failed save preserves the entered settings for retry');

  failSave = false;
  await dialog.getByRole('button', { name: '保存并测试连接', exact: true }).click();
  await dialog.locator('.form-message').filter({ hasText: '设置已保存，但连接测试未通过：模型接口返回 HTTP 401' }).waitFor();
  assert.equal(await dialog.getByLabel('API Key', { exact: true }).inputValue(), '');
  checked.push('A simulated cloud 401 is distinguished from a successful local save');
  await page.getByRole('button', { name: '关闭设置', exact: true }).click();
  await page.getByRole('button', { name: '模型设置', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.settings-modal input')?.value === 'https://test.invalid/v1');
  assert.equal(await dialog.getByLabel('模型名称', { exact: true }).inputValue(), 'browser-test-model');
  assert.equal(await dialog.getByLabel('API Key', { exact: true }).inputValue(), '');
  assert.ok(!(await dialog.innerText()).includes('browser-test-key'));
  checked.push('Reopening settings displays saved values without returning the key');
  await page.screenshot({ path: path.join(out, 'settings-recovered.png') });
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(out, 'settings-browser-result.json'), JSON.stringify({ checked, errors, saveAndCloudResponses: 'intercepted; real user settings unchanged' }, null, 2));
  console.log(JSON.stringify({ checked, errors }));
} finally {
  await browser.close();
}
