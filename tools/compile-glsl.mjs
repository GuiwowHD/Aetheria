import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { setTimeout as sleep } from 'node:timers/promises';

/**
 * Compile every WebGL2 shader in one page load and print the full info logs.
 *
 * The fallback compiles ten programs during `init()`, and the boot path stops at
 * the first failure, so iterating one error at a time costs a full build cycle
 * each. This drives `gl.compileShader` directly for every source and reports all
 * diagnostics at once.
 */
const require = createRequire(import.meta.url);
const CHROME = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
].find((p) => existsSync(p));

const puppeteer = require('puppeteer-core');
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--headless=new', '--enable-unsafe-webgpu', '--no-sandbox', '--disable-dev-shm-usage'],
  defaultViewport: { width: 800, height: 450 },
  protocolTimeout: 120_000,
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));

// The page must be on the app's origin for the module import to resolve. It is
// allowed to boot first: importing while the app is still starting up destroyed
// the execution context mid-evaluation.
await page.goto('http://127.0.0.1:5199/', { waitUntil: 'load', timeout: 60_000 });
await sleep(2500);

const report = await page.evaluate(async () => {
  // The shader sources are not importable by name from a built bundle, so the
  // probe re-imports the source module through Vite's dev server instead.
  let src;
  try {
    src = await import('/src/webgl2/shaders.ts');
  } catch (err) {
    return { error: String(err) };
  }
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  if (!gl) return { error: 'no webgl2 context' };

  const out = [];
  const compileOne = (label, type, source) => {
    const sh = gl.createShader(type);
    if (!sh) return out.push({ label, log: 'createShader null' });
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    const ok = gl.getShaderParameter(sh, gl.COMPILE_STATUS);
    const log = (gl.getShaderInfoLog(sh) || '').trim();
    if (!ok || log) out.push({ label, ok: !!ok, log });
    gl.deleteShader(sh);
  };

  for (const [name, value] of Object.entries(src)) {
    if (typeof value !== 'string') continue;
    if (!/^(SIM_|PARTICLE_|DEPTH_|FULLSCREEN_|TRAIL_|BRIGHT_|BLOOM_|VOLUMETRIC_|DOF_|COMPOSITE_|FXAA_)/.test(name)) continue;
    if (name.endsWith('_BLOCK_GLSL') || name === 'SIM_TF_VARYINGS') continue;
    const isVert = /_(VS|VERT)$/.test(name) || name.endsWith('_VS');
    compileOne(name, isVert ? gl.VERTEX_SHADER : gl.FRAGMENT_SHADER, value);
  }
  return { shaders: out, names: Object.keys(src).filter((k) => typeof src[k] === 'string') };
});

if (report.error) {
  console.log('probe failed:', report.error);
} else {
  console.log('module string exports:', report.names.join(', '));
  if (!report.shaders.length) console.log('ALL SHADERS COMPILED CLEAN');
  for (const s of report.shaders) {
    console.log(`\n=== ${s.label} ${s.ok ? '(compiled, warnings)' : '(FAILED)'} ===`);
    console.log(s.log.split('\n').slice(0, 14).join('\n'));
  }
}

void sleep;
await browser.close();
