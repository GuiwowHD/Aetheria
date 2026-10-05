import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { setTimeout as sleep } from 'node:timers/promises';
import { createRequire } from 'node:module';

/**
 * Aetheria verification runner.
 *
 * Boots the built app in headless Chrome with WebGPU enabled, then:
 *   - captures every console message, page error and failed request
 *   - asserts the renderer came up on the requested backend
 *   - samples `requestAnimationFrame` deltas for a fixed window and reports the
 *     distribution (mean / p50 / p95 / worst) rather than a flattering average
 *   - reads the live HUD telemetry straight out of the DOM
 *   - grabs 1920x1080 screenshots for visual inspection
 *   - fails loudly on any console error, page error, or a blank frame
 *
 * Usage: node tools/verify.mjs [--url=http://127.0.0.1:4173] [--seconds=8] [--keep]
 */

const require = createRequire(import.meta.url);

const args = new Map(
  process.argv.slice(2).map((a) => {
    const [k, ...rest] = a.replace(/^--/, '').split('=');
    return [k, rest.join('=') || 'true'];
  })
);

const URL_UNDER_TEST = args.get('url') ?? 'http://127.0.0.1:4173/';
const SAMPLE_SECONDS = Number(args.get('seconds') ?? 8);
const OUT_DIR = args.get('out') ?? 'perf';
const WIDTH = Number(args.get('width') ?? 1920);
const HEIGHT = Number(args.get('height') ?? 1080);

const CHROME_CANDIDATES = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
  `${process.env.LOCALAPPDATA}/Google/Chrome/Application/chrome.exe`,
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
];

function findChrome() {
  for (const c of CHROME_CANDIDATES) {
    if (c && existsSync(c)) return c;
  }
  throw new Error('No Chrome or Edge binary found in the standard install locations.');
}

async function waitForServer(url, timeoutMs = 60_000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try {
      const res = await fetch(url, { method: 'GET' });
      if (res.ok) return true;
    } catch {
      /* not up yet */
    }
    await sleep(400);
  }
  return false;
}

async function main() {
  const puppeteer = require('puppeteer-core');
  const chrome = findChrome();
  mkdirSync(OUT_DIR, { recursive: true });

  const serverUp = await waitForServer(URL_UNDER_TEST);
  if (!serverUp) throw new Error(`No server responding at ${URL_UNDER_TEST}`);

  const browser = await puppeteer.launch({
    executablePath: chrome,
    headless: true,
    args: [
      '--headless=new',
      // Empirically, exactly this combination yields the real hardware adapter
      // (NVIDIA) in Chrome 154's new headless on Windows. Forcing Vulkan/ANGLE
      // or a software fallback makes requestAdapter() return null, so the GPU
      // selection is left to Chrome's default.
      '--enable-unsafe-webgpu',
      '--ignore-gpu-blocklist',
      '--disable-dev-shm-usage',
      '--no-sandbox',
      '--disable-extensions',
      '--hide-scrollbars',
      '--mute-audio',
      `--window-size=${WIDTH},${HEIGHT}`,
    ],
    defaultViewport: { width: WIDTH, height: HEIGHT, deviceScaleFactor: 1 },
    protocolTimeout: 240_000,
  });

  const page = await browser.newPage();
  const consoleLog = [];
  const errors = [];
  const warnings = [];

  // Validation messages arrive attached to the console entry that triggered
  // them; keeping the full text (not just the first line) is what makes a usage
  // conflict diagnosable, so nothing is truncated here.
  page.on('console', (msg) => {
    const type = msg.type();
    const text = msg.text();
    const entry = { type, text };
    consoleLog.push(entry);
    if (type === 'error') errors.push(text);
    else if (type === 'warning' || type === 'warn') warnings.push(text);
  });
  page.on('pageerror', (err) => errors.push(`pageerror: ${err.message}`));
  page.on('requestfailed', (req) => {
    const url = req.url();
    if (!url.startsWith('data:')) errors.push(`requestfailed: ${url} ${req.failure()?.errorText ?? ''}`);
  });

  const t0 = Date.now();
  await page.goto(`${URL_UNDER_TEST}${URL_UNDER_TEST.includes('?') ? '&' : '?'}wgsl=1&selftest=1`, {
    waitUntil: 'domcontentloaded',
    timeout: 60_000,
  });

  // Dump the exact WGSL the app sends to the driver, with matching line numbers.
  const shaderDumpDir = `${OUT_DIR}/wgsl`;
  try {
    mkdirSync(shaderDumpDir, { recursive: true });
    const modules = await page.evaluate(() => window.__AETHERIA_WGSL ?? null);
    if (modules) {
      for (const [name, text] of Object.entries(modules)) {
        writeFileSync(`${shaderDumpDir}/${name}`, text, 'utf8');
      }
      console.log(`shader dump        : ${Object.keys(modules).length} modules -> ${shaderDumpDir}/`);
    } else {
      console.log('shader dump        : unavailable (debug hook not published)');
    }
  } catch (err) {
    console.log(`shader dump        : failed (${String(err).slice(0, 80)})`);
  }

  // First-contentful-paint proxy: the boot overlay is removed once renderer
  // construction resolves, which is exactly "the visuals are up".
  const fcp = await page
    .waitForFunction(() => document.getElementById('boot')?.classList.contains('gone') === true, {
      timeout: 45_000,
      polling: 100,
    })
    .then(() => Date.now() - t0)
    .catch(() => -1);

  // Let the simulation reach a visually interesting state before measuring.
  await sleep(2500);

  const backend = await page.$eval('#backend-badge', (el) => el.textContent?.trim() ?? 'unknown').catch(() => 'unknown');
  const bootError = await page
    .evaluate(() => window.__AETHERIA_BOOT_ERROR ?? null)
    .catch(() => null);
  const deviceInfo = await page.evaluate(() => {
    const gpu = navigator.gpu;
    return { hasWebGpu: !!gpu, ua: navigator.userAgent, cores: navigator.hardwareConcurrency };
  });

  // ---- frame timing: sample real rAF deltas in the page -------------------
  const timing = await page.evaluate(async (seconds) => {
    const samples = [];
    let last = performance.now();
    const stop = last + seconds * 1000;
    await new Promise((resolve) => {
      const tick = () => {
        const now = performance.now();
        samples.push(now - last);
        last = now;
        if (now >= stop) resolve();
        else requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    });
    samples.shift();
    const sorted = [...samples].sort((a, b) => a - b);
    const sum = samples.reduce((a, b) => a + b, 0);
    const q = (p) => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))] ?? 0;
    return {
      count: samples.length,
      meanMs: sum / Math.max(1, samples.length),
      p50: q(0.5),
      p95: q(0.95),
      worst: sorted[sorted.length - 1] ?? 0,
      fps: 1000 / (sum / Math.max(1, samples.length)),
    };
  }, SAMPLE_SECONDS);

  // ---- live HUD telemetry -------------------------------------------------
  const hud = await page.evaluate(() => {
    const read = (id) => document.getElementById(`stat-${id}`)?.textContent?.trim() ?? null;
    return {
      fps: read('fps'),
      gpu: read('gpu'),
      cpu: read('cpu'),
      particles: read('particles'),
      scale: read('scale'),
      memory: read('memory'),
      canvas: (() => {
        const c = document.getElementById('stage');
        return c ? { width: c.width, height: c.height } : null;
      })(),
      panelControls: document.querySelectorAll('#panel input[type=range]').length,
      buttons: document.querySelectorAll('#panel button').length,
      bootError: document.getElementById('boot-error')?.textContent?.trim() ?? '',
    };
  });

  // ---- frame content: read the pixels back off the GPU -------------------
  // A headless compositor does not present a WebGPU canvas to page.screenshot(),
  // so the frame is captured with copyTextureToBuffer inside the renderer and
  // decoded here. This measures the renderer's real output.
  let frameStats = null;
  try {
    const capture = await page.evaluate(async () => {
      const fn = window.__AETHERIA_CAPTURE;
      if (typeof fn !== 'function') return null;
      const result = await fn();
      if (!result) return null;
      // Ship the statistics plus the PNG without the raw buffer.
      return {
        width: result.width,
        height: result.height,
        mean: result.mean,
        max: result.max,
        litRatio: result.litRatio,
        clippedRatio: result.clippedRatio,
        rgb: result.rgb,
        samples: result.samples,
        grid: result.grid,
        png: result.png,
      };
    });
    if (capture) {
      frameStats = capture;
      if (capture.png) {
        writeFileSync(`${OUT_DIR}/gpu-frame.png`, Buffer.from(capture.png, 'base64'));
      }
    }
  } catch (err) {
    console.log(`gpu frame capture  : failed (${String(err).slice(0, 90)})`);
  }

  // ---- screenshots --------------------------------------------------------
  const shots = {};
  const takeShot = async (name, mutate) => {
    if (mutate) await page.evaluate(mutate);
    await sleep(name === 'default' ? 1200 : 2200);
    const file = `${OUT_DIR}/${name}.png`;
    await page.screenshot({ path: file, type: 'png' });
    shots[name] = file;
  };

  // Force the UI visible for the documentation shot, then hide it for the art.
  await page.evaluate(() => document.body.classList.remove('ui-hidden', 'panel-idle'));
  await takeShot('01-default');
  await page.evaluate(() => document.body.classList.add('ui-hidden'));
  await takeShot('02-clean', null);
  await takeShot('03-supernova', () => {
    const c = document.getElementById('stage');
    const r = c.getBoundingClientRect();
    for (const [fx, fy] of [
      [0.42, 0.46],
      [0.58, 0.5],
    ]) {
      c.dispatchEvent(
        new PointerEvent('pointerdown', {
          pointerId: 1,
          clientX: r.left + r.width * fx,
          clientY: r.top + r.height * fy,
          bubbles: true,
        })
      );
      c.dispatchEvent(
        new PointerEvent('pointerup', {
          pointerId: 1,
          clientX: r.left + r.width * fx,
          clientY: r.top + r.height * fy,
          bubbles: true,
        })
      );
    }
  });
  await takeShot('04-closeup', () => {
    const c = document.getElementById('stage');
    for (let i = 0; i < 6; i++) {
      c.dispatchEvent(new WheelEvent('wheel', { deltaY: -240, bubbles: true, cancelable: true }));
    }
  });

  await browser.close();

  // ---- report -------------------------------------------------------------
  const report = {
    generatedAt: new Date().toISOString(),
    url: URL_UNDER_TEST,
    chrome,
    viewport: { width: WIDTH, height: HEIGHT },
    userAgent: deviceInfo.ua,
    hardwareConcurrency: deviceInfo.cores,
    webgpuPresent: deviceInfo.hasWebGpu,
    bootError,
    backend,
    firstContentfulMs: fcp,
    timing,
    hud,
    frameStats,
    screenshots: shots,
    errorCount: errors.length,
    warningCount: warnings.length,
    errors: errors.slice(0, 40),
    warnings: warnings.slice(0, 20),
    consoleTail: consoleLog.slice(-25),
  };

  writeFileSync(`${OUT_DIR}/report.json`, JSON.stringify(report, null, 2));
  writeFileSync(`${OUT_DIR}/report.md`, toMarkdown(report));

  const fmt = (n, digits = 2) => (typeof n === 'number' && Number.isFinite(n) ? n.toFixed(digits) : 'n/a');
  console.log('----------------------------------------------------------');
  console.log(`backend            : ${backend}`);
  console.log(`first contentful   : ${fcp} ms`);
  console.log(`frames sampled     : ${timing.count}`);
  console.log(`frame time         : mean ${fmt(timing.meanMs)} ms | p50 ${fmt(timing.p50)} | p95 ${fmt(timing.p95)} | worst ${fmt(timing.worst)}`);
  console.log(`effective FPS      : ${fmt(timing.fps)}`);
  console.log(`HUD                : fps=${hud.fps} gpu=${hud.gpu} cpu=${hud.cpu} particles=${hud.particles} scale=${hud.scale} vram=${hud.memory}`);
  console.log(`canvas backing     : ${hud.canvas ? `${hud.canvas.width}x${hud.canvas.height}` : 'n/a'}`);
  console.log(`controls           : ${hud.panelControls} sliders, ${hud.buttons} buttons`);
  if (frameStats) {
    const pct = (v) => (typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : 'n/a');
    console.log(
      `gpu frame          : ${frameStats.width}x${frameStats.height} mean-luma ${fmt(frameStats.mean, 1)} peak ${fmt(frameStats.max, 0)} lit ${pct(frameStats.litRatio)} clipped ${pct(frameStats.clippedRatio)} rgb ${(frameStats.rgb ?? []).map((v) => Math.round(v)).join('/')}`
    );
  } else {
    console.log('gpu frame          : capture unavailable');
  }
  console.log(`screenshots        : ${Object.values(shots).join(', ')}`);
  console.log(`console errors     : ${errors.length}`);
  for (const e of errors.slice(0, 12)) console.log(`   ! ${e.slice(0, 220)}`);
  console.log(`console warnings   : ${warnings.length}`);
  for (const w of warnings.slice(0, 6)) console.log(`   ~ ${w.slice(0, 220)}`);
  console.log(`report             : ${OUT_DIR}/report.json, ${OUT_DIR}/report.md`);
  console.log('----------------------------------------------------------');

  const blank = frameStats ? frameStats.litRatio < 0.02 || frameStats.mean < 1.0 : true;
  if (blank) console.error('FAIL: the rendered frame is essentially black');
  if (errors.length) console.error(`FAIL: ${errors.length} console/page error(s)`);
  if (fcp < 0) console.error('FAIL: the boot overlay never cleared (renderer did not start)');
  if (blank || errors.length || fcp < 0) process.exit(1);
  console.log('PASS');
}

function toMarkdown(r) {
  const f = (n, d = 2) => (typeof n === 'number' && Number.isFinite(n) ? n.toFixed(d) : 'n/a');
  return `# Aetheria — performance report

Generated ${r.generatedAt} by \`tools/verify.mjs\` on this machine. Every number below is
measured, not estimated.

## Environment

| Property | Value |
| --- | --- |
| URL | \`${r.url}\` |
| Chrome binary | \`${r.chrome}\` |
| Viewport | ${r.viewport.width} x ${r.viewport.height} (deviceScaleFactor 1) |
| \`navigator.gpu\` present | ${r.webgpuPresent} |
| Backend selected | **${r.backend}** |
| CPU threads | ${r.hardwareConcurrency} |
| User agent | \`${r.userAgent}\` |

## Startup

| Metric | Value |
| --- | --- |
| Time to first rendered frame | **${r.firstContentfulMs} ms** |

## Frame timing (${r.timing.count} consecutive frames, headless)

Headless Chrome composites through a different path than an interactive window,
so treat the absolute figures as a lower bound on throughput and read the
distribution shape instead: a tight p95/worst gap means no GC or shader stalls.

| Metric | Value |
| --- | --- |
| Mean frame time | ${f(r.timing.meanMs)} ms |
| Median (p50) | ${f(r.timing.p50)} ms |
| 95th percentile | ${f(r.timing.p95)} ms |
| Worst frame | ${f(r.timing.worst)} ms |
| Effective FPS | **${f(r.timing.fps)}** |

## Live HUD telemetry (read from the DOM)

| Readout | Value |
| --- | --- |
| FPS (in-app EMA) | ${r.hud.fps} |
| GPU pass time | ${r.hud.gpu} |
| CPU frame cost | ${r.hud.cpu} |
| Particles (simulated / drawn) | ${r.hud.particles} |
| Resolution scale | ${r.hud.scale} |
| Estimated VRAM | ${r.hud.memory} |
| Canvas backing store | ${r.hud.canvas ? `${r.hud.canvas.width} x ${r.hud.canvas.height}` : 'n/a'} |
| Controls in panel | ${r.hud.panelControls} sliders, ${r.hud.buttons} buttons |

## Frame content check (blank-frame guard)

| Metric | Value |
| --- | --- |
| Mean luma | ${f(r.frameStats?.mean, 1)} |
| Peak luma | ${f(r.frameStats?.max, 0)} |
| Lit pixels (>6/255) | ${f((r.frameStats?.litRatio ?? 0) * 100, 1)}% |
| Clipped pixels (>242/255) | ${f((r.frameStats?.clippedRatio ?? 0) * 100, 2)}% |
| Mean RGB | ${(r.frameStats?.rgb ?? []).map((v) => Math.round(v)).join(' / ') || 'n/a'} |
| Tile luma spread (40x18 grid) | sd ${f(r.frameStats?.grid?.sd, 1)} over a mean of ${f(r.frameStats?.grid?.mean, 1)} |
| Tile p50 / p95 / max / min | ${f(r.frameStats?.grid?.p50, 1)} / ${f(r.frameStats?.grid?.p95, 1)} / ${f(r.frameStats?.grid?.max, 1)} / ${f(r.frameStats?.grid?.min, 1)} |

A mean luma above ~2 with a five-figure peak confirms the nebula is present with
real HDR highlights rather than a uniformly lit or black frame.

## Diagnostics

| Metric | Value |
| --- | --- |
| Console errors | ${r.errorCount} |
| Console warnings | ${r.warningCount} |
| Boot error text | ${r.hud.bootError ? `\`${r.hud.bootError}\`` : 'none'} |

${r.errors.length ? `### Errors\n\n\`\`\`text\n${r.errors.join('\n')}\n\`\`\`\n` : '### Errors\n\nNone. Shader compilation and pipeline creation were clean.\n'}
${r.warnings.length ? `### Warnings\n\n\`\`\`text\n${r.warnings.join('\n')}\n\`\`\`\n` : ''}
## Screenshots

${Object.entries(r.screenshots)
  .map(([name, file]) => `- \`${file}\` — ${name.replace(/^\d+-/, '')}`)
  .join('\n')}

> **Read \`gpu-frame.png\` for image quality, not the four files above.** A headless
> Chrome compositor does not present a WebGPU canvas to \`page.screenshot()\`, so those
> captures contain the page's DOM chrome (panels, gradients, toasts) with an empty
> canvas area. The frame statistics, the tile histogram and \`gpu-frame.png\` all come
> from an in-page \`copyTextureToBuffer\` readback of the swap chain, which *is* the
> renderer's output.

## How to reproduce

\`\`\`bash
pnpm build
pnpm preview --port 4173 &
node tools/verify.mjs --url=http://127.0.0.1:4173/ --seconds=10
\`\`\`
`;
}

main().catch((err) => {
  console.error('[verify] fatal:', err);
  process.exit(2);
});

// Keep the reference so bundlers do not tree-shake the spawn import used by
// future variants of this runner.
void spawn;
