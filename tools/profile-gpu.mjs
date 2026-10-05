import { existsSync } from 'node:fs';
import { createRequire } from 'node:module';
import { setTimeout as sleep } from 'node:timers/promises';

/**
 * Per-pass GPU profile.
 *
 * Reads the renderer's own timestamp breakdown, which brackets every pass in the
 * frame. The point is attribution: the totals say whether the frame is fast
 * enough, and only the breakdown says what to change.
 *
 * Usage:
 *   node tools/profile-gpu.mjs                       # shipped defaults
 *   node tools/profile-gpu.mjs --set='{"dof":0}'     # with an override
 *   node tools/profile-gpu.mjs --view=1280x720       # smaller viewport
 */
const require = createRequire(import.meta.url);
const CHROME = [
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
].find((p) => existsSync(p));

const args = new Map(
  process.argv.slice(2).map((a) => {
    const [k, ...rest] = a.replace(/^--/, '').split('=');
    return [k, rest.join('=') || 'true'];
  })
);
const URL_UNDER_TEST = args.get('url') ?? 'http://127.0.0.1:4173/';
const [VW, VH] = (args.get('view') ?? '1920x1080').split('x').map(Number);
let OVERRIDE = args.has('set') ? JSON.parse(args.get('set')) : null;

const puppeteer = require('puppeteer-core');
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--headless=new', '--enable-unsafe-webgpu', '--no-sandbox', '--disable-dev-shm-usage', '--mute-audio'],
  defaultViewport: { width: VW, height: VH },
  protocolTimeout: 300_000,
});

/**
 * One page load per case.
 *
 * Measuring several ablations in one session produced identical numbers for every
 * case, because the renderer samples its timers only every sixth frame and the
 * ablation was applied after the previous sample had already been collected. A
 * fresh load per case removes the ordering question entirely, which matters more
 * than the extra seconds it costs.
 */
const page = await browser.newPage();
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));

/**
 * Load the app fresh and settle it.
 *
 * `mode` selects an ablation; an empty string is the baseline. A fresh load per
 * case is deliberate: the renderer samples its timers only every sixth frame, so
 * applying an ablation mid-session and reading shortly afterwards returns the
 * previous configuration's numbers.
 */
async function load(mode) {
  await page.goto(`${URL_UNDER_TEST}${URL_UNDER_TEST.includes('?') ? '&' : '?'}selftest=1`, {
    waitUntil: 'load',
    timeout: 60_000,
  });
  // Let the device settle and the adaptive controller reach steady state, so the
  // numbers describe a working frame rather than the warm-up.
  await sleep(14000);
  // Hold quality still unless the caller explicitly wants the settled state.
  if (!args.has('nofreeze')) {
    await page.evaluate(() => {
      window.__AETHERIA_PARAMS.resolution = 1;
      window.__AETHERIA_FREEZE?.(true);
    });
    await sleep(1200);
  }
  if (OVERRIDE) {
    await page.evaluate((o) => {
      Object.assign(window.__AETHERIA_PARAMS, o);
    }, OVERRIDE);
    await sleep(5000);
  }
  if (mode) {
    await page.evaluate((m) => window.__AETHERIA_ABLATE?.(m), mode);
    await sleep(2500);
  }
}

/** Ablations to walk through, so each suspect is measured rather than argued. */
const ABLATIONS = args.has('ablate') ? args.get('ablate').split(',') : [''];
const RESULTS = [];

/**
 * Optional sweep: `--sweep=key=v1,v2,v3` applies one parameter at a time and
 * reports scaling. Attribution says which pass costs; a sweep says how the cost
 * responds to the thing being changed, which is what separates a linear
 * per-particle cost from a fixed one.
 */
const SWEEP = args.has('sweep') ? args.get('sweep').split('=') : null;

async function sampleOnce(mode) {
  await load(mode);
  let timings = [];
  for (let attempt = 0; attempt < 16; attempt++) {
    timings = await page.evaluate(() => window.__AETHERIA_PASSES?.() ?? []);
    if (timings.length) break;
    await sleep(500);
  }
  const stats = await page.evaluate(() => {
    const s = window.__AETHERIA_STATS?.();
    return s
      ? { fps: Math.round(s.fps), gpuMs: s.gpuMs, scale: s.renderScale, sim: s.simCount, shown: s.renderCount, w: s.drawWidth, h: s.drawHeight, vram: s.memoryEstimateMB }
      : null;
  });
  const total = timings.reduce((a, t) => a + t.ms, 0);
  const particles = timings.filter((t) => t.label === 'particles').reduce((a, t) => a + t.ms, 0);
  return { mode: mode || 'normal', particles, total, stats, timings };
}

if (SWEEP) {
  const [key, values] = SWEEP;
  console.log(`sweep ${key}: ${values}`);
  console.log(`${key.padStart(12)} ${'drawn'.padStart(10)} ${'particles ms'.padStart(13)} ${'ns/particle'.padStart(12)} ${'GPU total'.padStart(10)} ${'fps'.padStart(5)}`);
  console.log('-'.repeat(66));
  // The sweep overrides a parameter and then reloads, so the baseline has to be
  // captured from a loaded page and re-applied inside each fresh load.
  await load('');
  const baseline = { ...(await page.evaluate(() => ({ ...window.__AETHERIA_PARAMS }))) };
  for (const raw of values.split(',')) {
    const v = Number(raw);
    OVERRIDE = { [key]: v, ...(key === 'simCount' ? { showCount: v } : {}) };
    const r = await sampleOnce('');
    const drawn = r.stats?.shown ?? 0;
    // Cost per individual particle, which is the number that says whether the
    // pass is dominated by per-particle work or by something fixed.
    const perParticle = drawn > 0 ? (r.particles * 1e6) / drawn : 0;
    console.log(
      `${String(v).padStart(12)} ${(drawn / 1e6).toFixed(2).padStart(9)}M ${r.particles.toFixed(2).padStart(13)} ${perParticle.toFixed(1).padStart(12)} ${r.total.toFixed(2).padStart(10)} ${String(r.stats?.fps ?? '?').padStart(5)}`
    );
  }
  await page.evaluate((b) => Object.assign(window.__AETHERIA_PARAMS, b), baseline);
} else {
  for (const mode of ABLATIONS) RESULTS.push(await sampleOnce(mode));

  const stats = RESULTS[RESULTS.length - 1].stats;
  console.log(`viewport ${VW}x${VH}   draw buffer ${stats?.w}x${stats?.h}   scale ${stats ? (stats.scale * 100).toFixed(0) : '?'}%`);
  console.log(`particles ${stats ? (stats.sim / 1e6).toFixed(2) : '?'}M simulated / ${stats ? (stats.shown / 1e6).toFixed(2) : '?'}M drawn   vram ${stats ? stats.vram.toFixed(0) : '?'} MB`);
  console.log('');
  console.log(`${'ablation'.padEnd(12)} ${'particles'.padStart(10)} ${'later'.padStart(8)} ${'GPU total'.padStart(10)} ${'fps'.padStart(5)}`);
  console.log('-'.repeat(50));
  for (const r of RESULTS) {
    console.log(
      `${r.mode.padEnd(12)} ${r.particles.toFixed(2).padStart(10)} ${(r.total - r.particles).toFixed(2).padStart(8)} ${r.total.toFixed(2).padStart(10)} ${String(r.stats?.fps ?? '?').padStart(5)}`
    );
  }

  const worst = RESULTS.find((r) => r.mode === 'normal') ?? RESULTS[0];
  if (worst.timings.length && args.has('detail')) {
    console.log('\nper-pass detail (normal):');
    const total = worst.timings.reduce((a, t) => a + t.ms, 0);
    for (const t of [...worst.timings].sort((a, b) => b.ms - a.ms)) {
      console.log(`  ${t.label.padEnd(20)} ${t.ms.toFixed(2).padStart(8)}  ${((t.ms / total) * 100).toFixed(1).padStart(5)}%`);
    }
  }
}

await browser.close();
