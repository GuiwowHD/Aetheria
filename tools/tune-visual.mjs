import { existsSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { setTimeout as sleep } from 'node:timers/promises';

/**
 * Visual calibration loop.
 *
 * Renders a candidate parameter set through the real GPU pipeline, reads the
 * frame back off the swap chain, and prints the numbers that distinguish a
 * nebula from a smear: peak luma (are there real highlights), tile standard
 * deviation (is there structure), the lit fraction, and how centrally the light
 * is concentrated. A coarse ASCII luma map is printed too, because a shape is
 * easier to judge from 40x18 characters than from a histogram.
 *
 * Usage:
 *   node tools/tune-visual.mjs                 # uses the built-in candidate list
 *   node tools/tune-visual.mjs --shot=name     # also save perf/tune-<name>.png
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
const SAVE = args.has('shot');

/** Baseline: everything the candidate does not override. */
const BASE = {
  simCount: 1_000_000,
  showCount: 2_000_000,
  speed: 1,
  fractalDim: 0.7,
  curl: 0.85,
  damping: 0.3,
  gravity: 0.13,
  confinement: 0.85,
  shock: 1,
  exposure: 1.0,
  bloom: 0.62,
  bloomRadius: 0.55,
  dof: 0.45,
  chroma: 0.34,
  volumetric: 0.4,
  grain: 0.35,
  vignette: 0.45,
  particleSize: 1.6,
  trails: 0.72,
  resolution: 1,
};

const DEFAULT_CANDIDATES = [
  { label: 'FINAL s1.5 tr0.8 e1.0', set: { particleSize: 1.5, curl: 1.1, trails: 0.8 }, framing: [2.3, 1.15] },
  { label: 'FINAL s1.5 tr0.8 e0.85', set: { particleSize: 1.5, curl: 1.1, trails: 0.8, exposure: 0.85 }, framing: [2.3, 1.15] },
  { label: 'FINAL + bl0.85 thr2.8', set: { particleSize: 1.5, curl: 1.1, trails: 0.8, exposure: 0.85, bloom: 0.85, bloomThreshold: 2.8 }, framing: [2.3, 1.15] },
  { label: 'FINAL + vol0.5 dof0.55', set: { particleSize: 1.5, curl: 1.1, trails: 0.8, exposure: 0.85, bloom: 0.85, bloomThreshold: 2.8, volumetric: 0.5, dof: 0.55 }, framing: [2.3, 1.15] },
];

const candidates = args.has('candidates')
  ? JSON.parse(args.get('candidates'))
  : DEFAULT_CANDIDATES;

const puppeteer = require('puppeteer-core');
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: ['--headless=new', '--enable-unsafe-webgpu', '--no-sandbox', '--disable-dev-shm-usage', '--mute-audio'],
  defaultViewport: { width: 1280, height: 720 },
  protocolTimeout: 300_000,
});
const page = await browser.newPage();
page.on('pageerror', (e) => console.log(`[pageerror] ${e.message}`));
await page.goto(`${URL_UNDER_TEST}${URL_UNDER_TEST.includes('?') ? '&' : '?'}selftest=1`, {
  waitUntil: 'load',
  timeout: 60_000,
});
await sleep(5000);

const capture = () =>
  page.evaluate(async () => {
    const fn = window.__AETHERIA_CAPTURE;
    if (typeof fn !== 'function') return null;
    const t = new Promise((r) => setTimeout(() => r(null), 25_000));
    const res = await Promise.race([fn(), t]);
    return res ?? null;
  });

const analyse = (png) =>
  page.evaluate(async (b64) => {
    const img = new Image();
    img.src = 'data:image/png;base64,' + b64;
    await img.decode();
    const GW = 40;
    const GH = 18;
    const cv = document.createElement('canvas');
    cv.width = GW;
    cv.height = GH;
    const ctx = cv.getContext('2d');
    ctx.drawImage(img, 0, 0, GW, GH);
    const d = ctx.getImageData(0, 0, GW, GH).data;
    const tiles = [];
    for (let i = 0; i < d.length; i += 4) {
      tiles.push(0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]);
    }
    const mean = tiles.reduce((a, b) => a + b, 0) / tiles.length;
    const sd = Math.sqrt(tiles.reduce((a, b) => a + (b - mean) ** 2, 0) / tiles.length);
    // How much of the total light sits in the brightest 10% of tiles: a high
    // value means a blown-out core, a low one means an evenly spread nebula.
    const sorted = [...tiles].sort((a, b) => a - b);
    const top = sorted.slice(Math.floor(sorted.length * 0.9));
    const concentration = top.reduce((a, b) => a + b, 0) / Math.max(1, tiles.reduce((a, b) => a + b, 0));
    const chars = ' .:-=+*#%@';
    const ramp = [];
    for (let y = 0; y < GH; y++) {
      let line = '';
      for (let x = 0; x < GW; x++) {
        const v = tiles[y * GW + x];
        const t = Math.min(1, Math.pow(v / 110, 0.55));
        line += chars[Math.min(chars.length - 1, Math.floor(t * chars.length))];
      }
      ramp.push(line);
    }
    return {
      tileMean: mean,
      tileSd: sd,
      p50: sorted[Math.floor(sorted.length / 2)],
      p95: sorted[Math.floor(sorted.length * 0.95)],
      tileMax: sorted[sorted.length - 1],
      concentration,
      ramp,
    };
  }, png);

console.log(
  'candidate'.padEnd(26),
  'peak'.padStart(5),
  'tileMean'.padStart(8),
  'tileSd'.padStart(7),
  'tileP95'.padStart(8),
  'clust'.padStart(6),
  'lit%'.padStart(6),
  'clip%'.padStart(6)
);

for (const c of candidates) {
  await page.evaluate(
    (cfg) => {
      Object.assign(window.__AETHERIA_PARAMS, cfg.base, cfg.set);
      if (cfg.framing) window.__AETHERIA_FRAMING?.(cfg.framing[0], cfg.framing[1]);
    },
    { base: BASE, set: c.set, framing: c.framing }
  );
  // Give the simulation time to reach the new steady state: the confinement and
  // curl terms reshape the cloud over several seconds, not instantly.
  await sleep(7000);

  const shot = await capture();
  if (!shot) {
    console.log(`${c.label.padEnd(26)}  capture failed`);
    continue;
  }
  const a = await analyse(shot.png);
  console.log(
    c.label.padEnd(26),
    String(Math.round(shot.max)).padStart(5),
    a.tileMean.toFixed(1).padStart(8),
    a.tileSd.toFixed(1).padStart(7),
    a.p95.toFixed(1).padStart(8),
    `${(a.concentration * 100).toFixed(0)}%`.padStart(6),
    `${(shot.litRatio * 100).toFixed(1)}%`.padStart(6),
    `${(shot.clippedRatio * 100).toFixed(2)}%`.padStart(6)
  );
  console.log(a.ramp.map((l) => '   |' + l + '|').join('\n'));

  if (SAVE) {
    const name = c.label.replace(/[^a-z0-9]+/gi, '-').toLowerCase();
    writeFileSync(`perf/tune-${name}.png`, Buffer.from(shot.png, 'base64'));
    console.log(`   saved perf/tune-${name}.png`);
  }
}

await browser.close();
