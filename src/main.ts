/**
 * Aetheria 鈥?application shell.
 *
 * Responsibilities, in order:
 *   1. detect the device profile and decode a shared URL hash
 *   2. bring up the WebGPU renderer, falling back to WebGL2 (and reporting why)
 *   3. run the frame loop: audio analysis -> simulation -> render -> telemetry
 *   4. own the interaction verbs, media export and hash persistence
 *
 * Everything heavy lives behind the `Renderer` interface, so this file contains
 * no API-specific code at all.
 */

import './ui/styles.css';

import { clampParam, defaultParams, detectProfile, PARAM_RANGES, type Params } from './core/config';
import type { BackendKind, Renderer, RendererTelemetry } from './core/types';
import { GpuRenderer } from './gpu/renderer';
import { AudioEngine, type AudioSource } from './audio/analysis';
import { InputController, type KeyAction } from './input/controls';
import { Ui } from './ui/panel';
import { SelfTest } from './dev/selftest';
import {
  SIMULATE_WGSL,
  SEED_WGSL,
} from './gpu/wgsl/simulate.wgsl';
import { PARTICLE_VERTEX_WGSL, PARTICLE_FRAGMENT_WGSL } from './gpu/wgsl/particle.wgsl';
import {
  FADE_WGSL,
  BRIGHT_PASS_WGSL,
  BLOOM_DOWN_WGSL,
  BLOOM_UP_WGSL,
  GODRAY_WGSL,
  DOF_DOWN_WGSL,
  DOF_GATHER_WGSL,
  COMPOSITE_WGSL,
  FXAA_WGSL,
  FULLSCREEN_VS,
} from './gpu/wgsl/post.wgsl';

/**
 * Debug channel: with `?wgsl=1` the app republishes the exact WGSL it hands to
 * the driver. The verification runner reads it, so shader diagnostics can be
 * reproduced and inspected with the same line numbers the compiler reports 鈥? * which is the only reliable way to debug a template-assembled module.
 */
if (new URLSearchParams(window.location.search).has('wgsl')) {
  (window as unknown as Record<string, unknown>).__AETHERIA_PASSES = [];  (window as unknown as Record<string, unknown>).__AETHERIA_WGSL = {
    'simulate.wgsl': SIMULATE_WGSL,
    'seed.wgsl': SEED_WGSL,
    'particle.vert.wgsl': PARTICLE_VERTEX_WGSL,
    'particle.frag.wgsl': PARTICLE_FRAGMENT_WGSL,
    'fullscreen.vert.wgsl': FULLSCREEN_VS,
    'fade.wgsl': FADE_WGSL,
    'bright.wgsl': BRIGHT_PASS_WGSL,
    'bloomDown.wgsl': BLOOM_DOWN_WGSL,
    'bloomUp.wgsl': BLOOM_UP_WGSL,
    'godray.wgsl': GODRAY_WGSL,
    'dofDown.wgsl': DOF_DOWN_WGSL,
    'dofGather.wgsl': DOF_GATHER_WGSL,
    'composite.wgsl': COMPOSITE_WGSL,
    'fxaa.wgsl': FXAA_WGSL,
  };
}

/* ------------------------------------------------------------------ hash --- */

const HASH_KEYS: Record<string, keyof Params> = {
  n: 'simCount',
  r: 'showCount',
  sp: 'speed',
  fd: 'fractalDim',
  cu: 'curl',
  da: 'damping',
  gr: 'gravity',
  cf: 'confinement',
  sk: 'shock',
  tr: 'trails',
  ex: 'exposure',
  bl: 'bloom',
  br: 'bloomRadius',
  df: 'dof',
  ch: 'chroma',
  vo: 'volumetric',
  gn: 'grain',
  vg: 'vignette',
  hu: 'hue',
  au: 'audioSensitivity',
  rs: 'resolution',
  j0: 'julia',
};

function encodeHash(params: Params): string {
  const parts: string[] = [];
  const push = (key: string, value: number) => parts.push(`${key}=${Number.isInteger(value) ? value : value.toFixed(3)}`);
  push('n', params.simCount);
  push('r', params.showCount);
  push('sp', params.speed);
  push('fd', params.fractalDim);
  push('cu', params.curl);
  push('da', params.damping);
  push('gr', params.gravity);
  push('cf', params.confinement);
  push('sk', params.shock);
  push('tr', params.trails);
  push('ex', params.exposure);
  push('bl', params.bloom);
  push('br', params.bloomRadius);
  push('df', params.dof);
  push('ch', params.chroma);
  push('vo', params.volumetric);
  push('gn', params.grain);
  push('vg', params.vignette);
  push('hu', params.hue);
  push('au', params.audioSensitivity);
  push('rs', params.resolution);
  parts.push(`j=${params.julia.map((v) => v.toFixed(3)).join(',')}`);
  return parts.join('&');
}

function decodeHash(params: Params): void {
  const raw = window.location.hash.replace(/^#/, '');
  if (!raw) return;
  for (const chunk of raw.split('&')) {
    const eq = chunk.indexOf('=');
    if (eq < 0) continue;
    const key = chunk.slice(0, eq);
    const value = chunk.slice(eq + 1);
    if (key === 'j') {
      const nums = value.split(',').map(Number);
      if (nums.length === 4 && nums.every(Number.isFinite)) {
        params.julia = [nums[0]!, nums[1]!, nums[2]!, nums[3]!];
      }
      continue;
    }
    const target = HASH_KEYS[key];
    if (!target || target === 'julia') continue;
    const num = Number(value);
    if (!Number.isFinite(num)) continue;
    if (target === 'simCount' || target === 'showCount') {
      (params as unknown as Record<string, number>)[target] = Math.max(100_000, Math.round(num));
    } else if (target in PARAM_RANGES) {
      (params as unknown as Record<string, number>)[target] = clampParam(target as keyof typeof PARAM_RANGES, num);
    }
  }
}

/* ------------------------------------------------------------- bootstrap --- */

interface AppHandle {
  renderer: Renderer;
  kind: BackendKind;
  note: string;
}

async function bootstrapRenderer(
  canvas: HTMLCanvasElement,
  profile: ReturnType<typeof detectProfile>,
  params: Params,
  telemetry: RendererTelemetry,
  reducedMotion: boolean
): Promise<AppHandle> {
  const notes: string[] = [];
  // `?renderer=webgl2` exercises the fallback on a machine that has WebGPU, which
  // is otherwise very hard to test: the fallback is the one code path a normal
  // run never touches. `?renderer=webgpu` pins the primary path for the same
  // reason in the other direction.
  const forced = new URLSearchParams(window.location.search).get('renderer');
  const hasWebGpu = forced !== 'webgl2' && typeof navigator !== 'undefined' && 'gpu' in navigator;

  if (hasWebGpu) {
    try {
      const renderer = await GpuRenderer.create(canvas, profile, params, telemetry, reducedMotion);
      return { renderer, kind: 'webgpu', note: 'WebGPU' };
    } catch (err) {
      const message = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
      const detail = (err as { detail?: string }).detail ?? '';
      notes.push(`WebGPU unavailable (${message})${detail ? `\n${detail}` : ''}`);
      console.warn('[aetheria] falling back to WebGL2:', message, detail);
    }
  } else {
    notes.push(
      forced === 'webgl2' ? 'WebGPU was disabled by ?renderer=webgl2' : 'navigator.gpu is not present in this browser'
    );
  }

  // ---- WebGL2 fallback --------------------------------------------------
  // Imported lazily so a browser with working WebGPU never downloads or parses
  // the fallback's shader sources, and so a syntax problem in the fallback
  // cannot stop the primary path from starting.
  try {
    const { WebGL2Renderer } = await import('./webgl2/renderer');
    const renderer = new WebGL2Renderer(canvas, profile, params, telemetry);
    await renderer.init(canvas);
    return { renderer, kind: 'webgl2', note: `WebGL2 fallback - ${notes.join('; ')}` };
  } catch (err) {
    const message = err instanceof Error ? `${err.name}: ${err.message}` : String(err);
    const detail = (err as { detail?: string }).detail ?? '';
    notes.push(`WebGL2 unavailable (${message})${detail ? `\n${detail}` : ''}`);
  }

  // Publish the full diagnostic chain so the runner can read it back; a
  // truncated GLSL log is useless for fixing a shader.
  const text = notes.join('\n\n');
  (window as unknown as Record<string, unknown>).__AETHERIA_BOOT_ERROR = text;
  throw new Error(text);
}

/* ------------------------------------------------------------------ app ---- */

class Aetheria {
  private canvas: HTMLCanvasElement;
  private profile = detectProfile();
  private params: Params;
  private renderer!: Renderer;
  private audio = new AudioEngine();
  private input!: InputController;
  private ui!: Ui;
  private boot: HTMLElement;
  private raf = 0;
  private lastNow = 0;
  private reducedMotion: MediaQueryList;
  private hashTimer = 0;
  private recorder: MediaRecorder | null = null;
  private recordTimer = 0;
  private focused = false;
  private lastUiUpdate = 0;
  private selfTest: SelfTest | null = null;

  constructor() {
    this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.canvas = document.getElementById('stage') as HTMLCanvasElement;
    this.boot = document.getElementById('boot') as HTMLElement;
    if (!this.canvas) throw new Error('canvas #stage missing');
    this.params = defaultParams(this.profile);
    decodeHash(this.params);
    if (this.reducedMotion.matches) {
      // Low-motion mode: kill the long trails and the idle drift, keep the art.
      this.params.trails = Math.min(this.params.trails, 0.34);
      this.params.dof = Math.min(this.params.dof, 0.25);
      this.params.chroma = Math.min(this.params.chroma, 0.18);
      this.params.volumetric = Math.min(this.params.volumetric, 0.18);
    }
  }

  async start(): Promise<void> {
    // The UI is built first and the renderer attached afterwards: the renderer's
    // telemetry callbacks (fatal errors, adaptive degradation) fire during setup,
    // so they must have somewhere to go before that setup can fail.
    this.ui = new Ui(this.params, {
      onParams: (opts) => {
        this.ui.noteActivity();
        this.scheduleHash();
        if (opts?.structural) this.applyQualityScale();
      },
      onPauseToggle: () => this.setPaused(!this.params.paused),
      onReset: () => {
        this.renderer?.reset();
        this.ui.toast('Universe reset');
      },
      onExportPng: () => void this.exportPng(),
      onRecordToggle: () => this.toggleRecording(),
      onAudio: (source) => void this.setAudio(source),
      onGyroToggle: (on) => void this.setGyro(on),
      onFocusToggle: (on) => {
        this.focused = on;
      },
    });
    this.ui.setFocus(false);

    const telemetry: RendererTelemetry = {
      onFatal: (message, detail) => this.showFatal(message, detail),
      onDegrade: (reason) => {
        this.ui.setBackend(`adaptive: ${reason.replace('auto-degraded: ', '')}`, true);
      },
    };

    let handle: AppHandle;
    try {
      handle = await bootstrapRenderer(this.canvas, this.profile, this.params, telemetry, this.reducedMotion.matches);
    } catch (err) {
      this.ui.dispose();
      this.showFatal('This browser cannot run Aetheria', err instanceof Error ? err.message : String(err));
      return;
    }
    this.renderer = handle.renderer;

    const degraded = this.renderer.stats.degraded.length > 0;
    this.ui.setBackend(
      handle.kind === 'webgpu' ? `WebGPU - ${this.profile.tier}` : `WebGL2 - ${this.profile.tier}${degraded ? ' (reduced)' : ''}`,
      degraded
    );
    this.ui.setPaused(this.params.paused);
    this.ui.syncFromParams();

    // ---- input ------------------------------------------------------------
    this.input = new InputController(this.canvas, {
      orbit: (dx, dy, plane) => this.renderer.orbit(dx, dy, plane),
      zoom: (d) => this.renderer.zoom(d),
      slice: (d) => this.renderer.slice(d),
      ignite: (x, y) => this.renderer.shock(x, y),
      key: (action) => this.onKey(action),
      activity: () => {
        this.ui.noteActivity();
        if (this.focused) this.ui.setFocus(false);
      },
    });

    window.addEventListener('resize', this.onResize, { passive: true });
    window.visualViewport?.addEventListener('resize', this.onResize, { passive: true });
    document.addEventListener('visibilitychange', this.onVisibility);
    this.canvas.addEventListener('dblclick', () => void this.ui.toggleFocus(), { passive: true });
    this.reducedMotion.addEventListener('change', this.onReducedMotion);
    window.addEventListener('hashchange', () => {
      decodeHash(this.params);
      this.ui.syncFromParams();
      this.applyQualityScale();
      this.ui.toast('Loaded universe from URL');
    });
    window.addEventListener('beforeunload', () => this.renderer.dispose());

    if (handle.kind === 'webgl2') {
      this.ui.toast('WebGPU unavailable: running the WebGL2 fallback', 5200);
    } else if (degraded) {
      this.ui.toast(this.renderer.stats.degraded[0] ?? 'Quality reduced', 5200);
    } else {
      this.ui.toast('Drag to rotate in 4D - click to ignite a supernova', 4200);
    }

    this.applyQualityScale();
    this.onResize();

    // Opt-in GPU-side frame capture for the verification runner. Attaching it
    // after the first frames lets the simulation reach a representative state.
    if (new URLSearchParams(window.location.search).has('selftest')) {
      const renderer = this.renderer as unknown as { selfTest?: SelfTest | null };
      this.selfTest = new SelfTest();
      renderer.selfTest = this.selfTest;
      if (handle.kind === 'webgl2') {
        // The WebGL2 renderer is left untouched: its drawing buffer is directly
        // readable, so the capture wraps `frame()` here rather than threading a
        // test hook through the renderer's hot path.
        const gl = this.canvas.getContext('webgl2');
        const inner = this.renderer.frame.bind(this.renderer);
        this.renderer.frame = (now: number, delta: number) => {
          inner(now, delta);
          if (this.selfTest && gl) this.selfTest.attachGl(gl, this.canvas.width, this.canvas.height);
        };
      }
      (window as unknown as Record<string, unknown>).__AETHERIA_CAPTURE = () =>
        this.selfTest ? this.selfTest.request() : Promise.resolve(null);
      // Parameter hook for calibration runs: lets the runner sweep exposure,
      // bloom and the post grade without reloading the page.
      (window as unknown as Record<string, unknown>).__AETHERIA_PARAMS = this.params;
      (window as unknown as Record<string, unknown>).__AETHERIA_STATS = () => this.renderer?.stats ?? null;
      // Camera framing is not a user parameter, but the calibration runner needs
      // to sweep it, so it is reachable here alongside the other test hooks.
      (window as unknown as Record<string, unknown>).__AETHERIA_FRAMING = (distance: number, fovY: number) => {
        const r = this.renderer as unknown as { setFraming?: (d: number, f: number) => void };
        r.setFraming?.(distance, fovY);
      };
      (window as unknown as Record<string, unknown>).__AETHERIA_SETTLE = (ms: number) =>
        new Promise((resolve) => window.setTimeout(resolve, ms));
      // Per-pass GPU breakdown. Kept out of the UI on purpose: the panel has no
      // room for two dozen rows, and the numbers only matter while deciding what
      // to optimise.
      (window as unknown as Record<string, unknown>).__AETHERIA_PASSES = () => {
        const r = this.renderer as unknown as { passTimings?: { label: string; ms: number }[] };
        return r.passTimings ?? [];
      };
      // Ablation switch for the particle pass, so a profiling run can measure one
      // suspect at a time instead of reasoning about it.
      (window as unknown as Record<string, unknown>).__AETHERIA_FREEZE = (frozen: boolean) => {
        const r = this.renderer as unknown as { freezeAdaptation?: (f: boolean) => void };
        r.freezeAdaptation?.(frozen);
      };
      (window as unknown as Record<string, unknown>).__AETHERIA_ABLATE = (mode: string) => {
        const r = this.renderer as unknown as { setAblation?: (m: string) => void };
        r.setAblation?.(mode);
      };
    }
    this.hideBoot();
    this.lastNow = performance.now();
    this.raf = requestAnimationFrame(this.loop);
  }

  private hideBoot(): void {
    this.boot.classList.add('gone');
    window.setTimeout(() => this.boot.remove(), 900);
  }

  private showFatal(message: string, detail?: string): void {
    const text = detail ? `${message}\n\n${detail}` : message;
    // Keep the whole thing reachable for the verification runner; the console
    // truncates, and a GLSL/WGSL log is worthless when cut off.
    (window as unknown as Record<string, unknown>).__AETHERIA_BOOT_ERROR = text;
    const el = document.getElementById('boot-error');
    if (el) el.textContent = text;
    this.boot.classList.remove('gone');
    console.error('[aetheria]', message, detail ?? '');
  }

  /* --------------------------------------------------------------- loop --- */

  private loop = (now: number): void => {
    this.raf = requestAnimationFrame(this.loop);
    if (document.hidden && !this.recorder) return;

    const deltaMs = Math.min(now - this.lastNow, 120);
    this.lastNow = now;

    const audio = this.audio.update(Math.min(deltaMs, 50) / 1000);
    this.renderer.setAudio(audio.low, audio.mid, audio.high, audio.beat);

    try {
      this.renderer.frame(now, deltaMs);
    } catch (err) {
      cancelAnimationFrame(this.raf);
      this.showFatal('Rendering stopped', err instanceof Error ? `${err.message}\n${err.stack ?? ''}` : String(err));
      return;
    }

    // Beat-driven flourish: a supernova plus a palette jump on strong hits.
    if (audio.beat > 0.72 && this.params.shock > 0) {
      const x = Math.sin(now * 0.00021) * 0.62;
      const y = Math.cos(now * 0.00017) * 0.5;
      this.renderer.shock(x, y, 0.55 + audio.beat * 0.4);
    }

    if (now - this.lastUiUpdate > 180) {
      this.lastUiUpdate = now;
      this.ui.updateStats(this.renderer.stats, audio.rms);
      this.ui.tick();
    }
  };

  /* -------------------------------------------------------------- input --- */

  private onKey(action: KeyAction): void {
    switch (action) {
      case 'pause':
        this.setPaused(!this.params.paused);
        break;
      case 'reset':
        this.renderer.reset();
        this.ui.toast('Universe reset');
        break;
      case 'toggle-ui':
        this.ui.toggleVisible();
        break;
      case 'export-png':
        void this.exportPng();
        break;
      case 'record':
        this.toggleRecording();
        break;
      case 'help':
        this.ui.toast(
          'Drag: 4D rotate | Shift+drag: xw/yw planes | Wheel: zoom | Shift+wheel or Q/E: slice | Click: supernova | Space: pause | R: reset | H: hide UI | S: PNG | V: record',
          9000
        );
        break;
      case 'plane-xw':
      case 'plane-yw':
      case 'plane-zw': {
        const plane = action.slice(6) as 'xw' | 'yw' | 'zw';
        this.input.setPlane(this.input.plane === plane ? undefined : plane);
        this.ui.toast(this.input.plane ? `Drag plane locked to ${plane.toUpperCase()}` : 'Drag plane released');
        break;
      }
      default:
        break;
    }
  }

  private setPaused(paused: boolean): void {
    this.params.paused = paused;
    this.ui.setPaused(paused);
    this.ui.toast(paused ? 'Paused' : 'Running');
    this.scheduleHash();
  }

  private async setAudio(source: AudioSource): Promise<void> {
    if (source === 'off') {
      this.audio.stop();
      this.ui.setAudioState('off', 'Audio analysis off');
      return;
    }
    if (source === 'mic') {
      const ok = await this.audio.startMic();
      if (!ok) {
        this.ui.setAudioState('off', `Microphone unavailable: ${this.audio.lastError}`);
        return;
      }
      this.ui.setAudioState('mic', 'Microphone live - the nebula listens');
      return;
    }
    await this.audio.startSynth();
    this.ui.setAudioState('synth', 'Procedural drone live');
  }

  private async setGyro(enabled: boolean): Promise<void> {
    if (!enabled) {
      this.input.disableGyro();
      this.ui.setGyro(false);
      this.ui.toast('Gyroscope off');
      return;
    }
    const ok = await this.input.enableGyro();
    this.ui.setGyro(ok);
    this.ui.toast(ok ? 'Gyroscope steering' : 'Gyroscope unavailable or permission denied');
  }

  /* -------------------------------------------------------------- media --- */

  private async exportPng(): Promise<void> {
    try {
      const blob = await this.renderer.capture();
      if (!blob) {
        this.ui.toast('PNG export is not available on this backend');
        return;
      }
      download(blob, `aetheria-${stamp()}.png`);
      this.ui.toast(`Saved PNG (${(blob.size / 1048576).toFixed(1)} MB)`);
    } catch (err) {
      this.ui.toast(`PNG export failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  private toggleRecording(): void {
    if (this.recorder) {
      this.stopRecording();
      return;
    }
    if (typeof MediaRecorder === 'undefined' || typeof this.canvas.captureStream !== 'function') {
      this.ui.toast('WebM recording is not supported in this browser');
      return;
    }
    try {
      const stream = this.canvas.captureStream(60);
      const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
      const mimeType = types.find((t) => MediaRecorder.isTypeSupported(t));
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 12_000_000 } : undefined);
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (ev) => {
        if (ev.data.size) chunks.push(ev.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunks, { type: recorder.mimeType || 'video/webm' });
        download(blob, `aetheria-${stamp()}.webm`);
        this.ui.toast(`Saved WebM (${(blob.size / 1048576).toFixed(1)} MB)`);
      };
      recorder.start(250);
      this.recorder = recorder;
      this.ui.setRecording(true);
      this.ui.toast('Recording 10 seconds...', 2000);
      this.recordTimer = window.setTimeout(() => this.stopRecording(), 10_000);
    } catch (err) {
      this.ui.toast(`Recording failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  private stopRecording(): void {
    window.clearTimeout(this.recordTimer);
    try {
      this.recorder?.stop();
    } catch {
      /* already stopped */
    }
    this.recorder = null;
    this.ui.setRecording(false);
  }

  /* ---------------------------------------------------------- lifecycle --- */

  /** The user's resolution multiplier composes with the device profile cap. */
  private applyQualityScale(): void {
    this.renderer?.setQualityScale(this.params.resolution * this.profile.maxPostScale);
  }

  private onResize = (): void => {    const dpr = Math.min(window.devicePixelRatio || 1, this.profile.isMobile ? 2.5 : 2);
    this.profile.dpr = dpr;
    const w = this.canvas.clientWidth || window.innerWidth;
    const h = this.canvas.clientHeight || window.innerHeight;
    this.renderer.resize(w, h, dpr, this.params.resolution * this.profile.maxPostScale);
  };

  private onVisibility = (): void => {
    // Pause the clock while hidden so returning does not produce a huge dt jump.
    this.lastNow = performance.now();
  };

  private onReducedMotion = (): void => {
    this.ui.toast(this.reducedMotion.matches ? 'Reduced motion detected: low-motion mode' : 'Motion restored');
  };

  private scheduleHash(): void {
    window.clearTimeout(this.hashTimer);
    this.hashTimer = window.setTimeout(() => {
      const next = `#${encodeHash(this.params)}`;
      if (window.location.hash !== next) {
        history.replaceState(null, '', next);
      }
    }, 420);
  }
}

/* ---------------------------------------------------------------- utils --- */

function download(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.append(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 8000);
}

function stamp(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

/* ----------------------------------------------------------------- main --- */

const app = new Aetheria();
void app.start();
