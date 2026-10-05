/**
 * Aetheria — control surface.
 *
 * A minimal glass panel that (a) exposes the parameters that materially change
 * the image, (b) shows real telemetry, and (c) never gets in the way: it fades
 * to 16% opacity after a few seconds of stillness, restores on hover or focus,
 * and can be removed entirely with H.
 *
 * Accessibility: every control is a real `<input type="range">`/`<button>` with
 * a label, `aria-*` attributes, and full keyboard operation. Running the whole
 * universe is possible without a pointer.
 */

import { PARAM_RANGES, clampParam, type Params, type RangeKey } from '../core/config';
import type { AudioSource } from '../audio/analysis';
import type { HudStats } from '../core/types';

export interface UiCallbacks {
  onParams(opts?: { structural?: boolean }): void;
  onPauseToggle(): void;
  onReset(): void;
  onExportPng(): void;
  onRecordToggle(): void;
  onAudio(source: AudioSource): void;
  onGyroToggle(enabled: boolean): void;
  onFocusToggle(focused: boolean): void;
}

interface SliderSpec {
  key: RangeKey;
  label: string;
  title: string;
  format?: 'count' | 'pct' | 'num2' | 'num1';
}

const UNIVERSE_SLIDERS: SliderSpec[] = [
  { key: 'simCount', label: 'Particles', title: 'Simulated particles held in GPU storage buffers', format: 'count' },
  { key: 'showCount', label: 'Displayed', title: 'Drawn particles; extra copies are scattered sub-particles', format: 'count' },
  { key: 'speed', label: 'Speed', title: 'Simulation time scale', format: 'num2' },
  { key: 'fractalDim', label: 'Fractal dim.', title: 'Drives the Mandelbulb exponent and Julia iteration depth', format: 'pct' },
  { key: 'curl', label: 'Curl noise', title: 'Strength of the divergence-free 4D turbulence field', format: 'num2' },
  { key: 'damping', label: 'Damping', title: 'Velocity damping per second', format: 'num2' },
  { key: 'gravity', label: 'Gravity', title: 'Inverse-square pull toward the origin', format: 'num2' },
  { key: 'confinement', label: 'Confinement', title: 'Shell force that keeps the nebula bounded', format: 'num2' },
  { key: 'shock', label: 'Supernova', title: 'Shockwave impulse strength (click or tap the canvas)', format: 'num2' },
  { key: 'trails', label: 'Trails', title: 'Frame-history persistence: higher is silkier', format: 'pct' },
];

const OPTICS_SLIDERS: SliderSpec[] = [
  { key: 'exposure', label: 'Exposure', title: 'Linear exposure applied before ACES tone mapping', format: 'num2' },
  { key: 'bloom', label: 'Bloom', title: 'Multi-level HDR bloom intensity', format: 'num2' },
  { key: 'bloomRadius', label: 'Bloom size', title: 'Bloom kernel radius bias', format: 'pct' },
  { key: 'dof', label: 'Depth of field', title: 'Bokeh strength; focus tracks the cloud centre', format: 'num2' },
  { key: 'chroma', label: 'Chromatic', title: 'Radial chromatic aberration', format: 'num2' },
  { key: 'volumetric', label: 'Volumetric', title: 'Radial god-ray accumulation from bright cores', format: 'num2' },
  { key: 'grain', label: 'Film grain', title: 'Luma-weighted grain, applied after tone mapping', format: 'pct' },
  { key: 'vignette', label: 'Vignette', title: 'Dynamic radial darkening', format: 'pct' },
  { key: 'particleSize', label: 'Star size', title: 'Sprite radius; larger values merge the particles into continuous gas', format: 'num2' },
  { key: 'hue', label: 'Hue', title: 'Palette rotation across magenta, cyan, gold and violet', format: 'pct' },
  { key: 'audioSensitivity', label: 'Audio gain', title: 'Multiplier applied to the analysed bands', format: 'num2' },
];

function formatValue(spec: SliderSpec, value: number): string {
  switch (spec.format) {
    case 'count':
      return value >= 1_000_000 ? `${(value / 1_000_000).toFixed(value >= 10_000_000 ? 0 : 1)}M` : `${Math.round(value / 1000)}k`;
    case 'pct':
      return `${Math.round(value * 100)}%`;
    case 'num1':
      return value.toFixed(1);
    default:
      return value.toFixed(2);
  }
}

export class Ui {
  readonly root: HTMLElement;
  private readonly params: Params;
  private readonly callbacks: UiCallbacks;
  private readonly sliders = new Map<string, { input: HTMLInputElement; output: HTMLOutputElement; spec: SliderSpec }>();
  private readonly readouts = new Map<string, HTMLElement>();
  private readonly toastEl: HTMLElement;
  private toastTimer = 0;
  private idleTimer = performance.now();
  /**
   * Slider construction fires the change callback once per control to paint it.
   * Those initial passes must not run application side effects: the owner has not
   * received its `Ui` reference yet, and reporting "the user changed something"
   * for a control that has not been touched would be a lie.
   */
  private ready = false;
  private readonly buttons = new Map<string, HTMLButtonElement>();
  private frameMeter: HTMLElement | null = null;
  private rateMeter: HTMLElement | null = null;

  constructor(params: Params, callbacks: UiCallbacks) {
    this.params = params;
    this.callbacks = callbacks;
    this.root = document.createElement('aside');
    this.root.className = 'panel';
    this.root.id = 'panel';
    this.root.setAttribute('role', 'complementary');
    this.root.setAttribute('aria-label', 'Aetheria controls');
    this.toastEl = document.createElement('div');
    this.toastEl.className = 'toast';
    this.toastEl.setAttribute('role', 'status');
    this.toastEl.setAttribute('aria-live', 'polite');
    this.build();
    this.ready = true;
  }

  // -------------------------------------------------------------------------
  private build(): void {
    const head = document.createElement('div');
    head.className = 'panel-head';
    const title = document.createElement('h1');
    title.className = 'wordmark';
    title.textContent = 'Aetheria';
    const backend = document.createElement('span');
    backend.className = 'badge';
    backend.id = 'backend-badge';
    backend.textContent = 'booting';
    const uiToggle = this.iconButton('hide-ui', '\u25e1', 'Hide interface (H)', () => this.toggleVisible());
    uiToggle.setAttribute('aria-label', 'Hide interface');
    head.append(title, backend, uiToggle);

    const scroll = document.createElement('div');
    scroll.className = 'panel-scroll';

    scroll.append(this.buildStats());
    scroll.append(this.buildAudioRow());
    scroll.append(this.buildSection('Universe', UNIVERSE_SLIDERS, true));
    scroll.append(this.buildSection('Optics', OPTICS_SLIDERS, false));
    scroll.append(this.buildActions());

    this.root.append(head, scroll);
    document.body.append(this.root, this.toastEl);
  }

  private buildStats(): HTMLElement {
    const dl = document.createElement('dl');
    dl.className = 'stats';
    dl.setAttribute('aria-label', 'Performance');

    const entries: [string, string][] = [
      ['fps', 'FPS'],
      ['gpu', 'GPU'],
      ['cpu', 'CPU'],
      ['particles', 'Particles'],
      ['scale', 'Scale'],
      ['memory', 'VRAM'],
    ];
    for (const [id, label] of entries) {
      const wrap = document.createElement('div');
      wrap.className = 'stat';
      const dt = document.createElement('dt');
      dt.textContent = label;
      const dd = document.createElement('dd');
      dd.id = `stat-${id}`;
      dd.textContent = '--';
      wrap.append(dt, dd);
      dl.append(wrap);
      this.readouts.set(id, dd);
    }

    const meter = document.createElement('div');
    meter.className = 'meter';
    meter.setAttribute('role', 'presentation');
    const fill = document.createElement('i');
    meter.append(fill);
    dl.append(meter);
    this.frameMeter = fill;

    const audio = document.createElement('div');
    audio.className = 'meter';
    const afill = document.createElement('i');
    afill.style.background = 'linear-gradient(90deg, var(--accent-3), var(--accent-2))';
    audio.append(afill);
    dl.append(audio);
    this.rateMeter = afill;
    return dl;
  }

  private buildAudioRow(): HTMLElement {
    const section = document.createElement('div');
    section.className = 'section';
    const h = document.createElement('h3');
    h.textContent = 'Audio';
    const row = document.createElement('div');
    row.className = 'row';

    const make = (id: string, text: string, title: string, action: () => void) => {
      const b = document.createElement('button');
      b.id = id;
      b.textContent = text;
      b.title = title;
      b.setAttribute('aria-pressed', 'false');
      b.addEventListener('click', action);
      this.buttons.set(id, b);
      return b;
    };

    row.append(
      make('audio-off', 'Silent', 'No audio analysis (Procedural audio: synth drone)', () => this.callbacks.onAudio('off')),
      make('audio-synth', 'Drone', 'Procedural cosmic drone drives the visuals', () => this.callbacks.onAudio('synth')),
      make('audio-mic', 'Mic', 'Use the microphone (requires permission)', () => this.callbacks.onAudio('mic')),
      make('gyro', 'Gyro', 'Use device orientation (mobile)', () => this.toggleGyro())
    );
    section.append(h, row);
    return section;
  }

  private buildSection(title: string, specs: SliderSpec[], structural: boolean): HTMLElement {
    const section = document.createElement('div');
    section.className = 'section';
    const h = document.createElement('h3');
    h.textContent = title;
    section.append(h);
    for (const spec of specs) section.append(this.buildSlider(spec, structural));
    return section;
  }

  private buildSlider(spec: SliderSpec, structural: boolean): HTMLElement {
    const range = PARAM_RANGES[spec.key];
    const wrap = document.createElement('div');
    wrap.className = 'slider';

    const id = `p-${spec.key}`;
    const label = document.createElement('label');
    label.htmlFor = id;
    label.textContent = spec.label;
    label.title = spec.title;

    const input = document.createElement('input');
    input.type = 'range';
    input.id = id;
    input.min = String(range.min);
    input.max = String(range.max);
    input.step = String(range.step);
    input.value = String(this.params[spec.key] as number);
    input.title = spec.title;
    input.setAttribute('aria-label', spec.label);

    const output = document.createElement('output');
    output.htmlFor = id;
    output.textContent = formatValue(spec, input.valueAsNumber);

    const initial = this.params[spec.key] as number;
    let last = initial;
    const sync = (force = false) => {
      const v = input.valueAsNumber;
      (this.params as unknown as Record<string, number>)[spec.key] = clampParam(spec.key, v);
      output.textContent = formatValue(spec, v);
      const pct = ((v - range.min) / Math.max(1e-6, range.max - range.min)) * 100;
      input.style.setProperty('--fill', `${pct.toFixed(1)}%`);
      // Only report real edits: the initial pass just paints the control.
      if (this.ready && (force || Math.abs(v - last) > 1e-9)) {
        last = v;
        this.callbacks.onParams({ structural });
      } else {
        last = v;
      }
    };
    input.addEventListener('input', () => sync());
    input.addEventListener('change', () => sync());
    sync(true);

    wrap.append(label, input, output);
    this.sliders.set(spec.key, { input, output, spec });
    return wrap;
  }

  private buildActions(): HTMLElement {
    const section = document.createElement('div');
    section.className = 'section';
    const h = document.createElement('h3');
    h.textContent = 'Actions';
    const grid = document.createElement('div');
    grid.className = 'buttons';

    const add = (id: string, text: string, title: string, fn: () => void, pressed = false) => {
      const b = document.createElement('button');
      b.id = id;
      b.textContent = text;
      b.title = title;
      b.setAttribute('aria-pressed', String(pressed));
      b.addEventListener('click', fn);
      this.buttons.set(id, b);
      grid.append(b);
      return b;
    };

    add('pause', 'Pause', 'Pause simulation (Space)', () => this.callbacks.onPauseToggle());
    add('reset', 'Reset', 'Reset camera, rotation and particles (R)', () => this.callbacks.onReset());
    add('focus', 'Focus mode', 'Dim the interface to a minimal HUD (F)', () => this.toggleFocus());
    add('png', 'Save PNG', 'Export the current frame (S)', () => this.callbacks.onExportPng());
    add('webm', 'Record 10s', 'Record a 10 second WebM clip (V)', () => this.callbacks.onRecordToggle());
    add('fullscreen', 'Fullscreen', 'Fullscreen (double-click the canvas)', () => this.toggleFullscreen());

    section.append(h, grid);
    return section;
  }

  private iconButton(id: string, glyph: string, title: string, fn: () => void): HTMLButtonElement {
    const b = document.createElement('button');
    b.id = id;
    b.className = 'icon';
    b.textContent = glyph;
    b.title = title;
    b.setAttribute('aria-label', title);
    b.addEventListener('click', fn);
    this.buttons.set(id, b);
    return b;
  }

  // -------------------------------------------------------------------------
  // State reflection
  // -------------------------------------------------------------------------
  setBackend(label: string, degraded: boolean): void {
    const el = document.getElementById('backend-badge');
    if (!el) return;
    el.textContent = label;
    el.className = degraded ? 'badge warn' : 'badge';
    el.title = degraded ? 'Running with automatic quality reduction' : 'Running at full quality';
  }

  setAudioState(source: AudioSource, message: string): void {
    for (const [id, value] of [
      ['audio-off', 'off'],
      ['audio-synth', 'synth'],
      ['audio-mic', 'mic'],
    ] as const) {
      this.buttons.get(id)?.setAttribute('aria-pressed', String(source === value));
    }
    const badge = document.getElementById('backend-badge');
    if (badge && source === 'mic') badge.title = message;
    this.toast(message);
  }

  setPaused(paused: boolean): void {
    const b = this.buttons.get('pause');
    if (b) {
      b.textContent = paused ? 'Resume' : 'Pause';
      b.setAttribute('aria-pressed', String(paused));
    }
  }

  setRecording(recording: boolean): void {
    const b = this.buttons.get('webm');
    if (b) {
      b.textContent = recording ? 'Recording...' : 'Record 10s';
      b.setAttribute('aria-pressed', String(recording));
    }
  }

  setGyro(on: boolean): void {
    this.buttons.get('gyro')?.setAttribute('aria-pressed', String(on));
  }

  setFocus(on: boolean): void {
    this.buttons.get('focus')?.setAttribute('aria-pressed', String(on));
    this.callbacks.onFocusToggle(on);
  }

  /** Push every slider back from `params` (used after a URL-hash load). */
  syncFromParams(): void {
    for (const [, entry] of this.sliders) {
      const v = this.params[entry.spec.key] as number;
      entry.input.value = String(v);
      entry.output.textContent = formatValue(entry.spec, v);
      const range = PARAM_RANGES[entry.spec.key];
      const pct = ((v - range.min) / Math.max(1e-6, range.max - range.min)) * 100;
      entry.input.style.setProperty('--fill', `${pct.toFixed(1)}%`);
    }
  }

  updateStats(stats: HudStats, audioLevel: number): void {
    const set = (id: string, text: string, cls?: string) => {
      const el = this.readouts.get(id);
      if (!el) return;
      el.textContent = text;
      if (cls !== undefined) el.className = cls;
    };

    const fps = stats.fps;
    set('fps', `${fps.toFixed(0)}`, fps >= 55 ? 'good' : fps >= 30 ? 'mid' : 'bad');
    set('gpu', stats.gpuTimingSupported ? `${stats.gpuMs.toFixed(2)}ms` : `${stats.frameMs.toFixed(1)}ms`);
    set('cpu', `${stats.cpuMs.toFixed(2)}ms`);
    // The suffix shows how many particles the frame budget declined to draw, so a
    // reduced count is visible rather than looking like a broken slider.
    const culled = stats.culledCount > 0 ? ` (-${(stats.culledCount / 1e6).toFixed(1)}M)` : '';
    set('particles', `${(stats.simCount / 1e6).toFixed(2)}M / ${(stats.renderCount / 1e6).toFixed(1)}M${culled}`);
    set('scale', `${(stats.renderScale * 100).toFixed(0)}%`, stats.renderScale < 0.75 ? 'mid' : undefined);
    set('memory', `${stats.memoryEstimateMB.toFixed(0)}MB`, stats.memoryEstimateMB > 480 ? 'bad' : undefined);

    const budget = 16.6;
    if (this.frameMeter) {
      const pct = Math.min(100, (stats.frameMs / (budget * 2)) * 100);
      this.frameMeter.style.width = `${pct.toFixed(1)}%`;
      this.frameMeter.style.background =
        stats.frameMs > budget * 1.3
          ? 'linear-gradient(90deg, var(--warn), #ffd0c0)'
          : 'linear-gradient(90deg, var(--accent), var(--accent-2))';
    }
    if (this.rateMeter) this.rateMeter.style.width = `${Math.min(100, audioLevel * 100).toFixed(0)}%`;
  }

  toast(message: string, ms = 2600): void {
    if (!message) return;
    this.toastEl.textContent = message;
    this.toastEl.classList.add('show');
    window.clearTimeout(this.toastTimer);
    this.toastTimer = window.setTimeout(() => this.toastEl.classList.remove('show'), ms);
  }

  toggleVisible(): void {
    const hidden = document.body.classList.toggle('ui-hidden');
    const b = this.buttons.get('hide-ui');
    b?.setAttribute('aria-label', hidden ? 'Show interface' : 'Hide interface');
    b?.setAttribute('aria-pressed', String(hidden));
  }

  get visible(): boolean {
    return !document.body.classList.contains('ui-hidden');
  }

  toggleFocus(): void {
    this.setFocus(!document.body.classList.contains('panel-idle'));
  }

  private async toggleFullscreen(): Promise<void> {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await document.documentElement.requestFullscreen();
    } catch {
      this.toast('Fullscreen was blocked by the browser');
    }
  }

  private gyroOn = false;

  private toggleGyro(): void {
    this.gyroOn = !this.gyroOn;
    this.callbacks.onGyroToggle(this.gyroOn);
  }

  /** Idle fade: the panel dims itself when nothing has happened for a while. */
  noteActivity(): void {
    this.idleTimer = performance.now();
    document.body.classList.remove('panel-idle');
  }

  tick(): void {
    if (performance.now() - this.idleTimer > 6000 && !document.body.classList.contains('panel-idle')) {
      document.body.classList.add('panel-idle');
    }
  }

  /** Reflect external parameter changes (URL hash, adaptive degradation). */
  setParam(key: RangeKey, value: number, structural = false): void {
    (this.params as unknown as Record<string, number>)[key] = value;
    const entry = this.sliders.get(key);
    if (entry) {
      entry.input.value = String(value);
      entry.output.textContent = formatValue(entry.spec, value);
      const range = PARAM_RANGES[key];
      const pct = ((value - range.min) / Math.max(1e-6, range.max - range.min)) * 100;
      entry.input.style.setProperty('--fill', `${pct.toFixed(1)}%`);
    }
    this.callbacks.onParams({ structural });
  }

  dispose(): void {
    window.clearTimeout(this.toastTimer);
    this.root.remove();
    this.toastEl.remove();
  }
}
