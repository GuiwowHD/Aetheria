/**
 * Aetheria — audio analysis and the procedural "cosmic drone" source.
 *
 * Two signal paths feed one AnalyserNode, so the visual engine never learns
 * where its music came from:
 *
 *   microphone ──► MediaStreamSource ─┐
 *                                     ├──► AnalyserNode ──► band energies
 *   synth graph (oscillators + LFOs) ─┘        (never to the speakers)
 *
 * The synth exists because requiring a microphone would make the visual design
 * unavailable to most visitors. It is a slow, detuned minor-ninth pad with a
 * breathing filter and a sub pulse every few seconds — musically stable enough
 * that the derived band energies look intentional rather than noisy.
 *
 * Beat detection uses an adaptive threshold over a rolling energy history, which
 * works for both paths without per-source tuning.
 */

export type AudioSource = 'off' | 'mic' | 'synth';

export interface AudioFrame {
  low: number;
  mid: number;
  high: number;
  beat: number;
  /** Raw RMS, 0..1 — used for the HUD level meter. */
  rms: number;
  active: boolean;
}

const FFT_SIZE = 2048;
/** Rolling window for the adaptive beat threshold (~1.4 s at 60 fps). */
const HISTORY = 84;

export class AudioEngine {
  source: AudioSource = 'off';
  lastError = '';
  permission: 'unknown' | 'granted' | 'denied' | 'prompt' = 'unknown';

  private ctx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private freq: Uint8Array<ArrayBuffer> = new Uint8Array(FFT_SIZE / 2);
  private micStream: MediaStream | null = null;
  private micNode: MediaStreamAudioSourceNode | null = null;
  private synth: SynthGraph | null = null;

  private lowEnv = 0;
  private midEnv = 0;
  private highEnv = 0;
  private rmsEnv = 0;
  private history: number[] = [];
  private historyAt = 0;
  private beatEnv = 0;
  private lastBeat = -10;
  private clock = 0;

  get running(): boolean {
    return this.source !== 'off' && this.ctx !== null;
  }

  /** Lazily create the context. Must be called from a user gesture. */
  private ensureContext(): AudioContext {
    if (!this.ctx) {
      const Ctor: typeof AudioContext =
        window.AudioContext ?? (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      this.ctx = new Ctor({ latencyHint: 'interactive' });
      const analyser = this.ctx.createAnalyser();
      analyser.fftSize = FFT_SIZE;
      analyser.smoothingTimeConstant = 0.68;
      analyser.minDecibels = -96;
      analyser.maxDecibels = -14;
      this.analyser = analyser;
      this.freq = new Uint8Array(new ArrayBuffer(analyser.frequencyBinCount));
    }
    if (this.ctx.state === 'suspended') void this.ctx.resume();
    return this.ctx;
  }

  /** Start the procedural source (no permissions required). */
  async startSynth(): Promise<boolean> {
    const ctx = this.ensureContext();
    this.stopNodes();
    if (!this.synth) this.synth = new SynthGraph(ctx);
    this.synth.connect(this.analyser!);
    this.synth.start();
    this.source = 'synth';
    this.lastError = '';
    return true;
  }

  /** Request microphone access and analyse it. */
  async startMic(): Promise<boolean> {
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      this.lastError = 'Microphone API unavailable';
      return false;
    }
    const ctx = this.ensureContext();
    try {
      this.micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      });
    } catch (err) {
      this.permission = 'denied';
      this.lastError = err instanceof Error ? err.message : String(err);
      return false;
    }
    this.permission = 'granted';
    this.stopNodes();
    this.micNode = ctx.createMediaStreamSource(this.micStream);
    this.micNode.connect(this.analyser!);
    this.source = 'mic';
    this.lastError = '';
    return true;
  }

  private stopNodes(): void {
    try {
      this.synth?.stop();
    } catch {
      /* ignore */
    }
    try {
      this.micNode?.disconnect();
    } catch {
      /* ignore */
    }
    this.micNode = null;
    if (this.micStream) {
      for (const track of this.micStream.getTracks()) track.stop();
      this.micStream = null;
    }
  }

  stop(): void {
    this.stopNodes();
    this.source = 'off';
    this.lowEnv = this.midEnv = this.highEnv = this.rmsEnv = 0;
  }

  /** Analyser taps are cheap but not free, so the loop can skip them when idle. */
  setSensitivityTilt(enabled: boolean): void {
    this.tilt = enabled;
  }

  private tilt = true;

  /**
   * Read the analyser and derive four control signals.
   * Bands are perceptual-ish thirds of the spectrum: 30–180 Hz, 180–2 kHz,
   * 2–12 kHz. Energies are normalised to a 0..~1.6 range so `params.audioSensitivity`
   * has headroom to push them further.
   */
  update(dtSeconds: number): AudioFrame {
    const analyser = this.analyser;
    if (!analyser || this.source === 'off') {
      // Decay smoothly to silence rather than snapping, so turning audio off
      // does not produce a visible jolt in the nebula.
      this.lowEnv *= 1 - Math.min(1, dtSeconds * 3);
      this.midEnv *= 1 - Math.min(1, dtSeconds * 3);
      this.highEnv *= 1 - Math.min(1, dtSeconds * 3);
      this.beatEnv *= 1 - Math.min(1, dtSeconds * 4);
      this.rmsEnv *= 1 - Math.min(1, dtSeconds * 3);
      return { low: this.lowEnv, mid: this.midEnv, high: this.highEnv, beat: this.beatEnv, rms: this.rmsEnv, active: false };
    }

    this.clock += dtSeconds;
    analyser.getByteFrequencyData(this.freq);
    const nyquist = this.ctx!.sampleRate / 2;
    const binHz = nyquist / this.freq.length;

    const band = (fromHz: number, toHz: number, curve: number): number => {
      const a = Math.max(0, Math.floor(fromHz / binHz));
      const b = Math.min(this.freq.length - 1, Math.ceil(toHz / binHz));
      let sum = 0;
      for (let i = a; i <= b; i++) {
        const v = (this.freq[i] ?? 0) / 255;
        sum += v * v; // energy, not amplitude: matches perceived loudness better
      }
      const n = Math.max(1, b - a + 1);
      return Math.pow(sum / n, curve);
    };

    const rawLow = band(28, 180, 0.5);
    let rawMid = band(180, 2000, 0.5);
    let rawHigh = band(2000, 12000, 0.55);

    let sum = 0;
    for (let i = 0; i < this.freq.length; i++) {
      const v = (this.freq[i] ?? 0) / 255;
      sum += v * v;
    }
    const rms = Math.sqrt(sum / this.freq.length);

    // Optional perceptual tilt: quiet material should still move the nebula.
    if (this.tilt) {
      rawMid *= 1.25;
      rawHigh *= 1.5;
    }

    // Asymmetric smoothing: fast attack preserves transients (which is what
    // makes the nebula feel like it is reacting), slow release avoids flicker.
    const atk = 1 - Math.exp(-dtSeconds * 22);
    const rel = 1 - Math.exp(-dtSeconds * 6);
    const follow = (env: number, target: number) => env + (target - env) * (target > env ? atk : rel);

    this.lowEnv = follow(this.lowEnv, rawLow * 3.2);
    this.midEnv = follow(this.midEnv, rawMid * 3.4);
    this.highEnv = follow(this.highEnv, rawHigh * 4.0);
    this.rmsEnv = follow(this.rmsEnv, rms);

    // ---- adaptive beat detection on the low band -------------------------
    const energy = rawLow * rawLow;
    if (this.history.length < HISTORY) this.history.push(energy);
    else {
      this.history[this.historyAt % HISTORY] = energy;
      this.historyAt++;
    }
    let mean = 0;
    for (const v of this.history) mean += v;
    mean /= Math.max(1, this.history.length);
    let variance = 0;
    for (const v of this.history) variance += (v - mean) * (v - mean);
    variance /= Math.max(1, this.history.length);
    const std = Math.sqrt(variance);

    const threshold = mean * 1.35 + std * 1.15 + 1e-4;
    let beat = 0;
    if (energy > threshold && this.clock - this.lastBeat > 0.22 && this.history.length > 12) {
      this.lastBeat = this.clock;
      // Strength scales with how far above threshold the hit was.
      beat = Math.min(1.6, 0.55 + (energy - threshold) / Math.max(threshold, 1e-5) * 0.35);
    }
    this.beatEnv = Math.max(beat, this.beatEnv * Math.exp(-dtSeconds * 7.5));

    return {
      low: this.lowEnv,
      mid: this.midEnv,
      high: this.highEnv,
      beat: this.beatEnv,
      rms: this.rmsEnv,
      active: true,
    };
  }

  dispose(): void {
    this.stop();
    this.synth?.dispose();
    this.synth = null;
    void this.ctx?.close();
    this.ctx = null;
    this.analyser = null;
  }
}

/**
 * The procedural source. A minor-ninth drone: three detuned saw voices through a
 * breathing lowpass, a sub sine pulsing on a slow cycle, and a shimmer partial
 * that only appears on the pulse. Everything is generated; nothing is fetched.
 */
class SynthGraph {
  private ctx: AudioContext;
  private out: GainNode;
  private voices: { osc: OscillatorNode; gain: GainNode; detune: number }[] = [];
  private sub: OscillatorNode | null = null;
  private subGain: GainNode | null = null;
  private shimmer: OscillatorNode | null = null;
  private shimmerGain: GainNode | null = null;
  private filter: BiquadFilterNode;
  private pulseTimer: number | null = null;
  private lfoTimers: number[] = [];
  private root = 55; // A1

  constructor(ctx: AudioContext) {
    this.ctx = ctx;
    this.out = ctx.createGain();
    this.out.gain.value = 0.0001;
    const comp = ctx.createDynamicsCompressor();
    comp.threshold.value = -18;
    comp.knee.value = 24;
    comp.ratio.value = 4;
    this.filter = ctx.createBiquadFilter();
    this.filter.type = 'lowpass';
    this.filter.frequency.value = 420;
    this.filter.Q.value = 6.5;
    this.filter.connect(comp);
    comp.connect(this.out);
  }

  connect(destination: AudioNode): void {
    this.out.connect(destination);
  }

  start(): void {
    const ctx = this.ctx;
    const now = ctx.currentTime;
    // Fade in over 3 s so the visual response ramps rather than jumping.
    this.out.gain.cancelScheduledValues(now);
    this.out.gain.setValueAtTime(Math.max(this.out.gain.value, 0.0001), now);
    this.out.gain.exponentialRampToValueAtTime(0.32, now + 3);

    if (this.voices.length === 0) {
      const ratios = [1, 1.5, 2.25, 3, 4.5]; // root, fifth, ninth, octave+fifth
      const detunes = [-7, 5, -3, 9, -11];
      for (let i = 0; i < ratios.length; i++) {
        const osc = ctx.createOscillator();
        osc.type = i < 3 ? 'sawtooth' : 'triangle';
        osc.frequency.value = this.root * (ratios[i] as number);
        osc.detune.value = detunes[i] as number;
        const gain = ctx.createGain();
        gain.gain.value = 0.14 / (1 + i * 0.4);
        osc.connect(gain);
        gain.connect(this.filter);
        osc.start();
        this.voices.push({ osc, gain, detune: detunes[i] as number });
      }
      // Slow chorus: drifting detune reads as a breathing, living pad.
      const drift = () => {
        const t = this.ctx.currentTime;
        for (const v of this.voices) {
          const target = v.detune + (Math.random() * 2 - 1) * 14;
          v.osc.detune.setTargetAtTime(target, t, 3.5);
        }
      };
      drift();
      this.lfoTimers.push(window.setInterval(drift, 4200));

      // Filter breathing, fast enough to feel alive but not rhythmic.
      const breathe = () => {
        const t = this.ctx.currentTime;
        const f = 260 + Math.random() * 900;
        this.filter.frequency.setTargetAtTime(f, t, 1.6);
        this.filter.Q.setTargetAtTime(4 + Math.random() * 8, t, 2.0);
      };
      breathe();
      this.lfoTimers.push(window.setInterval(breathe, 2600));
    }

    if (!this.sub) {
      this.sub = ctx.createOscillator();
      this.sub.type = 'sine';
      this.sub.frequency.value = this.root * 0.5;
      this.subGain = ctx.createGain();
      this.subGain.gain.value = 0.0001;
      this.sub.connect(this.subGain);
      this.subGain.connect(this.out);
      this.sub.start();

      this.shimmer = ctx.createOscillator();
      this.shimmer.type = 'triangle';
      this.shimmer.frequency.value = this.root * 9.02;
      this.shimmerGain = ctx.createGain();
      this.shimmerGain.gain.value = 0.0001;
      this.shimmer.connect(this.shimmerGain);
      this.shimmerGain.connect(this.out);
      this.shimmer.start();
    }

    // Sub pulse every ~2.6 s: a soft kick that the low band and the beat
    // detector both register clearly.
    const pulse = () => {
      const t = this.ctx.currentTime;
      const g = this.subGain;
      if (g) {
        g.gain.cancelScheduledValues(t);
        g.gain.setValueAtTime(0.0001, t);
        g.gain.exponentialRampToValueAtTime(0.55, t + 0.035);
        g.gain.exponentialRampToValueAtTime(0.0008, t + 0.85);
      }
      const sg = this.shimmerGain;
      if (sg) {
        sg.gain.cancelScheduledValues(t);
        sg.gain.setValueAtTime(0.0001, t);
        sg.gain.exponentialRampToValueAtTime(0.09, t + 0.06);
        sg.gain.exponentialRampToValueAtTime(0.0004, t + 1.6);
      }
      // Root drift keeps the pad from becoming monotonous over long sessions.
      if (this.voices[0] && Math.random() < 0.3) {
        this.root = Math.random() < 0.5 ? 55 : 61.735; // A1 or B1
        const ratios = [1, 1.5, 2.25, 3, 4.5];
        for (let i = 0; i < this.voices.length; i++) {
          this.voices[i]!.osc.frequency.setTargetAtTime(this.root * (ratios[i] as number), t, 2.5);
        }
        if (this.sub) this.sub.frequency.setTargetAtTime(this.root * 0.5, t, 2.5);
      }
    };
    pulse();
    this.pulseTimer = window.setInterval(pulse, 2600);
  }

  stop(): void {
    if (this.pulseTimer !== null) {
      clearInterval(this.pulseTimer);
      this.pulseTimer = null;
    }
    for (const id of this.lfoTimers) clearInterval(id);
    this.lfoTimers = [];
    const t = this.ctx.currentTime;
    this.out.gain.cancelScheduledValues(t);
    this.out.gain.setTargetAtTime(0.0001, t, 0.25);
    // The oscillators keep running (cheap) so restarting is instant and
    // click-free; only the output gain is muted.
  }

  dispose(): void {
    this.stop();
    for (const v of this.voices) {
      try {
        v.osc.stop();
      } catch {
        /* already stopped */
      }
    }
    this.voices = [];
    try {
      this.sub?.stop();
      this.shimmer?.stop();
    } catch {
      /* ignore */
    }
    this.sub = null;
    this.shimmer = null;
    this.out.disconnect();
  }
}
