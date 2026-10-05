/**
 * Aetheria — global tunables and device capability detection.
 *
 * Every number that shapes the look of the universe lives here so that the
 * WebGPU renderer, the WebGL2 fallback, the UI panel, the URL-hash serializer
 * and the README all agree on a single source of truth.
 */

export const MAX_RENDER_PARTICLES = 10_000_000;

/** Simulated particles are the memory-bound part; rendered particles may exceed
 *  this via sub-particle scatter (see `multiplicity`). */
export const HARD_MAX_SIM_PARTICLES = 4_000_000;

/** Memory budget for the particle system alone, in bytes (documented in README). */
export const PARTICLE_BYTES_PER_SIM = 88; // 2x(Vec4+Vec4+Vec4+Vec2) = 2x44

export type QualityTier = 'mobile' | 'balanced' | 'high' | 'ultra';

export interface DeviceProfile {
  readonly tier: QualityTier;
  readonly isMobile: boolean;
  readonly isApple: boolean;
  readonly cores: number;
  /** Device pixel ratio; refreshed on resize, so deliberately not readonly. */
  dpr: number;
  readonly simParticles: number;
  readonly showParticles: number;
  readonly renderScale: number;
  readonly maxPostScale: number;
  readonly bloomLevels: number;
  readonly dof: boolean;
  readonly volumetric: boolean;
  readonly grain: boolean;
  readonly supersample: number;
  readonly juliaIterations: number;
}

export function detectProfile(): DeviceProfile {
  const nav = navigator as Navigator & { deviceMemory?: number; userAgentData?: { mobile?: boolean } };
  const ua = nav.userAgent;
  const isApple = /Mac|iPhone|iPad|iPod/.test(ua);
  const isMobile =
    nav.userAgentData?.mobile === true ||
    /Android|iPhone|iPad|iPod|Mobile|Tablet/i.test(ua) ||
    (isApple && navigator.maxTouchPoints > 1 && /Mac/.test(ua));
  const cores = navigator.hardwareConcurrency || 4;
  const dpr = Math.min(window.devicePixelRatio || 1, isMobile ? 2.5 : 2);

  if (isMobile) {
    return {
      tier: 'mobile',
      isMobile,
      isApple,
      cores,
      dpr,
      simParticles: 200_000,
      showParticles: 400_000,
      renderScale: 0.72,
      maxPostScale: 0.7,
      bloomLevels: 4,
      dof: false,
      volumetric: false,
      grain: true,
      supersample: 1,
      juliaIterations: 6,
    };
  }

  const weak = cores <= 4;
  const tier: QualityTier = weak ? 'balanced' : cores >= 12 ? 'ultra' : 'high';
  const presets: Record<Exclude<QualityTier, 'mobile'>, Omit<DeviceProfile, 'tier' | 'isMobile' | 'isApple' | 'cores' | 'dpr'>> = {
    balanced: {
      simParticles: 500_000,
      showParticles: 1_000_000,
      renderScale: 0.8,
      maxPostScale: 0.8,
      bloomLevels: 5,
      dof: true,
      volumetric: true,
      grain: true,
      supersample: 1,
      juliaIterations: 8,
    },
    high: {
      simParticles: 1_000_000,
      showParticles: 2_000_000,
      renderScale: 0.9,
      maxPostScale: 1,
      bloomLevels: 6,
      dof: true,
      volumetric: true,
      grain: true,
      supersample: 1.4,
      juliaIterations: 10,
    },
    ultra: {
      simParticles: 1_500_000,
      showParticles: 4_000_000,
      renderScale: 1,
      maxPostScale: 1,
      bloomLevels: 6,
      dof: true,
      volumetric: true,
      grain: true,
      supersample: 1.6,
      juliaIterations: 12,
    },
  };
  return { tier, isMobile, isApple, cores, dpr, ...presets[tier] };
}

/**
 * User-facing parameters. Persisted to the URL hash so a universe state is a
 * shareable link (`#a=0.31,0.02,...&n=2000000&...`).
 */
export interface Params {
  /** Simulated particle count (memory bound). */
  simCount: number;
  /** Rendered particle count, may exceed simCount via multiplicity. */
  showCount: number;
  /** Simulation time scale. */
  speed: number;
  /** 4D Julia set parameter, each in [-1.6, 1.6]. */
  julia: [number, number, number, number];
  /** Attractor / escape radius, the "fractal dimension" knob. */
  fractalDim: number;
  /** Curl-noise advection strength. */
  curl: number;
  /** Velocity damping per second. */
  damping: number;
  /** Inverse-square pull toward origin. */
  gravity: number;
  /** Confinement spring that keeps the nebula inside its shell. */
  confinement: number;
  /** Shockwave impulse strength (0 disables clicks). */
  shock: number;
  /** HDR exposure applied before ACES. */
  exposure: number;
  /** Bloom intensity. */
  bloom: number;
  /** Bloom radius bias, 0..1. */
  bloomRadius: number;
  /**
   * Linear HDR luminance above which a pixel contributes to bloom.
   *
   * This has to be tuned together with the exposure: it is deliberately well
   * above 1.0, because a dense particle cloud's *haze* sits near the exposure
   * level and only the filament cores and hero stars should halo. A threshold at
   * 1.0 turns the whole nebula into a uniform glow.
   */
  bloomThreshold: number;
  /** Depth-of-field strength. */
  dof: number;
  /** Chromatic aberration in pixels at the frame edge. */
  chroma: number;
  /** Volumetric light (god-ray) intensity. */
  volumetric: number;
  /** Film grain amount. */
  grain: number;
  /** Vignette amount. */
  vignette: number;
  /**
   * Sprite radius multiplier.
   *
   * This is the single most effective control over how the nebula *reads*: the
   * particles are additive, so at small radii a million of them are isolated
   * points and the cloud looks like sparse coloured dust, while at larger radii
   * they overlap into the continuous luminous gas that makes the structure
   * visible at all.
   */
  particleSize: number;
  /** Audio reactivity multiplier. */
  audioSensitivity: number;
  /** Trail persistence 0..0.98 (frame-to-frame history feedback). */
  trails: number;
  /** Palette rotation in turns. */
  hue: number;
  paused: boolean;
  /** Render scale multiplier applied on top of the device profile. */
  resolution: number;
}

export function defaultParams(p: DeviceProfile): Params {
  return {
    simCount: p.simParticles,
    showCount: p.showParticles,
    speed: 1,
    julia: [-0.42, 0.61, -0.27, 0.35],
    fractalDim: 0.72,
    curl: 1.1,
    damping: 0.3,
    gravity: 0.2,
    confinement: 0.62,
    shock: 1,
    exposure: 1.0,
    bloom: 0.62,
    bloomRadius: 0.55,
    bloomThreshold: 2.4,
    dof: p.dof ? 0.5 : 0,
    chroma: 0.34,
    volumetric: p.volumetric ? 0.45 : 0,
    grain: p.grain ? 0.35 : 0,
    vignette: 0.45,
    particleSize: 1.5,
    audioSensitivity: 1,
    trails: 0.8,
    hue: 0,
    paused: false,
    resolution: 1,
  };
}

export const PARAM_RANGES = {
  simCount: { min: 100_000, max: HARD_MAX_SIM_PARTICLES, step: 50_000 },
  showCount: { min: 100_000, max: MAX_RENDER_PARTICLES, step: 250_000 },
  speed: { min: 0, max: 3, step: 0.01 },
  fractalDim: { min: 0, max: 1, step: 0.01 },
  curl: { min: 0, max: 2, step: 0.01 },
  damping: { min: 0.02, max: 1.5, step: 0.01 },
  gravity: { min: 0, max: 0.6, step: 0.01 },
  confinement: { min: 0, max: 1.5, step: 0.01 },
  shock: { min: 0, max: 3, step: 0.05 },
  exposure: { min: 0.2, max: 3, step: 0.01 },
  bloom: { min: 0, max: 2, step: 0.01 },
  bloomRadius: { min: 0, max: 1, step: 0.01 },
  bloomThreshold: { min: 0.2, max: 8, step: 0.05 },
  dof: { min: 0, max: 1.5, step: 0.01 },
  chroma: { min: 0, max: 2, step: 0.01 },
  volumetric: { min: 0, max: 1.5, step: 0.01 },
  grain: { min: 0, max: 1, step: 0.01 },
  vignette: { min: 0, max: 1, step: 0.01 },
  particleSize: { min: 0.4, max: 5, step: 0.05 },
  audioSensitivity: { min: 0, max: 3, step: 0.01 },
  trails: { min: 0, max: 0.97, step: 0.01 },
  hue: { min: 0, max: 1, step: 0.005 },
  resolution: { min: 0.4, max: 1.4, step: 0.02 },
} as const satisfies Record<string, { min: number; max: number; step: number }>;

export type RangeKey = keyof typeof PARAM_RANGES;

export function clampParam(key: RangeKey, value: number): number {
  const r = PARAM_RANGES[key];
  if (!Number.isFinite(value)) return r.min;
  return Math.min(r.max, Math.max(r.min, value));
}
