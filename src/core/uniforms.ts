/**
 * Aetheria — CPU-side writers for the two uniform blocks declared in
 * `src/gpu/wgsl/common.wgsl.ts`. Layouts here are the contract; if you edit a
 * layout you must edit the WGSL struct in the same commit.
 */

import { SIM_UNIFORM_BYTES, POST_UNIFORM_BYTES } from '../gpu/wgsl/common.wgsl';

export { SIM_UNIFORM_BYTES, POST_UNIFORM_BYTES };

/** Byte offsets of every vec4 slot inside the Sim block. */
export const SIM_OFF = {
  time: 0,
  viewport: 16,
  camera: 32,
  slice: 48,
  julia: 64,
  force: 80,
  attract: 96,
  audio: 112,
  palette: 128,
  render: 144,
  proj: 160,
  view: 224,
  rot4: 288,
  rot0: 352,
  rot1: 368,
  misc: 384,
  focus: 400,
  shockMisc: 416,
  misc2: 432,
} as const;

export const POST_OFF = {
  resolution: 0,
  texel: 16,
  params0: 32,
  params1: 48,
  params2: 64,
  focus: 80,
  far: 96,
  color: 112,
} as const;

export interface SimState {
  time: number;
  dt: number;
  frame: number;
  simCount: number;

  width: number;
  height: number;

  eye: [number, number, number];
  zoom: number;

  wSlice: number;
  fov4: number;
  nearBlend: number;
  aspect: number;

  julia: [number, number, number, number];
  curl: number;
  damping: number;
  gravity: number;
  confinement: number;

  attractDepth: number;
  escape: number;
  power: number;
  warp: number;

  audioLow: number;
  audioMid: number;
  audioHigh: number;
  beat: number;

  hue: number;
  exposure: number;

  sizeScale: number;
  energyScale: number;
  multiplicity: number;
  speed: number;

  proj: Float32Array;
  view: Float32Array;
  rot4: Float32Array;

  rot0: [number, number, number, number];
  rot1: [number, number, number, number];

  shockGain: number;
  bloom: number;
  dpr: number;
  /**
   * Power-of-two particle stride for the vertex shader's instance mapping.
   *
   * It occupies the slot the render-scale value used to hold, on purpose: the
   * shader needs a per-frame scalar to decode the instance index without an
   * integer modulo, and the render scale was never read from the GPU. See the
   * mapping comment in `particle.wgsl.ts`.
   */
  stride: number;

  focusDist: number;
  aperture: number;
  grain: number;
  vignette: number;

  activeShocks: number;
  shockRadius: number;
  shockThickness: number;
  shockLife: number;

  trailDecay: number;
  paletteMix: number;
  saturation: number;
  /** Sprite-radius multiplier; 1 normally, tiny during a fill-rate ablation. */
  spriteScale: number;
}

export function makeSimState(): SimState {
  return {
    time: 0,
    dt: 1 / 60,
    frame: 0,
    simCount: 0,
    width: 1,
    height: 1,
    eye: [0, 0, 3.4],
    zoom: 3.4,
    wSlice: 0,
    fov4: 1.9,
    nearBlend: 0.35,
    aspect: 1,
    julia: [-0.42, 0.61, -0.27, 0.35],
    curl: 0.55,
    damping: 0.34,
    gravity: 0.16,
    confinement: 0.34,
    attractDepth: 0.62,
    escape: 2.6,
    power: 3.4,
    warp: 0.55,
    audioLow: 0,
    audioMid: 0,
    audioHigh: 0,
    beat: 0,
    hue: 0,
    exposure: 1.0,
    sizeScale: 1,
    energyScale: 1,
    multiplicity: 1,
    speed: 1,
    proj: new Float32Array(16),
    view: new Float32Array(16),
    rot4: new Float32Array(16),
    rot0: [0, 0, 0, 0],
    rot1: [0, 0, 1.1, 0],
    shockGain: 1,
    bloom: 0.62,
    dpr: 1,
    stride: 1024,
    focusDist: 3.4,
    aperture: 0.45,
    grain: 0.35,
    vignette: 0.45,
    activeShocks: 0,
    shockRadius: 1.2,
    shockThickness: 0.42,
    shockLife: 1.6,
    trailDecay: 0.62,
    paletteMix: 0.55,
    saturation: 1,
    spriteScale: 1,
  };
}

/** Writes `SimState` into a 448-byte ArrayBuffer ready for `writeBuffer`. */
export class SimWriter {
  readonly data = new ArrayBuffer(SIM_UNIFORM_BYTES);
  private readonly f32 = new Float32Array(this.data);

  write(s: SimState): ArrayBuffer {
    const f = this.f32;
    const v4 = (off: number, a: number, b: number, c: number, d: number) => {
      const i = off >> 2;
      f[i] = a;
      f[i + 1] = b;
      f[i + 2] = c;
      f[i + 3] = d;
    };

    v4(SIM_OFF.time, s.time, s.dt, s.frame, s.simCount);
    v4(SIM_OFF.viewport, s.width, s.height, 1 / s.width, 1 / s.height);
    v4(SIM_OFF.camera, s.eye[0], s.eye[1], s.eye[2], s.zoom);
    v4(SIM_OFF.slice, s.wSlice, s.fov4, s.nearBlend, s.aspect);
    v4(SIM_OFF.julia, s.julia[0], s.julia[1], s.julia[2], s.julia[3]);
    v4(SIM_OFF.force, s.curl, s.damping, s.gravity, s.confinement);
    v4(SIM_OFF.attract, s.attractDepth, s.escape, s.power, s.warp);
    v4(SIM_OFF.audio, s.audioLow, s.audioMid, s.audioHigh, s.beat);
    v4(SIM_OFF.palette, s.hue, s.saturation, s.paletteMix, s.exposure);
    v4(SIM_OFF.render, s.sizeScale, s.energyScale, s.multiplicity, s.speed);

    this.blit(s.proj, SIM_OFF.proj);
    this.blit(s.view, SIM_OFF.view);
    this.blit(s.rot4, SIM_OFF.rot4);

    v4(SIM_OFF.rot0, s.rot0[0], s.rot0[1], s.rot0[2], s.rot0[3]);
    v4(SIM_OFF.rot1, s.rot1[0], s.rot1[1], s.rot1[2], s.rot1[3]);
    v4(SIM_OFF.misc, s.shockGain, s.bloom, s.dpr, s.stride);
    v4(SIM_OFF.focus, s.focusDist, s.aperture, s.grain, s.vignette);
    v4(SIM_OFF.shockMisc, s.activeShocks, s.shockRadius, s.shockThickness, s.shockLife);
    v4(SIM_OFF.misc2, s.trailDecay, s.paletteMix, s.saturation, s.spriteScale);

    return this.data;
  }

  /** Patch a single slot without touching the rest (used for shock counters). */
  writeShockMisc(active: number, radius: number, thickness: number, life: number): void {
    const i = SIM_OFF.shockMisc >> 2;
    this.f32[i] = active;
    this.f32[i + 1] = radius;
    this.f32[i + 2] = thickness;
    this.f32[i + 3] = life;
  }

  private blit(src: Float32Array, byteOffset: number): void {
    this.f32.set(src, byteOffset >> 2);
  }
}

export interface PostState {
  width: number;
  height: number;
  texelX: number;
  texelY: number;
  srcTexelX: number;
  srcTexelY: number;
  exposure: number;
  bloom: number;
  bloomRadius: number;
  threshold: number;
  chroma: number;
  grain: number;
  vignette: number;
  dof: number;
  volumetric: number;
  time: number;
  aspect: number;
  frame: number;
  focusDepth: number;
  focusRange: number;
  maxCoc: number;
  nearPlane: number;
  farPlane: number;
  sourceLod: number;
  upsampleRadius: number;
  bloomLevels: number;
  hue: number;
  saturation: number;
  /**
   * Particle-count pre-exposure, written to pColor.z.
   *
   * Additive emission integrates with the number of emitters, so the composite
   * divides by sqrt(N) here; without it a 4M-particle cloud clips to white while a
   * 250k one is nearly black.
   */
  quality: number;
  sceneLuma: number;
}

export class PostWriter {
  readonly data = new ArrayBuffer(POST_UNIFORM_BYTES);
  private readonly f32 = new Float32Array(this.data);

  write(s: PostState): ArrayBuffer {
    const f = this.f32;
    const v4 = (off: number, a: number, b: number, c: number, d: number) => {
      const i = off >> 2;
      f[i] = a;
      f[i + 1] = b;
      f[i + 2] = c;
      f[i + 3] = d;
    };
    v4(POST_OFF.resolution, s.width, s.height, 1 / s.width, 1 / s.height);
    v4(POST_OFF.texel, s.texelX, s.texelY, s.srcTexelX, s.srcTexelY);
    v4(POST_OFF.params0, s.exposure, s.bloom, s.bloomRadius, s.threshold);
    v4(POST_OFF.params1, s.chroma, s.grain, s.vignette, s.dof);
    v4(POST_OFF.params2, s.volumetric, s.time, s.aspect, s.frame);
    v4(POST_OFF.focus, s.focusDepth, s.focusRange, s.maxCoc, s.nearPlane);
    v4(POST_OFF.far, s.farPlane, s.sourceLod, s.upsampleRadius, s.bloomLevels);
    v4(POST_OFF.color, s.hue, s.saturation, s.quality, s.sceneLuma);
    return this.data;
  }
}

export function makePostState(): PostState {
  return {
    width: 1,
    height: 1,
    texelX: 1,
    texelY: 1,
    srcTexelX: 1,
    srcTexelY: 1,
    exposure: 1.0,
    bloom: 0.62,
    bloomRadius: 0.55,
    threshold: 1.0,
    chroma: 0.34,
    grain: 0.35,
    vignette: 0.45,
    dof: 0.45,
    volumetric: 0.4,
    time: 0,
    aspect: 1,
    frame: 0,
    focusDepth: 3.4,
    focusRange: 0.6,
    maxCoc: 8,
    nearPlane: 0.05,
    farPlane: 60,
    sourceLod: 0,
    upsampleRadius: 0.8,
    bloomLevels: 6,
    hue: 0,
    saturation: 1,
    quality: 1,
    sceneLuma: 0.4,
  };
}
