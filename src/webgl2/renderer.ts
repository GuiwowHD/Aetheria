/**
 * Aetheria — WebGL2 fallback renderer.
 *
 * Implements the backend-agnostic `Renderer` contract with a transform-feedback
 * particle simulation and an HDR post chain, for the machines where WebGPU is
 * unavailable. The parameter vocabulary is shared with the WebGPU backend: this
 * class feeds `SimWriter`/`PostWriter` from `src/core/uniforms.ts` and uploads
 * their buffers verbatim, because the std140 layouts they produce are already
 * byte-compatible with the GLSL uniform blocks in `./shaders.ts`. One state, two
 * backends, no duplicated knob list.
 *
 * Pipeline per frame:
 *   1. one transform-feedback sim step (ping-pong VBO pairs, skipped when paused)
 *   2. scene pass: trail feedback -> depth prepass (only when DOF needs depth)
 *      -> additive instanced billboards, into an RGBA16F (or RGBA8) target
 *   3. post: bright pass -> 5-6 level downsample/upsample bloom -> 24-tap
 *      volumetric rays -> golden-angle DOF -> composite (ACES, grain, vignette,
 *      dither, sRGB) -> FXAA to the default framebuffer
 *
 * No allocation happens in the steady-state `frame()` path: every typed array,
 * uniform array, matrix and render target is created in `init()`/`resize()`.
 */

import { BackendUnavailable, estimateParticleMemory } from '../core/types';
import type { BackendKind, HudStats, Renderer, RendererTelemetry } from '../core/types';
import type { DeviceProfile, Params } from '../core/config';
import {
  TAU,
  clamp as clampNum,
  invert4,
  lookAt,
  mul4,
  mulMat4,
  perspective,
  rotationMatrix4,
} from '../core/math4d';
import { makePostState, makeSimState, PostWriter, SimWriter } from '../core/uniforms';
import type { PostState, SimState } from '../core/uniforms';
import {
  bindAttributeBuffer,
  createFullscreenVao,
  createProgram,
  createRenderTarget,
  deleteRenderTarget,
  getDeviceLabel,
  probeFloatColorBuffer,
} from './glutil';
import type { FloatProbe, GlProgram, RenderTarget } from './glutil';
import {
  BLOOM_DOWN_FS,
  BLOOM_UP_FS,
  BRIGHT_FS,
  COMPOSITE_FS,
  DEPTH_FS,
  DEPTH_VS,
  DOF_GATHER_FS,
  DOF_PREPARE_FS,
  FULLSCREEN_VS,
  FXAA_FS,
  PARTICLE_FS,
  PARTICLE_VS,
  SIM_FS,
  SIM_TF_VARYINGS,
  SIM_VS,
  TRAIL_FS,
  VOLUMETRIC_FS,
} from './shaders';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const FOV_Y = 1.0;
const NEAR_PLANE = 0.05;
const FAR_PLANE = 60;
const DEFAULT_ZOOM = 3.4;
const MIN_ZOOM = 1.4;
const MAX_ZOOM = 12;
const MAX_SLICE = 3;
const MAX_DPR = 3;
/** `showCount` may exceed the simulated count by at most this factor. */
const MAX_MULTIPLICITY_PASSES = 4;
const MAX_SHOCKS = 8;
const SHOCK_LIFE = 1.9;
const MAX_BLOOM_LEVELS = 6;
/** 4 components x 4 bytes: one vec4 per particle. */
const VEC4_BYTES = 16;

const CAMERA_TARGET: readonly number[] = [0, 0, 0];
const CAMERA_UP: readonly number[] = [0, 1, 0];
const DEFAULT_ANGLES: [number, number, number, number, number, number] = [
  0.32, -0.24, 0.55, 0.21, -0.38, 0.27,
];

/** Cosmology ramp, mirrored from `palette()` in ./shaders.ts. */
const PALETTE_STOPS = new Float32Array([
  0.62, 0.06, 0.94, // magenta
  0.05, 0.86, 0.98, // cyan
  1.00, 0.72, 0.16, // gold
  0.48, 0.16, 0.92, // deep purple
]);

// ---------------------------------------------------------------------------
// Deterministic helpers (bit-identical to the GLSL/WGSL hash used on the GPU)
// ---------------------------------------------------------------------------

/** PCG-style 3-round mix over uint32 lanes; mirrors `hash3()` in the shaders. */
function hash3u(x: number, y: number, z: number): number {
  let px = (Math.imul(x, 1664525) + 1013904223) >>> 0;
  let py = (Math.imul(y, 1664525) + 1013904223) >>> 0;
  let pz = (Math.imul(z, 1664525) + 1013904223) >>> 0;
  px = (px + Math.imul(py, pz)) >>> 0;
  py = (py + Math.imul(pz, px)) >>> 0;
  pz = (pz + Math.imul(px, py)) >>> 0;
  px = (px ^ (px >>> 16)) >>> 0;
  py = (py ^ (py >>> 16)) >>> 0;
  pz = (pz ^ (pz >>> 16)) >>> 0;
  px = (px + Math.imul(py, pz)) >>> 0;
  py = (py + Math.imul(pz, px)) >>> 0;
  pz = (pz + Math.imul(px, py)) >>> 0;
  return (px ^ py ^ pz) >>> 0;
}

function hash1u(n: number): number {
  return hash3u(n >>> 0, Math.imul(n, 0x9e3779b9) >>> 0, 0x85ebca6b) >>> 0;
}

function rnd1u(seed: number): number {
  return hash1u(seed) / 4294967296;
}

function rndRangeu(seed: number, stream: number, lo: number, hi: number): number {
  const s = (Math.imul(seed, 747796405) + Math.imul(stream, 2891336453) + 1) >>> 0;
  return lo + (hi - lo) * rnd1u(s);
}

function writePalette(t: number, dst: Float32Array, off: number): void {
  const x = (t - Math.floor(t)) * 4;
  const i = Math.min(3, Math.max(0, Math.floor(x)));
  const f = x - i;
  const s = f * f * (3 - 2 * f);
  const a = i * 3;
  const b = ((i + 1) & 3) * 3;
  for (let c = 0; c < 3; c++) {
    const lo = PALETTE_STOPS[a + c];
    const hi = PALETTE_STOPS[b + c];
    dst[off + c] = lo + (hi - lo) * s;
  }
}

/** Mirrors `planeRotation()` in src/core/math4d.ts without allocating. */
function planeRotationInto(i: number, j: number, a: number, m: Float64Array): void {
  m.fill(0);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
  const c = Math.cos(a);
  const s = Math.sin(a);
  m[i * 4 + i] = c;
  m[j * 4 + j] = c;
  m[i * 4 + j] = -s;
  m[j * 4 + i] = s;
}

function num(v: number, fallback: number): number {
  return Number.isFinite(v) ? v : fallback;
}

function finiteCount(v: number): number {
  return Number.isFinite(v) ? Math.max(0, Math.floor(v)) : 0;
}

function wrapAngle(a: number): number {
  if (!Number.isFinite(a)) return 0;
  return a - TAU * Math.round(a / TAU);
}

/** Accepts either raw wheel deltas (~+/-100 per notch) or normalised (~+/-1). */
function wheelNotches(delta: number): number {
  if (!Number.isFinite(delta)) return 0;
  return Math.abs(delta) > 8 ? delta / 100 : delta;
}

// ---------------------------------------------------------------------------
// GPU resource bundles (created in init()/resize(), never inside frame())
// ---------------------------------------------------------------------------

/** The ping-pong particle buffers plus the two VAO flavours that read them. */
interface ParticleData {
  bufPos: [WebGLBuffer, WebGLBuffer];
  bufVel: [WebGLBuffer, WebGLBuffer];
  bufMeta: [WebGLBuffer, WebGLBuffer];
  /** Static per-particle birth colour + energy, never written by feedback. */
  bufColor: WebGLBuffer;
  /** Transform-feedback source VAOs: attributes advance per vertex. */
  vaoSim: [WebGLVertexArrayObject, WebGLVertexArrayObject];
  /** Draw VAOs: the same buffers with an instance divisor of 1. */
  vaoDraw: [WebGLVertexArrayObject, WebGLVertexArrayObject];
}

interface ParticleBuffers {
  capacity: number;
  simUbo: WebGLBuffer;
  postUbo: WebGLBuffer;
  quadVao: WebGLVertexArrayObject;
  data: ParticleData;
}

interface Programs {
  sim: GlProgram;
  particle: GlProgram;
  depth: GlProgram;
  trail: GlProgram;
  bright: GlProgram;
  down: GlProgram;
  up: GlProgram;
  volumetric: GlProgram;
  dofPrepare: GlProgram;
  dofGather: GlProgram;
  composite: GlProgram;
  fxaa: GlProgram;
}

interface Framebuffers {
  scene: RenderTarget;
  trail: RenderTarget;
  ldr: RenderTarget;
  vol: RenderTarget;
  dofHalf: RenderTarget;
  dofFull: RenderTarget;
  bloom: RenderTarget[];
}

// ---------------------------------------------------------------------------

export class WebGL2Renderer implements Renderer {
  readonly kind: BackendKind = 'webgl2';
  readonly profile: DeviceProfile;
  readonly params: Params;
  readonly stats: HudStats;

  private readonly canvas: HTMLCanvasElement;
  private readonly telemetry: RendererTelemetry;

  private gl: WebGL2RenderingContext | null = null;
  private probe: FloatProbe | null = null;
  private progs: Programs | null = null;
  private gpu: ParticleBuffers | null = null;
  private targets: Framebuffers | null = null;
  private disposed = false;
  private contextLost = false;

  private readonly sim: SimState;
  private readonly post: PostState;
  private readonly simWriter = new SimWriter();
  private readonly postWriter = new PostWriter();

  // Camera + 4D rotation state.
  private readonly view = new Float32Array(16);
  private readonly proj = new Float32Array(16);
  private readonly mvp = new Float32Array(16);
  private readonly invMvp = new Float32Array(16);
  private readonly eye: [number, number, number] = [0, 0, DEFAULT_ZOOM];
  private readonly angles: [number, number, number, number, number, number] = [...DEFAULT_ANGLES];
  private readonly rotPlane = new Float64Array(16);
  private readonly rotAcc = new Float64Array(16);
  private readonly rotTmp = new Float64Array(16);
  private readonly rotOut = new Float64Array(16);
  private readonly rayA = new Float64Array(3);
  private readonly rayB = new Float64Array(3);
  private rotFastPath = true;
  private rotDirty = true;
  private camDirty = true;
  private camDist = DEFAULT_ZOOM;
  private wSlice = 0;

  // Frame/size bookkeeping.
  private ping = 0;
  private activeCount = 0;
  private drawCount = 0;
  private passes = 0;
  private fboBytes = 0;
  private qualityScale = 1;
  /**
   * Enables the composite's diagnostic tap. Only ever true under ?selftest=1, so
   * the uniform write it guards does not run in a normal session.
   */
  private readonly selfTestEnabled = typeof window !== 'undefined' && new URLSearchParams(window.location.search).has('selftest');
  private cssW = 1;
  private cssH = 1;
  private dpr = 1;
  private renderScale = 1;
  private drawW = 1;
  private drawH = 1;
  private sceneW = 1;
  private sceneH = 1;
  private sceneAspect = 1;
  private nowSeconds = 0;
  /** Largest renderable dimension; keeps extreme dpr x supersample configs alive. */
  private maxTextureSize = 8192;

  // Audio reactivity.
  private audioLow = 0;
  private audioMid = 0;
  private audioHigh = 0;
  private beat = 0;
  private audioPulse = 0;
  private lastBeatShock = -10;
  private autoShockIndex = 0;

  // Supernova ring buffer (uploaded as `vec4 shocks[8]` + `float uShockAges[8]`).
  private readonly shockOrigin = new Float32Array(MAX_SHOCKS * 4);
  private readonly shockAges = new Float32Array(MAX_SHOCKS);
  private shockCursor = 0;
  private shockWritten = 0;

  private readonly onContextLost = (): void => {
    this.contextLost = true;
    console.error('[webgl2] WebGL2 context lost — the fallback renderer is inert until reload.');
    this.telemetry.onFatal?.('WebGL2 context lost', 'The GPU reset or dropped the context.');
  };

  constructor(
    canvas: HTMLCanvasElement,
    profile: DeviceProfile,
    params: Params,
    telemetry: RendererTelemetry,
  ) {
    this.canvas = canvas;
    this.profile = profile;
    this.params = params;
    this.telemetry = telemetry;

    this.sim = makeSimState();
    this.post = makePostState();
    // Alias the camera vector so SimWriter always sees the live eye position.
    this.sim.eye = this.eye;
    this.sim.zoom = this.camDist;

    this.stats = {
      fps: 0,
      frameMs: 0,
      cpuMs: 0,
      gpuMs: 0,
      // WebGL2 has no timer-query support on this path, so the compute stage is
      // reported as unmeasurable rather than guessed at.
      computeMs: 0,
      simCount: 0,
      renderCount: 0,
      // The WebGL2 path has no radius-derived budget, so it never culls.
      culledCount: 0,
      drawVertices: 0,
      renderScale: 1,
      drawWidth: 1,
      drawHeight: 1,
      backend: 'webgl2',
      deviceLabel: 'WebGL2',
      gpuTimingSupported: false,
      memoryEstimateMB: 0,
      degraded: [],
    };
  }

  // -------------------------------------------------------------------------
  // Boot
  // -------------------------------------------------------------------------

  async init(canvas: HTMLCanvasElement): Promise<void> {
    if (this.gl) return;

    const attributes: WebGLContextAttributes = {
      alpha: false,
      antialias: false,
      depth: true,
      stencil: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
      desynchronized: false,
    };
    const ctx = canvas.getContext('webgl2', attributes) as WebGL2RenderingContext | null;
    if (!ctx) {
      throw new BackendUnavailable(
        'WebGL2 is unavailable',
        'canvas.getContext("webgl2") returned null — no GPU fallback exists on this device.',
      );
    }
    this.gl = ctx;

    const probe = probeFloatColorBuffer(ctx);
    this.probe = probe;
    this.stats.gpuTimingSupported = false;
    this.stats.gpuMs = 0;
    this.stats.deviceLabel = getDeviceLabel(ctx);
    const maxTexture = ctx.getParameter(ctx.MAX_TEXTURE_SIZE) as number;
    this.maxTextureSize = Number.isFinite(maxTexture) && maxTexture > 0 ? Math.floor(maxTexture) : 8192;
    if (probe.degradedReason) {
      this.stats.degraded.push(probe.degradedReason);
      this.telemetry.onDegrade?.(probe.degradedReason);
    }

    this.rotFastPath = this.verifyRotationPath();
    this.progs = this.createPrograms(ctx);

    // Uniform buffers: the std140 layouts written by SimWriter/PostWriter.
    const simUbo = this.createUniformBuffer();
    const postUbo = this.createUniformBuffer();
    ctx.bindBuffer(ctx.UNIFORM_BUFFER, simUbo);
    ctx.bufferData(ctx.UNIFORM_BUFFER, this.simWriter.data.byteLength, ctx.DYNAMIC_DRAW);
    ctx.bindBufferBase(ctx.UNIFORM_BUFFER, 0, simUbo);
    ctx.bindBuffer(ctx.UNIFORM_BUFFER, postUbo);
    ctx.bufferData(ctx.UNIFORM_BUFFER, this.postWriter.data.byteLength, ctx.DYNAMIC_DRAW);
    ctx.bindBufferBase(ctx.UNIFORM_BUFFER, 1, postUbo);
    ctx.bindBuffer(ctx.UNIFORM_BUFFER, null);

    // Allocate + seed the initial particle system at params.simCount.
    const capacity = Math.max(1, finiteCount(this.params.simCount));
    this.gpu = {
      capacity,
      simUbo,
      postUbo,
      quadVao: createFullscreenVao(ctx),
      data: this.createParticleData(capacity),
    };
    this.seedParticles(capacity);
    this.activeCount = 0;

    // Prime the shock uniforms so a paused first frame still has valid values.
    const sim = this.progs.sim;
    ctx.useProgram(sim.handle);
    ctx.uniform4fv(sim.loc('uShocks[0]'), this.shockOrigin);
    ctx.uniform1fv(sim.loc('uShockAges[0]'), this.shockAges);
    ctx.useProgram(null);

    canvas.addEventListener('webglcontextlost', this.onContextLost, false);

    const w = canvas.clientWidth || canvas.width || 960;
    const h = canvas.clientHeight || canvas.height || 540;
    this.resize(w, h, this.profile.dpr, this.qualityScale);

    const err = ctx.getError();
    if (err !== ctx.NO_ERROR) {
      console.error(`[webgl2] GL error during init: 0x${err.toString(16)}`);
    }

    const suffix = probe.degradedReason ? ` (degraded: ${probe.degradedReason})` : ' HDR';
    console.info(`[webgl2] Aetheria fallback ready on ${this.stats.deviceLabel} —${suffix} pipeline.`);
  }

  private createPrograms(gl: WebGL2RenderingContext): Programs {
    const floatOk = this.probe ? this.probe.floatRenderable : false;

    const sim = createProgram(gl, SIM_VS, SIM_FS, 'sim', SIM_TF_VARYINGS);
    sim.require(['uShocks[0]', 'uShockAges[0]', 'uJuliaIter']);
    gl.useProgram(sim.handle);
    gl.uniform1i(sim.loc('uJuliaIter'), Math.max(1, Math.min(16, this.profile.juliaIterations | 0)));

    const particle = createProgram(gl, PARTICLE_VS, PARTICLE_FS, 'particle');
    particle.require(['uMultiplicityPass', 'uAudioPulse', 'uLdr']);
    gl.useProgram(particle.handle);
    gl.uniform1f(particle.loc('uLdr'), floatOk ? 0 : 1);

    const depth = createProgram(gl, DEPTH_VS, DEPTH_FS, 'depth');
    depth.require(['uMultiplicityPass', 'uAudioPulse']);

    const trail = createProgram(gl, FULLSCREEN_VS, TRAIL_FS, 'trail');
    trail.require(['uPrev']);
    gl.useProgram(trail.handle);
    gl.uniform1i(trail.loc('uPrev'), 0);

    const bright = createProgram(gl, FULLSCREEN_VS, BRIGHT_FS, 'bright');
    bright.require(['uSrc']);
    gl.useProgram(bright.handle);
    gl.uniform1i(bright.loc('uSrc'), 0);

    const down = createProgram(gl, FULLSCREEN_VS, BLOOM_DOWN_FS, 'bloomDown');
    down.require(['uSrc']);
    gl.useProgram(down.handle);
    gl.uniform1i(down.loc('uSrc'), 0);

    const up = createProgram(gl, FULLSCREEN_VS, BLOOM_UP_FS, 'bloomUp');
    up.require(['uSrc']);
    gl.useProgram(up.handle);
    gl.uniform1i(up.loc('uSrc'), 0);

    const volumetric = createProgram(gl, FULLSCREEN_VS, VOLUMETRIC_FS, 'volumetric');
    volumetric.require(['uSrc']);
    gl.useProgram(volumetric.handle);
    gl.uniform1i(volumetric.loc('uSrc'), 0);

    const dofPrepare = createProgram(gl, FULLSCREEN_VS, DOF_PREPARE_FS, 'dofPrepare');
    dofPrepare.require(['uColor', 'uDepth']);
    gl.useProgram(dofPrepare.handle);
    gl.uniform1i(dofPrepare.loc('uColor'), 0);
    gl.uniform1i(dofPrepare.loc('uDepth'), 1);

    const dofGather = createProgram(gl, FULLSCREEN_VS, DOF_GATHER_FS, 'dofGather');
    dofGather.require(['uSrc']);
    gl.useProgram(dofGather.handle);
    gl.uniform1i(dofGather.loc('uSrc'), 0);

    const composite = createProgram(gl, FULLSCREEN_VS, COMPOSITE_FS, 'composite');
    composite.require(['uScene', 'uBloom', 'uVolumetric', 'uDof', 'uAces']);
    gl.useProgram(composite.handle);
    gl.uniform1i(composite.loc('uScene'), 0);
    gl.uniform1i(composite.loc('uBloom'), 1);
    gl.uniform1i(composite.loc('uVolumetric'), 2);
    gl.uniform1i(composite.loc('uDof'), 3);
    gl.uniform1f(composite.loc('uAces'), floatOk ? 1 : 0);

    const fxaa = createProgram(gl, FULLSCREEN_VS, FXAA_FS, 'fxaa');
    fxaa.require(['uSrc']);
    gl.useProgram(fxaa.handle);
    gl.uniform1i(fxaa.loc('uSrc'), 0);

    gl.useProgram(null);
    return {
      sim, particle, depth, trail, bright, down, up, volumetric,
      dofPrepare, dofGather, composite, fxaa,
    };
  }

  /**
   * Cheap one-off cross-check that the allocation-free rotation builder matches
   * `rotationMatrix4()` from src/core/math4d.ts. On the (unexpected) failure we
   * fall back to the shared implementation so the two backends cannot diverge
   * visually — correctness beats a few small temporaries.
   */
  private verifyRotationPath(): boolean {
    const ref = rotationMatrix4(this.angles);
    this.composeRotation(this.rotOut);
    for (let i = 0; i < 16; i++) {
      if (Math.abs(this.rotOut[i] - ref[i]) > 1e-6) {
        console.error('[webgl2] 4D rotation fast path disagrees with rotationMatrix4(); using the shared builder.');
        return false;
      }
    }
    return true;
  }

  /** out = Rzw . Ryw . Rxw . Ryz . Rxz . Rxy (allocation-free). */
  private composeRotation(out: Float64Array): void {
    const a = this.angles;
    const p = this.rotPlane;
    const m = this.rotAcc;
    const t = this.rotTmp;
    planeRotationInto(0, 1, a[0], p);
    m.set(p);
    planeRotationInto(0, 2, a[1], p);
    mul4(p, m, t);
    m.set(t);
    planeRotationInto(0, 3, a[2], p);
    mul4(p, m, t);
    m.set(t);
    planeRotationInto(1, 2, a[3], p);
    mul4(p, m, t);
    m.set(t);
    planeRotationInto(1, 3, a[4], p);
    mul4(p, m, t);
    m.set(t);
    planeRotationInto(2, 3, a[5], p);
    mul4(p, m, t);
    out.set(t);
  }

  // -------------------------------------------------------------------------
  // Resource creation
  // -------------------------------------------------------------------------

  private createUniformBuffer(): WebGLBuffer {
    const gl = this.gl as WebGL2RenderingContext;
    const buf = gl.createBuffer();
    if (!buf) throw new BackendUnavailable('WebGL2: gl.createBuffer() returned null (uniform block)');
    return buf;
  }

  private createVao(): WebGLVertexArrayObject {
    const gl = this.gl as WebGL2RenderingContext;
    const vao = gl.createVertexArray();
    if (!vao) throw new BackendUnavailable('WebGL2: gl.createVertexArray() returned null');
    return vao;
  }

  private createDataBuffer(bytes: number): WebGLBuffer {
    const gl = this.gl as WebGL2RenderingContext;
    const buf = gl.createBuffer();
    if (!buf) throw new BackendUnavailable('WebGL2: gl.createBuffer() returned null (particle data)');
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    // DYNAMIC_COPY: written by transform feedback, read as vertex data.
    gl.bufferData(gl.ARRAY_BUFFER, bytes, gl.DYNAMIC_COPY);
    return buf;
  }

  private initVao(
    vao: WebGLVertexArrayObject,
    pos: WebGLBuffer,
    vel: WebGLBuffer,
    meta: WebGLBuffer,
    color: WebGLBuffer,
    divisor: number,
  ): void {
    const gl = this.gl as WebGL2RenderingContext;
    gl.bindVertexArray(vao);
    bindAttributeBuffer(gl, 0, pos, divisor);
    bindAttributeBuffer(gl, 1, vel, divisor);
    bindAttributeBuffer(gl, 2, meta, divisor);
    bindAttributeBuffer(gl, 3, color, divisor);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
  }

  /** Allocate a fresh ping-pong pair + both VAO flavours at `capacity`. */
  private createParticleData(capacity: number): ParticleData {
    const bytes = capacity * VEC4_BYTES;
    const bufPos: [WebGLBuffer, WebGLBuffer] = [this.createDataBuffer(bytes), this.createDataBuffer(bytes)];
    const bufVel: [WebGLBuffer, WebGLBuffer] = [this.createDataBuffer(bytes), this.createDataBuffer(bytes)];
    const bufMeta: [WebGLBuffer, WebGLBuffer] = [this.createDataBuffer(bytes), this.createDataBuffer(bytes)];
    const bufColor = this.createDataBuffer(bytes);
    const vaoSim: [WebGLVertexArrayObject, WebGLVertexArrayObject] = [this.createVao(), this.createVao()];
    const vaoDraw: [WebGLVertexArrayObject, WebGLVertexArrayObject] = [this.createVao(), this.createVao()];
    for (let i = 0; i < 2; i++) {
      // Same buffers, two divisors: 0 for the feedback pass, 1 for the draw pass.
      this.initVao(vaoSim[i], bufPos[i], bufVel[i], bufMeta[i], bufColor, 0);
      this.initVao(vaoDraw[i], bufPos[i], bufVel[i], bufMeta[i], bufColor, 1);
    }
    return { bufPos, bufVel, bufMeta, bufColor, vaoSim, vaoDraw };
  }

  private deleteParticleData(data: ParticleData): void {
    const gl = this.gl;
    if (!gl) return;
    gl.bindVertexArray(null);
    for (let i = 0; i < 2; i++) {
      gl.deleteVertexArray(data.vaoSim[i]);
      gl.deleteVertexArray(data.vaoDraw[i]);
      gl.deleteBuffer(data.bufPos[i]);
      gl.deleteBuffer(data.bufVel[i]);
      gl.deleteBuffer(data.bufMeta[i]);
    }
    gl.deleteBuffer(data.bufColor);
  }

  /**
   * (Re)allocate the ping-pong particle buffers at `count` particles and reseed
   * every pair. Only called on init, `reset()` and when `simCount` grows past the
   * current capacity.
   */
  private allocateParticles(count: number): void {
    const gpu = this.gpu;
    if (!gpu || !this.gl) return;
    const capacity = Math.max(1, Math.floor(count));
    this.deleteParticleData(gpu.data);
    gpu.data = this.createParticleData(capacity);
    gpu.capacity = capacity;
    this.ping = 0;
    this.activeCount = Math.min(this.activeCount, capacity);
    this.seedParticles(capacity);
  }

  /**
   * Generate deterministic birth state in JS (same hash as the respawn path in
   * SIM_VS, so `reset()` produces the state the GPU would have produced) and
   * upload it to both ping-pong pairs in bounded chunks — a 4M particle config
   * would otherwise need a 256 MB transient scratch buffer.
   */
  private seedParticles(capacity: number): void {
    const gl = this.gl;
    const gpu = this.gpu;
    if (!gl || !gpu) return;
    const data = gpu.data;
    const chunk = Math.min(262144, capacity);
    const pos = new Float32Array(chunk * 4);
    const vel = new Float32Array(chunk * 4);
    const meta = new Float32Array(chunk * 4);
    const color = new Float32Array(chunk * 4);

    for (let base = 0; base < capacity; base += chunk) {
      const n = Math.min(chunk, capacity - base);
      this.fillSeedChunk(base, n, pos, vel, meta, color);
      const byteLength = n * VEC4_BYTES;
      const byteOffset = base * VEC4_BYTES;
      for (let i = 0; i < 2; i++) {
        this.uploadRange(data.bufPos[i], pos, byteOffset, byteLength);
        this.uploadRange(data.bufVel[i], vel, byteOffset, byteLength);
        this.uploadRange(data.bufMeta[i], meta, byteOffset, byteLength);
      }
      this.uploadRange(data.bufColor, color, byteOffset, byteLength);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
  }

  private uploadRange(buffer: WebGLBuffer, data: Float32Array, byteOffset: number, byteLength: number): void {
    const gl = this.gl as WebGL2RenderingContext;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, byteOffset, data, 0, byteLength / 4);
  }

  private fillSeedChunk(
    base: number,
    n: number,
    pos: Float32Array,
    vel: Float32Array,
    meta: Float32Array,
    color: Float32Array,
  ): void {
    for (let i = 0; i < n; i++) {
      const index = base + i;
      const seed = hash1u((Math.imul(index, 2654435761) + 0x9e3779b9) >>> 0);
      const o = i * 4;

      const a = rndRangeu(seed, 0, 0, TAU);
      const b = rndRangeu(seed, 1, -1, 1);
      const c = rndRangeu(seed, 2, 0, TAU);
      const s = Math.sqrt(Math.max(0, 1 - b * b));
      const radius = rndRangeu(seed, 4, 0.9, 2.6);
      pos[o] = s * Math.cos(a) * radius;
      pos[o + 1] = s * Math.sin(a) * radius;
      pos[o + 2] = s * Math.cos(c) * radius;
      pos[o + 3] = b * radius;

      const a2 = rndRangeu(seed, 11, 0, TAU);
      const b2 = rndRangeu(seed, 12, -1, 1);
      const c2 = rndRangeu(seed, 13, 0, TAU);
      const s2 = Math.sqrt(Math.max(0, 1 - b2 * b2));
      const speed = rndRangeu(seed, 14, 0.05, 0.4);
      vel[o] = s2 * Math.cos(a2) * speed;
      vel[o + 1] = s2 * Math.sin(a2) * speed;
      vel[o + 2] = s2 * Math.cos(c2) * speed;
      vel[o + 3] = b2 * speed;

      const lifespan = Math.max(0.25, rndRangeu(seed, 5, 2.5, 7.5));
      const phase = rndRangeu(seed, 7, 0, 1);
      // Stagger the initial ages so no death/birth wave is visible at startup.
      meta[o] = rndRangeu(seed, 9, 0, 0.85) * lifespan;
      meta[o + 1] = lifespan;
      meta[o + 2] = seed & 0x00ffffff;
      meta[o + 3] = phase;

      writePalette(phase, color, o);
      color[o + 3] = rndRangeu(seed, 17, 0.25, 1.0);
    }
  }

  private createTargets(): void {
    const gl = this.gl;
    const probe = this.probe;
    if (!gl || !probe) return;
    this.disposeTargets();

    const scene = this.sceneW;
    const sceneH = this.sceneH;
    const levels = Math.max(1, Math.min(MAX_BLOOM_LEVELS, Math.floor(num(this.profile.bloomLevels, 5))));

    const sceneTarget = createRenderTarget(gl, {
      width: scene,
      height: sceneH,
      internalFormat: probe.internalFormat,
      format: gl.RGBA,
      type: probe.type,
      bytesPerPixel: probe.bytesPerPixel,
      filter: gl.LINEAR,
      depth: true,
      label: 'scene',
    });
    const trail = createRenderTarget(gl, {
      width: scene,
      height: sceneH,
      internalFormat: probe.internalFormat,
      format: gl.RGBA,
      type: probe.type,
      bytesPerPixel: probe.bytesPerPixel,
      filter: gl.LINEAR,
      depth: false,
      label: 'trail',
    });
    const ldr = createRenderTarget(gl, {
      width: this.drawW,
      height: this.drawH,
      internalFormat: gl.RGBA8,
      format: gl.RGBA,
      type: gl.UNSIGNED_BYTE,
      bytesPerPixel: 4,
      filter: gl.LINEAR,
      depth: false,
      label: 'ldr',
    });

    const bloom: RenderTarget[] = [];
    let bw = scene;
    let bh = sceneH;
    for (let i = 0; i < levels; i++) {
      bw = Math.max(1, bw >> 1);
      bh = Math.max(1, bh >> 1);
      bloom.push(createRenderTarget(gl, {
        width: bw,
        height: bh,
        internalFormat: probe.internalFormat,
        format: gl.RGBA,
        type: probe.type,
        bytesPerPixel: probe.bytesPerPixel,
        filter: gl.LINEAR,
        depth: false,
        label: `bloom${i}`,
      }));
    }
    const first = bloom[0];
    const vol = createRenderTarget(gl, {
      width: first.width,
      height: first.height,
      internalFormat: probe.internalFormat,
      format: gl.RGBA,
      type: probe.type,
      bytesPerPixel: probe.bytesPerPixel,
      filter: gl.LINEAR,
      depth: false,
      label: 'volumetric',
    });
    const dofHalf = createRenderTarget(gl, {
      width: Math.max(1, scene >> 1),
      height: Math.max(1, sceneH >> 1),
      internalFormat: probe.internalFormat,
      format: gl.RGBA,
      type: probe.type,
      bytesPerPixel: probe.bytesPerPixel,
      filter: gl.LINEAR,
      depth: false,
      label: 'dofHalf',
    });
    const dofFull = createRenderTarget(gl, {
      width: scene,
      height: sceneH,
      internalFormat: probe.internalFormat,
      format: gl.RGBA,
      type: probe.type,
      bytesPerPixel: probe.bytesPerPixel,
      filter: gl.LINEAR,
      depth: false,
      label: 'dofFull',
    });

    this.targets = { scene: sceneTarget, trail, ldr, vol, dofHalf, dofFull, bloom };

    let bytes = 0;
    const all = [sceneTarget, trail, ldr, vol, dofHalf, dofFull, ...bloom];
    for (const t of all) bytes += t.bytes;
    this.fboBytes = bytes;

    gl.disable(gl.SCISSOR_TEST);
    gl.colorMask(true, true, true, true);
    gl.depthMask(true);
    gl.clearColor(0, 0, 0, 0);
    gl.clearDepth(1);
    for (const t of all) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, t.fbo);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private disposeTargets(): void {
    const gl = this.gl;
    const targets = this.targets;
    if (!gl || !targets) {
      this.targets = null;
      return;
    }
    deleteRenderTarget(gl, targets.scene);
    deleteRenderTarget(gl, targets.trail);
    deleteRenderTarget(gl, targets.ldr);
    deleteRenderTarget(gl, targets.vol);
    deleteRenderTarget(gl, targets.dofHalf);
    deleteRenderTarget(gl, targets.dofFull);
    for (const b of targets.bloom) deleteRenderTarget(gl, b);
    this.targets = null;
    this.fboBytes = 0;
  }

  // -------------------------------------------------------------------------
  // Renderer contract
  // -------------------------------------------------------------------------

  resize(cssWidth: number, cssHeight: number, dpr: number, scale: number): void {
    const gl = this.gl;
    if (!gl || this.disposed) return;

    this.cssW = Math.max(1, Math.floor(num(cssWidth, 1)));
    this.cssH = Math.max(1, Math.floor(num(cssHeight, 1)));
    this.dpr = clampNum(num(dpr, 1), 0.5, MAX_DPR);
    this.renderScale = clampNum(num(scale, 1), 0.3, 2);

    let drawW = Math.max(1, Math.round(this.cssW * this.dpr * this.renderScale));
    let drawH = Math.max(1, Math.round(this.cssH * this.dpr * this.renderScale));
    const ss = clampNum(num(this.profile.supersample, 1), 1, 1.5);
    let sceneW = Math.max(1, Math.round(drawW * ss));
    let sceneH = Math.max(1, Math.round(drawH * ss));

    // Stay inside MAX_TEXTURE_SIZE on extreme dpr x supersample combinations
    // rather than failing framebuffer creation.
    const cap = Math.max(512, this.maxTextureSize);
    const fit = Math.min(1, cap / drawW, cap / drawH, cap / sceneW, cap / sceneH);
    if (fit < 1) {
      drawW = Math.max(1, Math.floor(drawW * fit));
      drawH = Math.max(1, Math.floor(drawH * fit));
      sceneW = Math.max(1, Math.floor(sceneW * fit));
      sceneH = Math.max(1, Math.floor(sceneH * fit));
    }

    const changed = drawW !== this.drawW || drawH !== this.drawH
      || sceneW !== this.sceneW || sceneH !== this.sceneH;

    this.drawW = drawW;
    this.drawH = drawH;
    this.sceneW = sceneW;
    this.sceneH = sceneH;
    this.sceneAspect = sceneW / sceneH;
    this.canvas.width = drawW;
    this.canvas.height = drawH;
    this.camDirty = true;

    if (changed || !this.targets) this.createTargets();
    gl.viewport(0, 0, drawW, drawH);
  }

  frame(nowMs: number, deltaMs: number): void {
    const start = performance.now();
    const gl = this.gl;
    const progs = this.progs;
    const gpu = this.gpu;
    const targets = this.targets;
    if (!gl || !progs || !gpu || !targets || this.disposed || this.contextLost) return;

    this.nowSeconds = num(nowMs, 0) * 0.001;
    const params = this.params;
    const dtRaw = clampNum(num(deltaMs, 1000 / 60) * 0.001, 0, 0.1);
    const speed = clampNum(num(params.speed, 1), 0, 3);

    this.syncCounts();
    const stepping = !params.paused && speed > 0.0001 && this.activeCount > 0;
    const dt = clampNum(dtRaw * Math.max(speed, 1e-3), 1 / 240, 1 / 30);

    if (stepping) {
      this.sim.time += dt;
      this.sim.frame += 1;
    }
    this.audioPulse *= Math.exp(-6 * dtRaw);
    this.advanceShocks(stepping ? dt : 0);
    this.updateMatrices();
    this.fillSimState(dt);

    // One upload feeds both the sim and the draw programs (shared SimBlock).
    gl.bindBuffer(gl.UNIFORM_BUFFER, gpu.simUbo);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, this.simWriter.write(this.sim));

    if (stepping) this.stepSimulation();
    this.renderScene();
    this.copySceneToTrail();
    this.runBloom();
    this.runVolumetric();
    this.runDof();
    this.runComposite();
    this.runFxaa();

    this.updateStats(dtRaw, start);
  }

  shock(ndcX: number, ndcY: number, strength = 1): void {
    if (this.disposed || this.contextLost) return;
    if (!Number.isFinite(ndcX) || !Number.isFinite(ndcY)) return;
    if (clampNum(num(this.params.shock, 1), 0, 3) <= 0.0001) return;
    const power = clampNum(num(strength, 1), 0, 4);
    if (power <= 0) return;

    this.updateMatrices();
    this.unproject(ndcX, ndcY, 0, this.rayA);
    this.unproject(ndcX, ndcY, 1, this.rayB);

    let dx = this.rayB[0] - this.rayA[0];
    let dy = this.rayB[1] - this.rayA[1];
    let dz = this.rayB[2] - this.rayA[2];
    const len = Math.hypot(dx, dy, dz) || 1;
    dx /= len;
    dy /= len;
    dz /= len;

    // Camera forward axis, from eye toward the origin.
    let fx = CAMERA_TARGET[0] - this.eye[0];
    let fy = CAMERA_TARGET[1] - this.eye[1];
    let fz = CAMERA_TARGET[2] - this.eye[2];
    const fl = Math.hypot(fx, fy, fz) || 1;
    fx /= fl;
    fy /= fl;
    fz /= fl;

    // Intersect with the plane through the origin whose normal is the forward axis.
    const denom = dx * fx + dy * fy + dz * fz;
    let t = Number.NaN;
    if (Math.abs(denom) > 1e-4) {
      t = -(this.rayA[0] * fx + this.rayA[1] * fy + this.rayA[2] * fz) / denom;
    }
    // Fallback: the point at focus distance along the ray.
    if (!Number.isFinite(t) || t < 0 || t > 200) t = this.camDist;

    const idx = this.shockCursor;
    this.shockCursor = (idx + 1) % MAX_SHOCKS;
    const o = idx * 4;
    this.shockOrigin[o] = this.rayA[0] + dx * t;
    this.shockOrigin[o + 1] = this.rayA[1] + dy * t;
    this.shockOrigin[o + 2] = this.rayA[2] + dz * t;
    this.shockOrigin[o + 3] = power; // params.shock is applied by the shader
    this.shockAges[idx] = 0;
    this.shockWritten = Math.min(MAX_SHOCKS, this.shockWritten + 1);
  }

  setAudio(low: number, mid: number, high: number, beat: number): void {
    const sens = clampNum(num(this.params.audioSensitivity, 1), 0, 3);
    this.audioLow = clampNum(num(low, 0) * sens, 0, 4);
    this.audioMid = clampNum(num(mid, 0) * sens, 0, 4);
    this.audioHigh = clampNum(num(high, 0) * sens, 0, 4);
    this.beat = clampNum(num(beat, 0), 0, 4);

    if (this.beat > 0.6) {
      this.audioPulse = Math.max(this.audioPulse, Math.min(this.beat * 0.5, 0.6));
      if (this.nowSeconds - this.lastBeatShock > 0.28) {
        this.lastBeatShock = this.nowSeconds;
        this.autoShockIndex += 1;
        const h = hash1u((Math.imul(this.autoShockIndex, 2654435761) ^ 0x9e3779b9) >>> 0);
        const nx = (rnd1u(h) * 2 - 1) * 0.55;
        const ny = (rnd1u(hash1u(h)) * 2 - 1) * 0.5;
        this.shock(nx, ny, 0.35 + this.audioLow * 0.5);
      }
    }
  }

  orbit(dx: number, dy: number, plane?: 'xw' | 'yw' | 'zw'): void {
    const step = 0.006;
    const ax = num(dx, 0) * step;
    const ay = num(dy, 0) * step;
    const a = this.angles;
    if (plane === 'xw') {
      a[2] += ax;
      a[4] += ay;
    } else if (plane === 'yw') {
      a[4] += ax;
      a[5] += ay;
    } else if (plane === 'zw') {
      a[5] += ax;
      a[3] += ay;
    } else {
      // Default drag maps to the (x,w) and (y,z) planes, which is what the app sends.
      a[2] += ax;
      a[3] += ay;
    }
    for (let i = 0; i < 6; i++) a[i] = wrapAngle(a[i]);
    this.rotDirty = true;
  }

  zoom(delta: number): void {
    this.camDist = clampNum(this.camDist * Math.exp(wheelNotches(delta) * 0.12), MIN_ZOOM, MAX_ZOOM);
    this.camDirty = true;
  }

  slice(delta: number): void {
    this.wSlice = clampNum(this.wSlice + wheelNotches(delta) * 0.35, -MAX_SLICE, MAX_SLICE);
  }

  reset(): void {
    if (this.disposed) return;
    this.camDist = DEFAULT_ZOOM;
    this.wSlice = 0;
    for (let i = 0; i < 6; i++) this.angles[i] = DEFAULT_ANGLES[i];
    this.rotDirty = true;
    this.camDirty = true;
    this.ping = 0;
    this.shockCursor = 0;
    this.shockWritten = 0;
    this.shockOrigin.fill(0);
    this.shockAges.fill(0);
    this.audioPulse = 0;
    this.beat = 0;
    this.audioLow = 0;
    this.audioMid = 0;
    this.audioHigh = 0;
    this.sim.time = 0;
    this.sim.frame = 0;
    const gpu = this.gpu;
    if (this.gl && gpu) this.seedParticles(gpu.capacity);
  }

  capture(): Promise<Blob | null> {
    return new Promise<Blob | null>((resolve) => {
      if (this.disposed) {
        resolve(null);
        return;
      }
      try {
        // preserveDrawingBuffer: true keeps the last frame readable here.
        this.canvas.toBlob((blob) => resolve(blob), 'image/png');
      } catch (err) {
        console.error('[webgl2] capture() failed', err);
        resolve(null);
      }
    });
  }

  setQualityScale(scale: number): void {
    this.qualityScale = clampNum(num(scale, 1), 0.3, 2);
  }

  /** True once the context, programs and particle buffers all exist. */
  get ready(): boolean {
    return this.gl !== null && this.progs !== null && this.activeCount > 0;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.canvas.removeEventListener('webglcontextlost', this.onContextLost);

    const gl = this.gl;
    if (gl) {
      this.disposeTargets();
      const gpu = this.gpu;
      if (gpu) {
        this.deleteParticleData(gpu.data);
        gl.deleteBuffer(gpu.simUbo);
        gl.deleteBuffer(gpu.postUbo);
        gl.deleteVertexArray(gpu.quadVao);
      }
      const progs = this.progs;
      if (progs) {
        for (const program of Object.values(progs)) program.dispose();
      }
      const lose = gl.getExtension('WEBGL_lose_context');
      if (lose) lose.loseContext();
    }
    this.gpu = null;
    this.progs = null;
    this.gl = null;
  }

  // -------------------------------------------------------------------------
  // Per-frame state
  // -------------------------------------------------------------------------

  private syncCounts(): void {
    const gpu = this.gpu;
    if (!gpu) return;
    const wantSim = finiteCount(this.params.simCount);
    if (wantSim > gpu.capacity) this.allocateParticles(wantSim);
    this.activeCount = Math.min(wantSim, gpu.capacity);

    const wantDraw = finiteCount(this.params.showCount);
    this.drawCount = Math.min(wantDraw, this.activeCount * MAX_MULTIPLICITY_PASSES);
    this.passes = this.activeCount > 0 && this.drawCount > 0
      ? Math.max(1, Math.min(MAX_MULTIPLICITY_PASSES, Math.ceil(this.drawCount / this.activeCount)))
      : 0;
    this.sim.multiplicity = this.passes;
  }

  private updateMatrices(): void {
    const s = this.sim;
    if (this.rotDirty) {
      if (this.rotFastPath) {
        this.composeRotation(this.rotOut);
        s.rot4.set(this.rotOut);
      } else {
        s.rot4.set(rotationMatrix4(this.angles));
      }
      const a = this.angles;
      s.rot0[0] = a[0];
      s.rot0[1] = a[1];
      s.rot0[2] = a[2];
      s.rot0[3] = a[3];
      s.rot1[0] = a[4];
      s.rot1[1] = a[5];
      this.rotDirty = false;
      this.camDirty = true;
    }
    if (this.camDirty) {
      this.eye[0] = 0;
      this.eye[1] = 0;
      this.eye[2] = this.camDist;
      lookAt(this.eye, CAMERA_TARGET, CAMERA_UP, this.view);
      perspective(FOV_Y, this.sceneAspect, NEAR_PLANE, FAR_PLANE, this.proj);
      mulMat4(this.proj, this.view, this.mvp);
      invert4(this.mvp, this.invMvp);
      this.camDirty = false;
    }
    s.view.set(this.view);
    s.proj.set(this.proj);
    s.zoom = this.camDist;
  }

  private unproject(ndcX: number, ndcY: number, ndcZ: number, out: Float64Array): void {
    if (this.camDirty) this.updateMatrices();
    const m = this.invMvp;
    const x = m[0] * ndcX + m[4] * ndcY + m[8] * ndcZ + m[12];
    const y = m[1] * ndcX + m[5] * ndcY + m[9] * ndcZ + m[13];
    const z = m[2] * ndcX + m[6] * ndcY + m[10] * ndcZ + m[14];
    const w = m[3] * ndcX + m[7] * ndcY + m[11] * ndcZ + m[15];
    const iw = 1 / (Math.abs(w) > 1e-9 ? w : 1);
    out[0] = x * iw;
    out[1] = y * iw;
    out[2] = z * iw;
  }

  private advanceShocks(dt: number): void {
    if (dt <= 0) return;
    for (let i = 0; i < this.shockWritten; i++) {
      const age = this.shockAges[i] + dt;
      this.shockAges[i] = age;
      // A dead shell reports zero strength so the shader skips its slot.
      if (age >= SHOCK_LIFE) this.shockOrigin[i * 4 + 3] = 0;
    }
  }

  private fillSimState(dt: number): void {
    const s = this.sim;
    const p = this.params;
    const fractal = clampNum(num(p.fractalDim, 0.6), 0, 1);

    s.dt = dt;
    s.simCount = this.activeCount;
    s.width = this.sceneW;
    s.height = this.sceneH;
    s.aspect = this.sceneAspect;
    s.wSlice = this.wSlice;
    s.fov4 = 0.45;
    s.nearBlend = 0.35;

    s.julia = p.julia;
    s.curl = clampNum(num(p.curl, 0.55), 0, 2);
    s.damping = clampNum(num(p.damping, 0.34), 0.02, 1.5);
    s.gravity = clampNum(num(p.gravity, 0.16), 0, 0.6);
    s.confinement = clampNum(num(p.confinement, 0.34), 0, 1.5);

    s.attractDepth = fractal;
    s.escape = 2.5 + fractal * 3.5;
    // fractalDim 0..1 -> power 2.4..6.5 and warp 0.25..0.9 (shared with WebGPU).
    s.power = 2.4 + fractal * 4.1;
    s.warp = 0.25 + fractal * 0.65;

    s.audioLow = this.audioLow;
    s.audioMid = this.audioMid;
    s.audioHigh = this.audioHigh;
    s.beat = this.beat;

    s.hue = wrapAngle(num(p.hue, 0) * TAU) / TAU;
    s.saturation = 1;
    s.paletteMix = 0.55;
    s.exposure = clampNum(num(p.exposure, 1.05), 0.05, 8);

    // Matching the WebGPU path's sprite scale keeps the two backends looking like
    // the same universe: the particle radius is the main control over whether the
    // cloud reads as continuous gas or as sparse coloured dust.
    s.sizeScale = clampNum(num(p.particleSize, 1.5) * 0.34, 0.1, 5);
    s.energyScale = 1;
    s.multiplicity = this.passes;
    s.trailDecay = clampNum(num(p.trails, 0.6), 0, 0.97);

    s.rot1[2] = Math.tan(FOV_Y * 0.5);
    s.rot1[3] = s.time * 0.25; // slow 4D-only drift

    s.shockGain = clampNum(num(p.shock, 1), 0, 3);
    s.bloom = clampNum(num(p.bloom, 0.62), 0, 4);
    s.dpr = this.dpr;

    s.focusDist = this.camDist;
    s.aperture = clampNum(num(p.dof, 0.45), 0, 1.5) * 0.6;
    s.grain = clampNum(num(p.grain, 0.35), 0, 1);
    s.vignette = clampNum(num(p.vignette, 0.45), 0, 1);

    s.activeShocks = this.shockWritten;
    s.shockRadius = 0.6;
    s.shockThickness = 0.32;
    s.shockLife = SHOCK_LIFE;
  }

  private uploadPost(dstW: number, dstH: number, src: RenderTarget, dirX = 0, dirY = 0): void {
    const gl = this.gl as WebGL2RenderingContext;
    const gpu = this.gpu as ParticleBuffers;
    const p = this.post;
    const params = this.params;
    const floatOk = this.probe ? this.probe.floatRenderable : false;

    p.width = dstW;
    p.height = dstH;
    p.texelX = dirX;
    p.texelY = dirY;
    p.srcTexelX = 1 / Math.max(src.width, 1);
    p.srcTexelY = 1 / Math.max(src.height, 1);
    p.exposure = clampNum(num(params.exposure, 1.05), 0.05, 8);
    p.bloom = clampNum(num(params.bloom, 0.62), 0, 4);
    p.bloomRadius = clampNum(num(params.bloomRadius, 0.55), 0, 1);
    // Without float targets the scene is already compressed, so bright-pass lower.
    p.threshold = floatOk ? 1.0 : 0.45;
    p.chroma = clampNum(num(params.chroma, 0.34), 0, 2);
    p.grain = clampNum(num(params.grain, 0.35), 0, 1);
    p.vignette = clampNum(num(params.vignette, 0.45), 0, 1);
    p.dof = clampNum(num(params.dof, 0.45), 0, 1.5);
    p.volumetric = clampNum(num(params.volumetric, 0.4), 0, 1.5);
    p.time = this.sim.time;
    p.aspect = dstW / Math.max(dstH, 1);
    p.frame = this.sim.frame;
    p.focusDepth = this.camDist;
    p.focusRange = Math.max(0.2, this.camDist * 0.35);
    p.maxCoc = 10 * (this.sceneH / 1080);
    p.nearPlane = NEAR_PLANE;
    p.farPlane = FAR_PLANE;
    p.sourceLod = 0;
    p.upsampleRadius = 1 + p.bloomRadius * 1.5;
    p.bloomLevels = this.targets ? this.targets.bloom.length : 1;
    p.hue = num(params.hue, 0);
    p.saturation = 1;
    p.quality = this.qualityScale;
    // pColor.w carries the rendered particle count into the composite, which uses
    // it for the 1/sqrt(N) pre-exposure that keeps additive emission in range.
    p.sceneLuma = this.activeCount * MAX_MULTIPLICITY_PASSES;
    // Diagnostic tap, reachable only under ?selftest=1: pColor.y = 1..4 makes the
    // composite output a single stage of the HDR chain so a saturation problem can
    // be attributed to a pass instead of guessed at. The tap is driven through the
    // hue slider, which the composite does not otherwise read.
    if (this.selfTestEnabled) {
      const tap = clampNum(num(params.hue, 0), 0, 1);
      p.saturation = tap > 0.9 ? 4 : tap > 0.65 ? 3 : tap > 0.4 ? 2 : tap > 0.15 ? 1 : 0;
      p.hue = 0;
    } else {
      p.saturation = 1;
    }

    gl.bindBuffer(gl.UNIFORM_BUFFER, gpu.postUbo);
    gl.bufferSubData(gl.UNIFORM_BUFFER, 0, this.postWriter.write(p));
  }

  // -------------------------------------------------------------------------
  // Passes
  // -------------------------------------------------------------------------

  private stepSimulation(): void {
    const gl = this.gl as WebGL2RenderingContext;
    const gpu = this.gpu as ParticleBuffers;
    const progs = this.progs as Programs;
    const src = this.ping;
    const dst = 1 - src;
    const data = gpu.data;

    gl.useProgram(progs.sim.handle);
    gl.uniform4fv(progs.sim.loc('uShocks[0]'), this.shockOrigin);
    gl.uniform1fv(progs.sim.loc('uShockAges[0]'), this.shockAges);

    gl.bindVertexArray(data.vaoSim[src]);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, data.bufPos[dst]);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 1, data.bufVel[dst]);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 2, data.bufMeta[dst]);

    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, this.activeCount);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);

    // Release the capture points so no later draw can alias them.
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 1, null);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 2, null);
    gl.bindVertexArray(null);

    this.ping = dst;
  }

  private renderScene(): void {
    const gl = this.gl as WebGL2RenderingContext;
    const gpu = this.gpu as ParticleBuffers;
    const progs = this.progs as Programs;
    const targets = this.targets as Framebuffers;
    const params = this.params;

    gl.bindFramebuffer(gl.FRAMEBUFFER, targets.scene.fbo);
    gl.viewport(0, 0, targets.scene.width, targets.scene.height);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(true);
    gl.colorMask(true, true, true, true);
    gl.clearColor(0, 0, 0, 0);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    // 1. Trails: previous frame's HDR scene, scaled by params.trails.
    const trails = clampNum(num(params.trails, 0.6), 0, 0.97);
    if (trails > 0.001) {
      gl.useProgram(progs.trail.handle);
      this.bindTexture(0, targets.trail.color);
      gl.bindVertexArray(gpu.quadVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    const dofAmount = clampNum(num(params.dof, 0.45), 0, 1.5);
    const count = this.activeCount;
    const drawVao = gpu.data.vaoDraw[this.ping];

    // 2. Depth prepass: real depth for the DOF chain, colour writes masked off.
    if (count > 0 && dofAmount > 0.001) {
      gl.useProgram(progs.depth.handle);
      gl.uniform1f(progs.depth.loc('uMultiplicityPass'), 0);
      gl.uniform1f(progs.depth.loc('uAudioPulse'), this.audioPulse);
      gl.bindVertexArray(drawVao);
      gl.colorMask(false, false, false, false);
      gl.depthMask(true);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LESS);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, count);
    }

    // 3. Additive billboards. Depth test off: additive light is order-independent,
    //    and the prepass depth is what the post chain blurs against.
    if (count > 0 && this.drawCount > 0 && this.passes > 0) {
      gl.useProgram(progs.particle.handle);
      gl.uniform1f(progs.particle.loc('uAudioPulse'), this.audioPulse);
      gl.bindVertexArray(drawVao);
      gl.colorMask(true, true, true, true);
      gl.depthMask(false);
      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      if (this.probe && this.probe.floatRenderable) {
        gl.blendFunc(gl.ONE, gl.ONE);
      } else {
        // RGBA8 accumulation: screen-style blending keeps the core from clipping.
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_COLOR);
      }
      for (let k = 0; k < this.passes; k++) {
        const remaining = this.drawCount - k * count;
        const n = Math.min(count, remaining);
        if (n <= 0) break;
        // Instanced attributes cannot be re-indexed in WebGL2, so multiplicity is
        // one draw per scatter copy with the copy index passed as a uniform.
        gl.uniform1f(progs.particle.loc('uMultiplicityPass'), k);
        gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, n);
      }
      gl.disable(gl.BLEND);
    }

    gl.bindVertexArray(null);
    gl.depthMask(true);
    gl.colorMask(true, true, true, true);
  }

  private copySceneToTrail(): void {
    const gl = this.gl as WebGL2RenderingContext;
    const targets = this.targets as Framebuffers;
    gl.colorMask(true, true, true, true);
    gl.depthMask(true);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, targets.scene.fbo);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, targets.trail.fbo);
    gl.blitFramebuffer(
      0, 0, targets.scene.width, targets.scene.height,
      0, 0, targets.trail.width, targets.trail.height,
      gl.COLOR_BUFFER_BIT, gl.NEAREST,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private beginFullscreen(): void {
    const gl = this.gl as WebGL2RenderingContext;
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.colorMask(true, true, true, true);
  }

  private drawQuad(): void {
    const gl = this.gl as WebGL2RenderingContext;
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private runBloom(): void {
    const gl = this.gl as WebGL2RenderingContext;
    const gpu = this.gpu as ParticleBuffers;
    const progs = this.progs as Programs;
    const targets = this.targets as Framebuffers;
    const levels = targets.bloom.length;

    this.beginFullscreen();
    gl.bindVertexArray(gpu.quadVao);

    // Bright pass -> level 0 (half resolution).
    const base = targets.bloom[0];
    gl.bindFramebuffer(gl.FRAMEBUFFER, base.fbo);
    gl.viewport(0, 0, base.width, base.height);
    gl.useProgram(progs.bright.handle);
    this.bindTexture(0, targets.scene.color);
    this.uploadPost(base.width, base.height, targets.scene);
    this.drawQuad();

    // Downsample chain (13-tap filter).
    gl.useProgram(progs.down.handle);
    for (let i = 1; i < levels; i++) {
      const src = targets.bloom[i - 1];
      const dst = targets.bloom[i];
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
      gl.viewport(0, 0, dst.width, dst.height);
      this.bindTexture(0, src.color);
      this.uploadPost(dst.width, dst.height, src);
      this.drawQuad();
    }

    // Upsample chain (9-tap tent), additively accumulating into finer levels.
    gl.useProgram(progs.up.handle);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let i = levels - 2; i >= 0; i--) {
      const src = targets.bloom[i + 1];
      const dst = targets.bloom[i];
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fbo);
      gl.viewport(0, 0, dst.width, dst.height);
      this.bindTexture(0, src.color);
      this.uploadPost(dst.width, dst.height, src);
      this.drawQuad();
    }
    gl.disable(gl.BLEND);
  }

  private runVolumetric(): void {
    const gl = this.gl as WebGL2RenderingContext;
    const gpu = this.gpu as ParticleBuffers;
    const progs = this.progs as Programs;
    const targets = this.targets as Framebuffers;
    const amount = clampNum(num(this.params.volumetric, 0.4), 0, 1.5);
    if (amount <= 0.001) return;

    this.beginFullscreen();
    gl.bindVertexArray(gpu.quadVao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, targets.vol.fbo);
    gl.viewport(0, 0, targets.vol.width, targets.vol.height);
    gl.useProgram(progs.volumetric.handle);
    this.bindTexture(0, targets.bloom[0].color);
    this.uploadPost(targets.vol.width, targets.vol.height, targets.bloom[0]);
    this.drawQuad();
  }

  private runDof(): void {
    const gl = this.gl as WebGL2RenderingContext;
    const gpu = this.gpu as ParticleBuffers;
    const progs = this.progs as Programs;
    const targets = this.targets as Framebuffers;
    const amount = clampNum(num(this.params.dof, 0.45), 0, 1.5);
    const depthTex = targets.scene.depth;
    if (amount <= 0.001 || !depthTex) return;

    this.beginFullscreen();
    gl.bindVertexArray(gpu.quadVao);

    // CoC prefilter: full-res colour + depth -> half-res colour with CoC in alpha.
    gl.bindFramebuffer(gl.FRAMEBUFFER, targets.dofHalf.fbo);
    gl.viewport(0, 0, targets.dofHalf.width, targets.dofHalf.height);
    gl.useProgram(progs.dofPrepare.handle);
    this.bindTexture(0, targets.scene.color);
    this.bindTexture(1, depthTex);
    this.uploadPost(targets.dofHalf.width, targets.dofHalf.height, targets.scene);
    this.drawQuad();

    // Golden-angle bokeh gather, upsampled back to full resolution.
    gl.bindFramebuffer(gl.FRAMEBUFFER, targets.dofFull.fbo);
    gl.viewport(0, 0, targets.dofFull.width, targets.dofFull.height);
    gl.useProgram(progs.dofGather.handle);
    this.bindTexture(0, targets.dofHalf.color);
    this.uploadPost(targets.dofFull.width, targets.dofFull.height, targets.dofHalf);
    this.drawQuad();
  }

  private runComposite(): void {
    const gl = this.gl as WebGL2RenderingContext;
    const gpu = this.gpu as ParticleBuffers;
    const progs = this.progs as Programs;
    const targets = this.targets as Framebuffers;
    const params = this.params;

    const volTex = clampNum(num(params.volumetric, 0.4), 0, 1.5) > 0.001
      ? targets.vol.color
      : targets.bloom[0].color;
    const dofTex = clampNum(num(params.dof, 0.45), 0, 1.5) > 0.001
      ? targets.dofFull.color
      : targets.scene.color;

    this.beginFullscreen();
    gl.bindVertexArray(gpu.quadVao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, targets.ldr.fbo);
    gl.viewport(0, 0, targets.ldr.width, targets.ldr.height);
    gl.useProgram(progs.composite.handle);
    this.bindTexture(0, targets.scene.color);
    this.bindTexture(1, targets.bloom[0].color);
    this.bindTexture(2, volTex);
    this.bindTexture(3, dofTex);
    this.uploadPost(targets.ldr.width, targets.ldr.height, targets.scene);
    this.drawQuad();
  }

  private runFxaa(): void {
    const gl = this.gl as WebGL2RenderingContext;
    const gpu = this.gpu as ParticleBuffers;
    const progs = this.progs as Programs;
    const targets = this.targets as Framebuffers;

    this.beginFullscreen();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.drawW, this.drawH);
    gl.useProgram(progs.fxaa.handle);
    gl.bindVertexArray(gpu.quadVao);
    this.bindTexture(0, targets.ldr.color);
    this.uploadPost(this.drawW, this.drawH, targets.ldr);
    this.drawQuad();
    gl.bindVertexArray(null);
  }

  private bindTexture(unit: number, texture: WebGLTexture): void {
    const gl = this.gl as WebGL2RenderingContext;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
  }

  private updateStats(dtRaw: number, startMs: number): void {
    const stats = this.stats;
    const dt = Math.max(dtRaw, 0.0001);
    const instantFps = 1 / Math.max(dt, 0.0002);
    stats.fps += (instantFps - stats.fps) * 0.1;
    stats.frameMs += (dt * 1000 - stats.frameMs) * 0.1;
    const cpu = performance.now() - startMs;
    stats.cpuMs += (cpu - stats.cpuMs) * 0.1;
    stats.gpuMs = 0;
    stats.gpuTimingSupported = false;
    stats.simCount = this.activeCount;
    // Multiplicity copies are drawn as extra passes; the reported counts follow
    // the simulated particle budget (matching the WebGPU backend's reporting).
    stats.renderCount = this.activeCount;
    stats.drawVertices = this.activeCount * 6;
    stats.renderScale = this.renderScale;
    stats.drawWidth = this.drawW;
    stats.drawHeight = this.drawH;
    stats.memoryEstimateMB = (estimateParticleMemory(this.activeCount) + this.fboBytes) / 1048576;
    this.telemetry.onTelemetry?.(stats);
  }
}
