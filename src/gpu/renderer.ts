/**
 * Aetheria — the WebGPU renderer.
 *
 * Frame structure (one command encoder, one submit):
 *
 *   1. CPU: integrate audio, advance the 4D rotation, build MVP, write uniforms
 *   2. compute: simulate (ping -> pong), then parity blit (pong -> ping)
 *   3. render:  fade history -> particle billboards (additive, HDR)
 *   4. post:    bright -> bloom pyramid -> god rays -> DOF -> composite -> FXAA
 *
 * Adaptation philosophy: quality is a control loop, not a constant. Frametime is
 * tracked with an EMA; when it exceeds the 16.6 ms budget the renderer first
 * sheds resolution (cheapest, least visible), then post effects, then simulated
 * particles — the last thing a viewer notices is fewer particles inside a
 * four-million-particle cloud.
 */

import type { GpuContext } from '../gpu/device';
import { createGpuContext, compileChecked, maxParticlesForDevice } from '../gpu/device';
import { ParticleSystem, type ShockEvent } from '../gpu/particles';
import { MAX_SHOCKS } from '../gpu/wgsl/simulate.wgsl';
import { PostChain } from '../gpu/post';
import { PassTimer, type PassTiming } from '../gpu/timing';
import { PARTICLE_VERTEX_WGSL, PARTICLE_FRAGMENT_WGSL } from '../gpu/wgsl/particle.wgsl';
import {
  BackendUnavailable,
  estimateParticleMemory,
  type BackendKind,
  type HudStats,
  type Renderer,
  type RendererTelemetry,
} from '../core/types';
import { HARD_MAX_SIM_PARTICLES, MAX_RENDER_PARTICLES, type DeviceProfile, type Params } from '../core/config';
import {
  SimWriter,
  makeSimState,
  makePostState,
  SIM_UNIFORM_BYTES,
} from '../core/uniforms';
import { clamp, lookAt, mulMat4, perspective, invert4, xform4, rotationMatrix4, TAU } from '../core/math4d';

const VERTICES_PER_QUAD = 6;
/** Storage-buffer reads cost 16 B per vertex, so instances per call are capped. */
const DEFAULT_INSTANCES_PER_CALL = 2_000_000;

const NEAR = 0.08;
const FAR = 90;

/**
 * Camera defaults, in world units.
 *
 * The cloud's own radius is roughly 2.5-3 (see the birth shell and the
 * confinement radius in the simulation), so the distance and field of view
 * together decide how much of the frame it fills. These are tuned so the nebula
 * covers most of the viewport with soft structure reaching the edges, rather than
 * sitting as a small knot at the centre.
 */
const DEFAULT_CAMERA_DISTANCE = 2.3;
const DEFAULT_FOV_Y = 1.15;

/** Six rotation-plane angles: xy, xz, xw, yz, yw, zw. */
type Rot6 = [number, number, number, number, number, number];

export class GpuRenderer implements Renderer {
  readonly kind: BackendKind = 'webgpu';
  readonly params: Params;
  readonly profile: DeviceProfile;
  readonly stats: HudStats;

  private readonly ctx: GpuContext;
  private readonly telemetry: RendererTelemetry;
  private readonly particles: ParticleSystem;
  private readonly post: PostChain;

  private readonly simUniform: GPUBuffer;
  private readonly simWriter = new SimWriter();
  private readonly state = makeSimState();
  private readonly postState = makePostState();

  private readonly particlePipeline: GPURenderPipeline;
  private readonly particlePipelineNoDepth: GPURenderPipeline;
  private readonly particlesPerCall: number;
  /** Brackets every pass in the frame; the adaptive loop reads its totals. */
  private timer: PassTimer | null = null;

  private lastTimings: PassTiming[] = [];

  private rot: Rot6 = [0.35, 0.12, 0.85, -0.22, 0.55, 0.1];
  private fovY = DEFAULT_FOV_Y;
  private readonly idleAutoRotate: boolean;
  private idleTimer = 0;

  private readonly proj = new Float32Array(16);
  private readonly view = new Float32Array(16);
  private readonly mvp = new Float32Array(16);
  private readonly invMvp = new Float32Array(16);

  private shocks: ShockEvent[] = [];
  private shockCursor = 0;

  private elapsed = 0;
  private frameIndex = 0;
  private started = false;

  private cssWidth = 1;
  private cssHeight = 1;
  private cssDpr = 1;
  private renderScale = 1;
  private targetScale = 1;

  private rttEma = 16.6;
  private cpuEma = 1;
  private fpsEma = 60;
  /** Total GPU cost of the last profiled frame, and its per-pass breakdown. */
  private gpuMs = 0;
  private computeMs = 0;

  private readonly downgrades: string[] = [];
  private postLevel = 2; // 2 = full, 1 = no DOF/god rays, 0 = bloom only
  /** Ablation for the particle pass, set by the profiling tools. See setAblation(). */
  private ablation = '';

  /** Optional GPU-side frame capture, attached by the verification runner. */
  selfTest: { consume(device: GPUDevice, canvas: GPUTexture): boolean } | null = null;

  private constructor(
    ctx: GpuContext,
    profile: DeviceProfile,
    params: Params,
    telemetry: RendererTelemetry,
    particles: ParticleSystem,
    post: PostChain,
    simUniform: GPUBuffer,
    particlePipeline: GPURenderPipeline,
    particlePipelineNoDepth: GPURenderPipeline,
    particlesPerCall: number,
    reducedMotion: boolean
  ) {
    this.ctx = ctx;
    this.profile = profile;
    this.params = params;
    this.telemetry = telemetry;
    this.particles = particles;
    this.post = post;
    this.simUniform = simUniform;
    this.particlePipeline = particlePipeline;
    this.particlePipelineNoDepth = particlePipelineNoDepth;
    this.particlesPerCall = particlesPerCall;
    this.idleAutoRotate = !reducedMotion;
    this.stats = {
      fps: 60,
      frameMs: 16.6,
      cpuMs: 1,
      gpuMs: 0,
      computeMs: 0,
      simCount: params.simCount,
      renderCount: params.showCount,
      culledCount: 0,
      drawVertices: 0,
      renderScale: profile.renderScale,
      drawWidth: 1,
      drawHeight: 1,
      backend: 'webgpu',
      deviceLabel: describeAdapter(ctx),
      gpuTimingSupported: ctx.hasTimestamp,
      memoryEstimateMB: 0,
      degraded: this.downgrades,
    };
  }

  static async create(
    canvas: HTMLCanvasElement,
    profile: DeviceProfile,
    params: Params,
    telemetry: RendererTelemetry,
    reducedMotion = false
  ): Promise<GpuRenderer> {
    let fatal = '';
    const ctx = await createGpuContext(canvas, {
      onMessage: (level, text) => {
        if (level === 'error') fatal = text;
      },
    });
    const device = ctx.device;
    const downgrades: string[] = [];

    const simUniform = device.createBuffer({
      label: 'sim-uniform',
      size: SIM_UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // ---- particle budget -------------------------------------------------
    const adapterMax = maxParticlesForDevice(ctx);
    const requested = Math.min(params.simCount, HARD_MAX_SIM_PARTICLES);
    let simCount = Math.min(requested, adapterMax);
    const memCap = Math.floor((380 * 1024 * 1024) / (88 * 2));
    if (simCount > memCap) {
      downgrades.push(`particles capped at ${memCap.toLocaleString()} by the 380 MB budget`);
      simCount = memCap;
    }
    if (simCount < requested) {
      downgrades.push(`adapter storage limit capped particles at ${simCount.toLocaleString()}`);
    }
    params.simCount = simCount;

    const particles = await ParticleSystem.create(
      ctx,
      simUniform,
      { capacity: simCount, maxParticles: Math.min(adapterMax, HARD_MAX_SIM_PARTICLES) },
      (msg) => {
        fatal = msg;
      }
    );
    const post = await PostChain.create(
      ctx,
      { bloomLevels: profile.bloomLevels, dof: profile.dof, volumetric: profile.volumetric },
      (msg) => {
        fatal = msg;
      }
    );

    // ---- particle pipeline ----------------------------------------------
    const vs = await compileChecked(device, PARTICLE_VERTEX_WGSL, 'particle.vert.wgsl');
    const fs = await compileChecked(device, PARTICLE_FRAGMENT_WGSL, 'particle.frag.wgsl');
    if (vs.errors.length || fs.errors.length) fatal = [...vs.errors, ...fs.errors].join('\n');

    /**
     * Two variants of the particle pipeline, differing only in the depth
     * attachment.
     *
     * Writing depth for every particle costs a full-width depth write plus a
     * late-Z update per fragment, at a million instances. The DOF pass reads
     * that depth, so it cannot simply be removed - but it can be made optional,
     * which lets the profiling tools measure exactly what it costs instead of
     * arguing about it.
     */
    const buildParticlePipeline = (noDepth: boolean) =>
      device.createRenderPipeline({
        label: noDepth ? 'particles-no-depth' : 'particles',
        layout: device.createPipelineLayout({ bindGroupLayouts: [particles.renderLayout] }),
        vertex: { module: vs.module, entryPoint: 'vs_main' },
        fragment: {
          module: fs.module,
          entryPoint: 'fs_main',
          targets: [
            {
              format: 'rgba16float',
              // Additive HDR: the sprite alpha is premultiplied in the fragment
              // shader, so overlapping particles integrate light linearly.
              blend: {
                color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
                alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              },
            },
          ],
        },
        primitive: { topology: 'triangle-list' },
        // Depth *test* is intentionally disabled: additive light must never be
        // occluded by other light. Only the write remains, and only when the
        // variant asks for it.
        ...(noDepth
          ? {}
          : {
              depthStencil: {
                format: 'depth32float' as GPUTextureFormat,
                depthWriteEnabled: true,
                depthCompare: 'always' as GPUCompareFunction,
              },
            }),
      });

    const particlePipeline = buildParticlePipeline(false);
    const particlePipelineNoDepth = buildParticlePipeline(true);

    const perCallFromBuffer = Math.floor(ctx.limits.maxBufferSize / 16) - 1;
    const particlesPerCall = Math.max(65_536, Math.min(DEFAULT_INSTANCES_PER_CALL, perCallFromBuffer));

    const renderer = new GpuRenderer(
      ctx,
      profile,
      params,
      telemetry,
      particles,
      post,
      simUniform,
      particlePipeline,
      particlePipelineNoDepth,
      particlesPerCall,
      reducedMotion
    );
    renderer.downgrades.push(...downgrades);
    renderer.setupTiming();
    renderer.renderScale = clamp(profile.renderScale, 0.4, profile.maxPostScale);
    renderer.targetScale = renderer.renderScale;
    renderer.cssWidth = canvas.clientWidth || window.innerWidth;
    renderer.cssHeight = canvas.clientHeight || window.innerHeight;
    renderer.cssDpr = profile.dpr;
    renderer.applyCanvasSize();
    renderer.syncStats();
    if (fatal) telemetry.onFatal?.('WebGPU setup failed', fatal);
    return renderer;
  }

  private setupTiming(): void {
    if (!this.ctx.hasTimestamp) return;
    // 24 passes covers the deepest frame: fade, compute, particles, the bloom
    // descent and ascent, god rays, DOF and the final two stages, with headroom.
    this.timer = new PassTimer(this.ctx.device, 24);
  }


  // -------------------------------------------------------------------------
  // Sizing
  // -------------------------------------------------------------------------
  resize(cssWidth: number, cssHeight: number, dpr: number, scale: number): void {
    this.cssWidth = cssWidth;
    this.cssHeight = cssHeight;
    this.cssDpr = dpr;
    this.renderScale = clamp(scale, 0.4, this.profile.maxPostScale * 1.4);
    this.applyCanvasSize();
  }

  setQualityScale(scale: number): void {
    this.targetScale = clamp(scale, 0.4, this.profile.maxPostScale * 1.4);
  }

  private applyCanvasSize(): void {
    const canvas = this.ctx.canvas;
    const dpr = Math.max(0.4, this.cssDpr);
    const w = Math.max(64, Math.floor(this.cssWidth * dpr * this.renderScale));
    const h = Math.max(64, Math.floor(this.cssHeight * dpr * this.renderScale));
    if (canvas.width === w && canvas.height === h && this.stats.drawWidth === w) return;
    canvas.width = w;
    canvas.height = h;
    this.post.resize(w, h);
    this.stats.drawWidth = w;
    this.stats.drawHeight = h;
    this.stats.renderScale = this.renderScale;
  }

  // -------------------------------------------------------------------------
  // Interaction
  // -------------------------------------------------------------------------
  orbit(dx: number, dy: number, plane?: 'xw' | 'yw' | 'zw'): void {
    this.idleTimer = 0;
    if (plane === 'xw') {
      this.rot[2] += dx * 0.006;
      this.rot[3] += dy * 0.006;
    } else if (plane === 'yw') {
      this.rot[4] += dx * 0.006;
      this.rot[5] += dy * 0.006;
    } else if (plane === 'zw') {
      this.rot[5] += dx * 0.006;
      this.rot[0] += dy * 0.006;
    } else {
      // Default drag drives the two planes the eye reads as tumbling, because
      // the hyperplane slice is what the viewer interprets as depth.
      this.rot[2] += dx * 0.0055;
      this.rot[3] += dy * 0.0055;
      this.rot[0] += dx * 0.0016;
      this.rot[4] += dy * 0.0016;
    }
    for (let i = 0; i < 6; i++) {
      const a = this.rot[i] as number;
      if (a > Math.PI) this.rot[i] = a - TAU;
      else if (a < -Math.PI) this.rot[i] = a + TAU;
    }
  }

  zoom(delta: number): void {
    this.idleTimer = 0;
    this.state.zoom = clamp(this.state.zoom * Math.exp(delta * 0.0012), 1.3, 14);
  }

  slice(delta: number): void {
    this.idleTimer = 0;
    this.state.wSlice = clamp(this.state.wSlice + delta * 0.0016, -3.2, 3.2);
  }

  shock(ndcX: number, ndcY: number, strength = 1): void {
    if (this.params.shock <= 0) return;
    // Unproject through the inverse MVP so the supernova ignites where the
    // pointer actually pointed, at the current focal depth.
    const head = xform4(this.invMvp, ndcX, ndcY, 0.0, 1.0);
    const tail = xform4(this.invMvp, ndcX, ndcY, 1.0, 1.0);
    const hw = Math.abs(head[3]) < 1e-6 ? 1e-6 : head[3];
    const tw = Math.abs(tail[3]) < 1e-6 ? 1e-6 : tail[3];
    const px = head[0] / hw;
    const py = head[1] / hw;
    const pz = head[2] / hw;
    let dx = tail[0] / tw - px;
    let dy = tail[1] / tw - py;
    let dz = tail[2] / tw - pz;
    const len = Math.hypot(dx, dy, dz) || 1;
    dx /= len;
    dy /= len;
    dz /= len;

    const dist = clamp(this.state.focusDist, 1.2, 8);
    const origin: [number, number, number] = [px + dx * dist, py + dy * dist, pz + dz * dist];
    const event: ShockEvent = { origin, age: 0, strength: strength * this.params.shock };
    if (this.shocks.length < MAX_SHOCKS) this.shocks.push(event);
    else {
      this.shocks[this.shockCursor % MAX_SHOCKS] = event;
      this.shockCursor++;
    }
  }

  setAudio(low: number, mid: number, high: number, beat: number): void {
    this.state.audioLow = low;
    this.state.audioMid = mid;
    this.state.audioHigh = high;
    this.state.beat = beat;
  }

  reset(): void {
    this.rot = [0.35, 0.12, 0.85, -0.22, 0.55, 0.1];
    this.state.zoom = DEFAULT_CAMERA_DISTANCE;
    this.state.wSlice = 0;
    this.shocks.length = 0;
    this.shockCursor = 0;
    this.elapsed = 0;
    this.particles.seed();
  }

  async capture(): Promise<Blob | null> {
    const canvas = this.ctx.canvas;
    return await new Promise<Blob | null>((resolve) => {
      if (typeof canvas.toBlob === 'function') canvas.toBlob((b) => resolve(b), 'image/png');
      else resolve(null);
    });
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------
  frame(nowMs: number, deltaMs: number): void {
    const t0 = performance.now();
    const device = this.ctx.device;

    const rawDt = this.started ? Math.min(deltaMs, 100) / 1000 : 1 / 60;
    this.started = true;
    const dt = rawDt;

    if (this.frameIndex % 30 === 0) this.adapt();
    if (!this.params.paused) {
      this.elapsed += dt;
      this.frameIndex++;
    } else {
      // A trickle of time keeps grain and god rays alive while paused, so the
      // image still breathes instead of looking like a frozen screenshot.
      this.elapsed += dt * 0.06;
    }
    const t = this.elapsed;

    // ---- idle auto-rotation ----------------------------------------------
    this.idleTimer += dt;
    if (this.idleAutoRotate && this.idleTimer > 2.2 && !this.params.paused) {
      const k = Math.min(this.idleTimer - 2.2, 1.2);
      this.rot[0] += dt * 0.042 * k;
      this.rot[4] += dt * 0.026 * k;
      this.rot[5] += dt * 0.018 * k;
    }

    // ---- 4D rotation + camera --------------------------------------------
    const s = this.state;
    s.rot0 = [this.rot[0], this.rot[1], this.rot[2], this.rot[3]];
    s.rot1 = [this.rot[4], this.rot[5], Math.tan(this.fovY / 2), t];
    s.rot4.set(rotationMatrix4(this.rot));

    const aspect = this.stats.drawWidth / Math.max(1, this.stats.drawHeight);
    s.eye = [Math.cos(t * 0.09) * 0.1, Math.sin(t * 0.13) * 0.12, s.zoom];
    s.aspect = aspect;
    perspective(this.fovY, aspect, NEAR, FAR, this.proj);
    lookAt(s.eye, [0, 0, 0], [0, 1, 0], this.view);
    s.proj.set(this.proj);
    s.view.set(this.view);
    mulMat4(this.proj, this.view, this.mvp);
    invert4(this.mvp, this.invMvp);

    // Focus sits at the cloud centre, nudged by the hyperplane slice so the
    // near and far sides of the 4D cut receive different blur.
    s.focusDist = clamp(s.zoom - s.wSlice * 0.25, 1.0, 20);
    s.aperture = this.params.dof * 0.5;

    // ---- shockwaves -------------------------------------------------------
    if (!this.params.paused) {
      for (const sh of this.shocks) sh.age += dt;
      if (this.shocks.length > 1 || (this.shocks[0]?.age ?? 0) > 0) {
        this.shocks = this.shocks.filter((sh) => sh.age < 2.4);
      }
    }
    this.particles.tickShocks(this.shocks);

    // ---- particle budget --------------------------------------------------
    /*
     * The instance -> particle mapping in the vertex shader needs a power-of-two
     * stride to avoid an integer modulo, so the count of *live* particles here is
     * always a power of two. That is the whole reason for the rounding.
     *
     * Copies first, primary particles last: the drawn ranges are then
     * [0, stride) for copy 0, [stride, 2*stride) for copy 1, and so on, which is
     * exactly what `f32(inst) - stride * floor(f32(inst) / stride)` decodes.
     *
     * How many the frame can afford depends on how large the sprites are. Two
     * measurements set that relationship. A fill-rate ablation (sprite radius
     * forced to about a pixel) halved the particle pass, so fill is a real term;
     * the same ablation still cost ~29 ms at 1.57M drawn, which is per-instance
     * vertex work that ignores sprite size entirely. Both terms scale with the
     * drawn count, and fill additionally scales with radius squared.
     *
     * Deriving the budget from the radius keeps visual density roughly constant -
     * larger sprites merging into gas at a lower count, smaller ones forming dense
     * star fields at a higher count - while the frame cost stays bounded.
     *
     * The constant is fitted to measured cost, and it is deliberately conservative:
     * the particle pass is the dominant term in the frame, so the budget aims it at
     * roughly half of a 16.6 ms frame and leaves the resolution ladder and the post
     * levels to account for the rest.
     *
     * On the reference machine the pass costs about 55 ns per drawn particle at the
     * nominal sprite sizes, which puts the two calibration points at roughly 135k
     * drawn particles for an 11 px radius and 230k for a 5 px radius, fitting
     * budget ~ 6.0e6 / (r^1.1 + 2.1e3 / r).
     *
     * That figure was measured through a headless compositor, so treat it as a
     * floor: a foreground window with a real swap chain should sustain more.
     * Raising SHOW PARTICLES spends more of the frame on the cloud by choice;
     * raising STAR SIZE trades count for per-sprite size at roughly constant cost.
     */
    const spriteRadius = 3.6 * this.params.particleSize;
    const affordable = 6.0e6 / (Math.pow(spriteRadius, 1.1) + 2.1e3 / spriteRadius);
    const wantShow = clamp(Math.floor(this.params.showCount), 1, MAX_RENDER_PARTICLES);
    const showTarget = Math.min(wantShow, clamp(Math.floor(affordable), 60_000, MAX_RENDER_PARTICLES));

    // The count is exact, not rounded to a power of two: the vertex shader decodes
    // the instance index with an exact float division, so the budget can be
    // honoured at whatever value it computes. Power-of-two rounding used to
    // overshoot the budget by nearly 2x at unlucky sizes.
    const liveSim = Math.max(64, this.particles.count);
    const multiplicity = clamp(Math.ceil(showTarget / liveSim), 1, 8);
    const stride = liveSim;
    const drawnTotal = stride * multiplicity;

    // ---- uniforms ---------------------------------------------------------
    const sens = this.params.audioSensitivity;
    s.time = t;
    s.dt = dt;
    s.frame = this.frameIndex;
    s.simCount = stride;
    s.width = this.stats.drawWidth;
    s.height = this.stats.drawHeight;
    s.julia = this.params.julia;
    s.curl = this.params.curl;
    s.damping = this.params.damping;
    s.gravity = this.params.gravity;
    s.confinement = this.params.confinement;
    s.attractDepth = this.params.fractalDim;
    s.power = 2.35 + this.params.fractalDim * 4.15;
    s.warp = 0.22 + this.params.fractalDim * 0.63;
    s.hue = this.params.hue;
    s.paletteMix = 0.55 + s.audioMid * 0.25;
    s.saturation = 1.05;
    s.exposure = this.params.exposure;
    s.sizeScale = this.params.particleSize;
    s.energyScale = 0.75;
    s.multiplicity = multiplicity;
    s.speed = this.params.speed;
    s.shockGain = this.params.shock;
    // The particle shader decodes the instance index against this power-of-two
    // stride instead of taking an integer modulo.
    s.stride = stride;
    s.bloom = this.params.bloom;
    s.dpr = this.renderScale;
    s.grain = this.params.grain;
    s.vignette = this.params.vignette;
    s.activeShocks = this.particles.activeShocks;
    s.trailDecay = this.params.trails;
    // Sprite-scale ablation (see particle.wgsl.ts); always 1 outside a profile run.
    s.spriteScale = this.ablation === 'small' ? 0.06 : 1;
    s.audioLow = clamp(s.audioLow * (0.6 + sens * 0.4), 0, 3);
    s.audioMid = clamp(s.audioMid * (0.6 + sens * 0.4), 0, 3);
    s.audioHigh = clamp(s.audioHigh * (0.6 + sens * 0.4), 0, 3);
    device.queue.writeBuffer(this.simUniform, 0, this.simWriter.write(s));

    const p = this.postState;
    p.width = this.stats.drawWidth;
    p.height = this.stats.drawHeight;
    p.exposure = this.params.exposure;
    p.bloom = this.params.bloom;
    p.bloomRadius = this.params.bloomRadius;
    p.threshold = this.params.bloomThreshold;
    p.chroma = this.params.chroma;
    p.grain = this.params.grain;
    p.vignette = this.params.vignette;
    p.dof = this.params.dof;
    p.volumetric = this.params.volumetric;
    p.time = t;
    p.aspect = aspect;
    // Grain and god rays must evolve on their own clock while paused.
    p.frame = this.frameIndex;
    p.focusDepth = s.focusDist;
    p.focusRange = 1;
    p.maxCoc = 0.9;
    p.nearPlane = NEAR;
    p.farPlane = FAR;
    p.sourceLod = 0;
    p.upsampleRadius = 1;
    p.bloomLevels = this.post.bloomLevelCount;
    p.hue = 0; // the palette grade is applied per particle, not per pixel
    p.saturation = 1;
    // Additive emission integrates with the emitter count, so the composite gets
    // a 1/sqrt(N) pre-exposure. sqrt rather than N keeps "more particles" a
    // visible reward (denser, brighter nebula) without clipping to white.
    // Calibrated so the nebula's filament cores land near 0.9 after ACES while the
    // empty sky stays under 0.02. Derived from a measured exposure sweep, not a guess.
    p.quality = 4.45 / Math.sqrt(Math.max(1, drawnTotal));
    // Every pass owns a uniform slot (they need different texel sizes), so the
    // whole frame's post uniforms are published in one call.
    this.post.prepareFrame(p, this.params.trails, s.focusDist, this.params.dof * 0.5);

    // ---- encode -----------------------------------------------------------
    const swapTexture = this.ctx.context.getCurrentTexture();
    const canvasView = swapTexture.createView();
    const encoder = device.createCommandEncoder({ label: 'aetheria-frame' });

    // The timer's slot list describes exactly one frame. Without this reset the
    // labels accumulate forever and the reported durations become differences
    // between timestamps from different frames, which is how a 0.2 ms pass was
    // once reported as 24 ms.
    this.timer?.reset();

    // The simulation step is bracketed like every other pass, so the per-pass
    // breakdown covers the whole frame with no gaps.
    const simTiming = this.markPass(encoder, 'simulate');
    this.particles.step(encoder, simTiming);
    if (simTiming) this.timer?.close(encoder);

    // Trail feedback must precede the particle pass: it decays last frame's
    // accumulated HDR into this frame's accumulation buffer, which the particles
    // then load and add to. Skipping it would leave a stale image under the cloud.
    if (this.timer) this.post.setTimer(this.timer, true);
    this.post.fadeScene(encoder);

    const ts = this.markPass(encoder, 'particles');
    // An ablation can drop the depth attachment entirely, which changes both the
    // pipeline and the pass descriptor, so they are chosen together.
    const noDepth = this.ablation === 'nodepth' || this.ablation === 'nz';
    const passDesc: GPURenderPassDescriptor = {
      label: 'particles',
      colorAttachments: [{ view: this.post.sceneView, loadOp: 'load', storeOp: 'store' }],
      ...(noDepth
        ? {}
        : {
            depthStencilAttachment: {
              view: this.post.depthView,
              depthLoadOp: 'clear' as GPULoadOp,
              depthStoreOp: 'store' as GPUStoreOp,
              depthClearValue: 1,
            },
          }),
      ...(ts ? { timestampWrites: ts } : {}),
    };
    const pass = encoder.beginRenderPass(passDesc);
    if (this.ablation === 'skipdraw') {
      // Draw nothing, but still clear and store the attachments so the rest of the
      // frame is unaffected.
      pass.end();
      if (ts) this.timer?.close(encoder);
      this.post.commitFrame();
      this.post.setQuality(this.params.dof > 0 && this.postLevel >= 2, this.params.volumetric > 0 && this.postLevel >= 2);
      this.post.run(encoder, canvasView);
      device.queue.submit([encoder.finish()]);
      this.finishTiming();
      const cpuMs2 = performance.now() - t0;
      this.cpuEma += (cpuMs2 - this.cpuEma) * 0.08;
      this.rttEma += (rawDt * 1000 - this.rttEma) * 0.08;
      this.fpsEma += (1 / Math.max(rawDt, 1e-4) - this.fpsEma) * 0.08;
      this.syncStats();
      void nowMs;
      return;
    }
    pass.setPipeline(noDepth ? this.particlePipelineNoDepth : this.particlePipeline);
    pass.setBindGroup(0, this.particles.renderBindGroup);
    this.drawParticles(pass, drawnTotal);
    pass.end();
    if (ts) this.timer?.close(encoder);

    // This frame's accumulation becomes next frame's history. Swapping roles is
    // what removes the need for a full-resolution copy per frame.
    this.post.commitFrame();

    this.post.setQuality(
      this.params.dof > 0 && this.postLevel >= 2,
      this.params.volumetric > 0 && this.postLevel >= 2
    );
    this.post.run(encoder, canvasView);
    device.queue.submit([encoder.finish()]);
    this.finishTiming();

    // Opt-in self test: the copy is recorded in a *second* submission after the
    // frame, because reading a mappable buffer that a pending submission is still
    // writing yields zero bytes.
    this.selfTest?.consume(device, swapTexture);

    // ---- telemetry --------------------------------------------------------
    const cpuMs = performance.now() - t0;
    this.cpuEma += (cpuMs - this.cpuEma) * 0.08;
    this.rttEma += (rawDt * 1000 - this.rttEma) * 0.08;
    this.fpsEma += (1 / Math.max(rawDt, 1e-4) - this.fpsEma) * 0.08;
    this.stats.fps = this.fpsEma;
    this.stats.frameMs = this.rttEma;
    this.stats.cpuMs = this.cpuEma;
    this.stats.gpuMs = this.gpuMs;
    this.stats.computeMs = this.computeMs;
    this.stats.simCount = stride;
    this.stats.renderCount = drawnTotal;
    this.stats.culledCount = this.culledByBudget;
    this.stats.renderScale = this.renderScale;
    this.stats.memoryEstimateMB =
      (estimateParticleMemory(this.particles.capacity) + this.post.estimateBytes()) / 1048576;
    this.stats.degraded = this.downgrades;
    void nowMs;
  }

  /**
   * The particle pass draws `simCount * multiplicity` instances, split across as
   * many calls as the per-call instance ceiling requires. `firstInstance` carries
   * the global instance id into the shader, which decodes both which particle and
   * which sub-copy it is drawing.
   */
  private drawParticles(pass: GPURenderPassEncoder, total: number): void {
    const perCall = this.particlesPerCall;
    let drawn = 0;
    let calls = 0;
    while (drawn < total && calls < 64) {
      const n = Math.min(perCall, total - drawn);
      pass.draw(VERTICES_PER_QUAD, n, 0, drawn);
      drawn += n;
      calls++;
    }
    this.stats.drawVertices = drawn * VERTICES_PER_QUAD;
  }

  // -------------------------------------------------------------------------
  // Adaptation
  // -------------------------------------------------------------------------
  private adapt(): void {
    // A profiling run needs the quality level held still: the adaptive loop would
    // otherwise move the resolution underneath the measurement, so the numbers
    // would describe two different configurations rather than one.
    if (this.adaptationFrozen) return;

    // Warm-up grace: the first frames include pipeline compilation, the initial
    // seed dispatch and the browser's own first-frame work. Reacting to those
    // would start every session in a degraded state.
    if (this.frameIndex < 90) {
      this.renderScale += (this.targetScale - this.renderScale) * 0.25;
      this.applyCanvasSize();
      return;
    }

    const budget = this.profile.isMobile ? 33.3 : 16.6;

    /*
     * Choose the cost signal deliberately.
     *
     * rAF deltas measure the *browser's* throughput, which includes compositor
     * backpressure, a throttled background tab, and any other page on the
     * machine. Adapting to that made the renderer shed quality for reasons that
     * had nothing to do with it — in headless testing it degraded to 8% of the
     * particle budget while the GPU stage took 16 ms of a 60 ms frame.
     *
     * With timestamp queries available the GPU stages are measured directly, so
     * the control loop responds to its own cost. Without them, rAF is the only
     * signal there is, and it is used with a wider tolerance.
     */
    const gpuTotal = this.gpuMs + this.computeMs;
    const measured = this.ctx.hasTimestamp && gpuTotal > 0.05 ? gpuTotal : this.rttEma;
    const tolerance = this.ctx.hasTimestamp ? 1 : 1.35;

    /*
     * The drawn-particle count is capped by its share of the frame, and this runs
     * BEFORE the resolution ladder.
     *
     * Rasterising particles is the one cost that scales linearly with a number the
     * user sets directly, and it is the largest term in the frame by an order of
     * magnitude: measured at roughly 20 ns per drawn particle against a whole post
     * chain that costs under 2 ms. Shedding resolution to pay for it is the wrong
     * lever - it reduces the quality of everything else to fund the one stage that
     * is overspent - so the particle budget is trimmed first and by proportion,
     * which converges in a frame or two instead of a percent at a time.
     */
    // 1. Resolution: cheapest to shed, least visible.
    if (measured > budget * (this.ctx.hasTimestamp ? 1.25 : 1.18) && this.targetScale > 0.55) {
      this.targetScale = Math.max(0.55, this.targetScale - 0.06);
      this.markDegrade('resolution');
    } else if (measured < budget * 0.85 * tolerance && this.targetScale < this.profile.maxPostScale) {
      this.targetScale = Math.min(this.profile.maxPostScale, this.targetScale + 0.03);
    }

    // 2. Post effects.
    if (measured > budget * 1.55 && this.postLevel > 0) {
      this.postLevel--;
      this.markDegrade(this.postLevel === 1 ? 'depth of field + volumetric light' : 'post-processing');
    } else if (measured < budget * 0.72 * tolerance && this.postLevel < 2) {
      this.postLevel++;
    }

    // 3. Simulated particles: last resort, and rarely needed, because the
    //    simulation measures about 0.2 ms at a million particles. Shrinking the
    //    store invalidates every particle's history, so it carries strong
    //    hysteresis.
    if (measured > budget * 1.8 && this.computeMs > budget * 0.25 && this.params.simCount > 150_000) {
      const next = Math.max(150_000, Math.floor(this.params.simCount * 0.82));
      this.params.simCount = next;
      this.particles.setSimCount(next);
      this.markDegrade('simulated particles');
    }

    this.renderScale += (this.targetScale - this.renderScale) * 0.25;
    this.applyCanvasSize();
  }

  private markDegrade(what: string): void {
    const label = `auto-degraded: ${what}`;
    if (!this.downgrades.includes(label)) {
      this.downgrades.push(label);
      this.telemetry.onDegrade?.(label);
      if (this.downgrades.length > 6) this.downgrades.shift();
    }
  }

  private syncStats(): void {
    this.stats.memoryEstimateMB =
      (estimateParticleMemory(this.particles.capacity) + this.post.estimateBytes()) / 1048576;
  }

  // -------------------------------------------------------------------------
  // GPU timing
  //
  // One shared PassTimer brackets every pass in the frame, so the numbers add up
  // to the frame's real GPU cost and each stage can be attributed. The adaptive
  // loop reads the totals; ?profile=1 prints the breakdown.
  // -------------------------------------------------------------------------
  /** Bracket a pass and return the descriptor to attach, or undefined. */
  private markPass(encoder: GPUCommandEncoder, label: string): GPUComputePassTimestampWrites | undefined {
    return this.timer ? this.timer.bracket(encoder, label, true) : undefined;
  }

  /**
   * Close the frame's brackets, resolve them, and read the breakdown back.
   *
   * Sampling is throttled rather than done every frame: a readback per frame
   * would add a map/unmap pair and a queue submission to the very loop being
   * measured, and the pass contents change only when quality or the pass list
   * does. Once every sixth frame keeps the numbers representative while leaving
   * the measured frame almost untouched.
   */
  private finishTiming(): void {
    const timer = this.timer;
    if (!timer) return;
    timer.finish();
    if (this.profileCountdown > 0) {
      this.profileCountdown--;
      // Still resolve so the brackets do not accumulate, but skip the map.
      void timer.collect();
      return;
    }
    this.profileCountdown = 5;
    void timer.collect().then((timings) => {
      if (!timings.length) return;
      this.lastTimings = timings;
      let total = 0;
      let compute = 0;
      for (const t of timings) {
        total += t.ms;
        if (t.label.startsWith('simulate')) compute += t.ms;
      }
      this.gpuMs = total;
      this.computeMs = compute;
    });
  }

  private profileCountdown = 0;
  private adaptationFrozen = false;
  /** Drawn particles the frame could not afford, for the HUD. */
  private culledByBudget = 0;

  /** Per-pass GPU breakdown from the most recent profiled frame. */
  get passTimings(): PassTiming[] {
    return this.lastTimings;
  }

  /** Hold the quality level still for a profiling run. */
  freezeAdaptation(frozen: boolean): void {
    this.adaptationFrozen = frozen;
  }

  /** Select an ablation for the particle pass. Used by the profiling tools. */
  setAblation(mode: string): void {
    this.ablation = mode;
  }


  dispose(): void {
    this.particles.dispose();
    this.post.dispose();
    this.simUniform.destroy();
    this.timer?.dispose();
    this.timer = null;
  }

  /** Live camera framing, settable so the calibration runner can measure it. */
  setFraming(distance: number, fovY: number): void {
    this.state.zoom = clamp(distance, 0.5, 14);
    this.fovY = clamp(fovY, 0.4, 2.4);
  }

  /** Compute-stage GPU time, exposed for the HUD and the adaptation tests. */
  get computeTimeMs(): number {
    return this.computeMs;
  }

  /** Effective render scale the controller settled on (used by the HUD). */
  get currentScale(): number {
    return this.renderScale;
  }

  get ready(): boolean {
    return this.particles.count > 0;
  }
}

function describeAdapter(ctx: GpuContext): string {
  const i = ctx.info;
  const parts = [i.vendor, i.architecture, i.device, i.description].filter((x) => x && x !== '');
  return parts.length ? parts.join(' / ') : 'WebGPU adapter';
}

export { BackendUnavailable };
