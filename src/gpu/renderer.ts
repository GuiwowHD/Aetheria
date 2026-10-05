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
  private readonly particlesPerCall: number;
  private readonly querySets: GPUQuerySet[] = [];
  private readonly queryBufs: GPUBuffer[] = [];
  private readonly queryBusy: boolean[] = [];

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
  private gpuMs = 0;
  private queryIndex = 0;
  private pendingQuery: { slot: number; descriptor: GPURenderPassTimestampWrites } | null = null;

  private readonly downgrades: string[] = [];
  private postLevel = 2; // 2 = full, 1 = no DOF/god rays, 0 = bloom only
  /** Dedicated query slot for the compute stage (slot 0 is the render pass). */
  private computeSlot = 1;
  private computeMs = 0;
  private computePending = false;
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

    const particlePipeline = device.createRenderPipeline({
      label: 'particles',
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
      depthStencil: {
        format: 'depth32float',
        depthWriteEnabled: true,
        // Depth *test* is intentionally disabled: additive light must never be
        // occluded by other light. Depth is still written because DOF reads it.
        depthCompare: 'always',
      },
    });

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
    const device = this.ctx.device;
    for (let i = 0; i < 3; i++) {
      this.querySets.push(device.createQuerySet({ label: `ts${i}`, type: 'timestamp', count: 2 }));
      // A buffer with MAP_READ may only also carry COPY_DST, so the query is
      // resolved into a GPU-side buffer and then copied into a staging buffer
      // that the CPU can map. Resolving straight into a mappable buffer is a
      // validation error, not a warning.
      this.queryResolve.push(
        device.createBuffer({
          label: `ts-resolve${i}`,
          size: 16,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        })
      );
      this.queryBufs.push(
        device.createBuffer({
          label: `ts-readback${i}`,
          size: 16,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        })
      );
      this.queryBusy.push(false);
    }
  }

  private readonly queryResolve: GPUBuffer[] = [];

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
    const simCount = this.particles.count;
    const wantShow = clamp(Math.floor(this.params.showCount), 1, MAX_RENDER_PARTICLES);
    const showCount = Math.min(wantShow, simCount * 8);
    const multiplicity = clamp(showCount / simCount, 1, 8);
    const effectiveSim = clamp(Math.floor(simCount / multiplicity), 64, simCount);

    // ---- uniforms ---------------------------------------------------------
    const sens = this.params.audioSensitivity;
    s.time = t;
    s.dt = dt;
    s.frame = this.frameIndex;
    s.simCount = effectiveSim;
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
    s.bloom = this.params.bloom;
    s.dpr = this.renderScale;
    s.quality = this.postLevel / 2;
    s.grain = this.params.grain;
    s.vignette = this.params.vignette;
    s.activeShocks = this.particles.activeShocks;
    s.trailDecay = this.params.trails;
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
    p.quality = 4.45 / Math.sqrt(Math.max(1, effectiveSim * multiplicity));
    // Every pass owns a uniform slot (they need different texel sizes), so the
    // whole frame's post uniforms are published in one call.
    this.post.prepareFrame(p, this.params.trails, s.focusDist, this.params.dof * 0.5);

    // ---- encode -----------------------------------------------------------
    const swapTexture = this.ctx.context.getCurrentTexture();
    const canvasView = swapTexture.createView();
    const encoder = device.createCommandEncoder({ label: 'aetheria-frame' });

    // A second timestamp slot measures the compute stage on its own. The
    // simulation cost is what the adaptation ladder most needs: it is the term
    // that a frame-time average hides completely when the browser is throttling
    // presentation (a background tab, or headless compositor backpressure).
    this.particles.step(encoder, this.beginComputeTiming());

    // Trail feedback must precede the particle pass: it decays last frame's
    // accumulated HDR into this frame's accumulation buffer, which the particles
    // then load and add to. Skipping it would leave a stale image under the cloud.
    this.post.fadeScene(encoder);

    const ts = this.beginTiming();
    const passDesc: GPURenderPassDescriptor = {
      label: 'particles',
      colorAttachments: [{ view: this.post.sceneView, loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: {
        view: this.post.depthView,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
        depthClearValue: 1,
      },
    };
    if (ts) passDesc.timestampWrites = ts;
    const pass = encoder.beginRenderPass(passDesc);
    pass.setPipeline(this.particlePipeline);
    pass.setBindGroup(0, this.particles.renderBindGroup);
    this.drawParticles(pass, effectiveSim, multiplicity);
    pass.end();

    // This frame's accumulation becomes next frame's history. Swapping roles is
    // what removes the need for a full-resolution copy per frame.
    this.post.commitFrame();

    this.post.setQuality(
      this.params.dof > 0 && this.postLevel >= 2,
      this.params.volumetric > 0 && this.postLevel >= 2
    );
    this.post.run(encoder, canvasView);
    device.queue.submit([encoder.finish()]);
    this.endTiming(device);

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
    this.stats.simCount = effectiveSim;
    this.stats.renderCount = Math.floor(effectiveSim * multiplicity);
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
  private drawParticles(pass: GPURenderPassEncoder, simCount: number, multiplicity: number): void {
    const total = Math.floor(simCount * multiplicity);
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

    // 3. Particles: last resort, with strong hysteresis because shrinking the
    //    store invalidates every particle's history.
    if (measured > budget * 1.8 && this.params.simCount > 150_000) {
      const next = Math.max(150_000, Math.floor(this.params.simCount * 0.82));
      this.params.simCount = next;
      this.particles.setSimCount(next);
      this.markDegrade('particle count');
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
  // GPU timing (timestamp-query, opportunistic and non-blocking)
  // -------------------------------------------------------------------------
  /**
   * Timing for the compute stage, using the slot that the particle pass is not
   * using this frame. Returns undefined when timestamps are unavailable or the
   * chosen slot still has an outstanding map.
   */
  private beginComputeTiming(): GPUComputePassTimestampWrites | undefined {
    if (!this.ctx.hasTimestamp || this.querySets.length < 2) return undefined;
    const slot = this.computeSlot;
    if (this.queryBusy[slot]) return undefined;
    this.computePending = true;
    return { querySet: this.querySets[slot]!, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 };
  }

  private beginTiming(): GPURenderPassTimestampWrites | undefined {
    if (!this.ctx.hasTimestamp || this.pendingQuery || this.querySets.length === 0) return undefined;
    const slot = this.queryIndex % this.querySets.length;
    if (this.queryBusy[slot]) return undefined;
    const set = this.querySets[slot]!;
    // Pass-level timestamps: the pass writes index 0 on entry and index 1 on
    // exit, which avoids depending on GPUCommandEncoder.writeTimestamp (a newer
    // addition that is not present in every implementation).
    this.pendingQuery = {
      slot,
      descriptor: { querySet: set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    };
    return this.pendingQuery.descriptor;
  }

  private endTiming(device: GPUDevice): void {
    const pending = this.pendingQuery;
    if (!pending) {
      this.measureCompute(device);
      return;
    }
    this.pendingQuery = null;
    const slot = pending.slot;

    // Both stages are resolved in one submission: the render pass timing from
    // `pending`, and the compute pass timing from `computeSlot` (they are always
    // different slots, which is why two query sets are allocated).
    const enc = device.createCommandEncoder({ label: 'timestamp-resolve' });
    enc.resolveQuerySet(pending.descriptor.querySet, 0, 2, this.queryResolve[slot]!, 0);
    enc.copyBufferToBuffer(this.queryResolve[slot]!, 0, this.queryBufs[slot]!, 0, 16);
    const computeSlot = this.computeSlot;
    if (this.computePending) {
      this.computePending = false;
      enc.resolveQuerySet(this.querySets[computeSlot]!, 0, 2, this.queryResolve[computeSlot]!, 0);
      enc.copyBufferToBuffer(this.queryResolve[computeSlot]!, 0, this.queryBufs[computeSlot]!, 0, 16);
    }
    device.queue.submit([enc.finish()]);

    this.queryBusy[slot] = true;
    this.queryIndex++;

    // The render-pass timer brackets only the particle pass, so it is reported
    // as the draw cost rather than as the whole frame.
    this.readBack(device, slot, (ms) => {
      this.gpuMs = this.gpuMs * 0.85 + ms * 0.15;
    });
    if (this.queryBusy[computeSlot]) return;
    this.readBack(device, computeSlot, (ms) => {
      this.computeMs = this.computeMs * 0.85 + ms * 0.15;
    });
  }

  /** Resolve the compute stage alone when no render timing was recorded. */
  private measureCompute(device: GPUDevice): void {
    const slot = this.computeSlot;
    if (!this.computePending || this.queryBusy[slot]) return;
    this.computePending = false;
    const enc = device.createCommandEncoder({ label: 'compute-timestamp-resolve' });
    enc.resolveQuerySet(this.querySets[slot]!, 0, 2, this.queryResolve[slot]!, 0);
    enc.copyBufferToBuffer(this.queryResolve[slot]!, 0, this.queryBufs[slot]!, 0, 16);
    device.queue.submit([enc.finish()]);
    this.readBack(device, slot, (ms) => {
      this.computeMs = this.computeMs * 0.85 + ms * 0.15;
    });
  }

  private readBack(device: GPUDevice, slot: number, onValue: (ms: number) => void): void {
    const readBuf = this.queryBufs[slot]!;
    this.queryBusy[slot] = true;
    readBuf
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const data = new BigUint64Array(readBuf.getMappedRange().slice(0));
        readBuf.unmap();
        const start = data[0] ?? 0n;
        const end = data[1] ?? 0n;
        if (end > start) onValue(Number(end - start) / 1e6);
      })
      .catch(() => {
        /* mapping can fail on device loss; timing is optional */
      })
      .finally(() => {
        this.queryBusy[slot] = false;
      });
    void device;
  }

  dispose(): void {
    this.particles.dispose();
    this.post.dispose();
    this.simUniform.destroy();
    for (const b of this.queryBufs) b.destroy();
    for (const b of this.queryResolve) b.destroy();
    this.queryBufs.length = 0;
    this.queryResolve.length = 0;
    this.querySets.length = 0;
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
