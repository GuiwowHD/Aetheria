/**
 * Aetheria — HDR post-processing chain.
 *
 * Resource graph (everything linear HDR until the final composite):
 *
 *   scene / previous   rgba16f full   this frame's accumulation and last frame's
 *   scratch            rgba16f full   LDR composite result, read by FXAA
 *   depth              depth32f full  written by the particle pass, read by DOF
 *   bloomDown[0..N-1]  rgba16f        bright pass, then successive downsamples
 *   bloomUp[0..N]      rgba16f        upsample accumulation, ending at level 0
 *   godray             rgba16f full   radial volumetric accumulation
 *   dof                rgba16f half   tile colour (rgb) + max CoC (a)
 *
 * Per-frame pass order:
 *   fadeScene -> [particles into scene] -> bright/6x downsample -> 6x upsample
 *   -> godray -> dofDown -> dofGather -> composite -> fxaa -> canvas
 *
 * WebGPU forbids a pass from sampling a texture it is also rendering into. That
 * single rule dictates the shape of everything below:
 *   - the fade reads `previous` and writes `scene`, which the particles then
 *     load and add to (reading then writing within one pass is legal, unlike
 *     sampling a texture while writing it);
 *   - the two scene textures swap roles after the frame, so no copy is needed;
 *   - the bloom pyramid uses two parallel chains, `bloomDown` for the descent
 *     and `bloomUp` for the ascent, so no level is ever both source and target;
 *   - god rays accumulate into their own target rather than into the bloom mip
 *     the composite samples;
 *   - DOF gathers into `scratch`, leaving `scene` purely a source.
 */

import type { GpuContext } from './device';
import { compileChecked } from './device';
import { PostWriter } from '../core/uniforms';
import {
  FULLSCREEN_VS,
  BRIGHT_PASS_WGSL,
  BLOOM_DOWN_WGSL,
  BLOOM_UP_WGSL,
  GODRAY_WGSL,
  DOF_DOWN_WGSL,
  DOF_GATHER_WGSL,
  COMPOSITE_WGSL,
  FXAA_WGSL,
  FADE_WGSL,
} from './wgsl/post.wgsl';

export interface PostOptions {
  bloomLevels: number;
  dof: boolean;
  volumetric: boolean;
}

interface Pass {
  pipeline: GPURenderPipeline;
}

interface PlannedPass {
  label: string;
  pass: Pass;
  bind: GPUBindGroup;
  target: GPUTextureView;
  slot: number;
  load: 'clear' | 'load';
}

const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
/** Bytes of one Post block, matching the WGSL struct. */
const POST_UNIFORM_BYTES = 128;
/** Dynamic-offset stride: a dynamic offset must be 256-byte aligned. */
const SLOT_STRIDE = 256;
/** Uniform slots, one per pass instance within a frame. */
const MAX_DOWN = 8;
const MAX_UP = 8;

export class PostChain {
  private readonly ctx: GpuContext;
  private readonly uniformBuffer: GPUBuffer;
  private readonly writer = new PostWriter();

  private vsModule!: GPUShaderModule;

  private sceneTex: GPUTexture | null = null;
  private previousTex: GPUTexture | null = null;
  /** Tile-averaged colour + max CoC, half resolution. */
  private dofLowTex: GPUTexture | null = null;
  /**
   * LDR composite target. Separate from scratchTex because the composite reads
   * the gathered DOF result (which lives in scratch) and a pass may not sample
   * the texture it renders into.
   */
  private litTex: GPUTexture | null = null;
  private depthTex: GPUTexture | null = null;
  private bloomDown: GPUTexture[] = [];
  private bloomUp: GPUTexture[] = [];
  private godrayTex: GPUTexture | null = null;
  private dofTex: GPUTexture | null = null;

  private width = 0;
  private height = 0;
  private levels = 6;
  private hasDof = true;
  private hasVolumetric = true;

  private samplerLinear!: GPUSampler;
  private samplerDepth!: GPUSampler;

  private layoutA!: GPUBindGroupLayout;
  private layoutAB!: GPUBindGroupLayout;
  private layoutComposite!: GPUBindGroupLayout;
  private layoutDof!: GPUBindGroupLayout;

  private pFade!: Pass;
  private pBright!: Pass;
  private pDown!: Pass;
  private pUp!: Pass;
  private pGod!: Pass;
  private pDofDown!: Pass;
  private pDofGather!: Pass;
  private pComposite!: Pass;
  private pFxaa!: Pass;

  private constructor(ctx: GpuContext) {
    this.ctx = ctx;
    // Slot map: 0 fade, 1 bright, 2..9 downsample, 10..17 upsample, 18 godray,
    // 19 dofDown, 20 dofGather, 21 composite, 22 fxaa. The +5 covers the five
    // tail slots; an undersized uniform buffer fails at write time rather than
    // at creation, so this arithmetic has to be right.
    const slots = 2 + MAX_DOWN + MAX_UP + 5;
    this.uniformBuffer = ctx.device.createBuffer({
      label: 'post-uniform',
      size: slots * SLOT_STRIDE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  static async create(
    ctx: GpuContext,
    opts: PostOptions,
    onError: (msg: string) => void
  ): Promise<PostChain> {
    const device = ctx.device;
    const vs = await compileChecked(device, FULLSCREEN_VS, 'fullscreen.vert.wgsl');
    const keys = [
      'bright', 'bloomDown', 'bloomUp', 'godray', 'dofDown', 'dofGather', 'composite', 'fxaa', 'fade',
    ] as const;
    const srcs = [
      BRIGHT_PASS_WGSL, BLOOM_DOWN_WGSL, BLOOM_UP_WGSL, GODRAY_WGSL,
      DOF_DOWN_WGSL, DOF_GATHER_WGSL, COMPOSITE_WGSL, FXAA_WGSL, FADE_WGSL,
    ] as const;
    const compiled = await Promise.all(srcs.map((s, i) => compileChecked(device, s, `${keys[i]}.wgsl`)));
    const errors = [...vs.errors, ...compiled.flatMap((c) => c.errors)];
    if (errors.length) onError(errors.join('\n'));

    const chain = new PostChain(ctx);
    chain.vsModule = vs.module;
    chain.levels = Math.max(2, Math.min(opts.bloomLevels, MAX_DOWN));
    chain.hasDof = opts.dof;
    chain.hasVolumetric = opts.volumetric;
    chain.initSamplers();
    chain.initLayouts();
    chain.buildPipelines(
      Object.fromEntries(keys.map((k, i) => [k, compiled[i]!.module])) as Record<(typeof keys)[number], GPUShaderModule>
    );
    return chain;
  }

  // -------------------------------------------------------------------------
  // Layouts + pipelines
  // -------------------------------------------------------------------------
  private initSamplers(): void {
    const device = this.ctx.device;
    this.samplerLinear = device.createSampler({
      label: 'post-linear',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
    // A depth texture must be paired with a comparison sampler in WGSL; the
    // depth value itself is read with textureLoad (see the DOF shader).
    this.samplerDepth = device.createSampler({
      label: 'post-depth',
      compare: 'less',
      magFilter: 'nearest',
      minFilter: 'nearest',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
  }

  private initLayouts(): void {
    const device = this.ctx.device;
    const tex = (binding: number, sampleType: GPUTextureSampleType = 'float'): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.FRAGMENT,
      texture: { sampleType, viewDimension: '2d' },
    });
    const uni: GPUBindGroupLayoutEntry = {
      binding: 4,
      visibility: GPUShaderStage.FRAGMENT,
      buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: POST_UNIFORM_BYTES },
    };
    const samp: GPUBindGroupLayoutEntry = {
      binding: 1,
      visibility: GPUShaderStage.FRAGMENT,
      sampler: { type: 'filtering' },
    };
    const sampCompare: GPUBindGroupLayoutEntry = {
      binding: 2,
      visibility: GPUShaderStage.FRAGMENT,
      sampler: { type: 'comparison' },
    };
    const depthTex: GPUBindGroupLayoutEntry = {
      binding: 3,
      visibility: GPUShaderStage.FRAGMENT,
      texture: { sampleType: 'depth', viewDimension: '2d' },
    };

    // A=[texA,samp,U]  AB=[texA,samp,texB,U]  ABC=[texA,samp,texB,texC,U]
    // DOF=[texA,samp,compareSamp,depth,U]
    this.layoutA = device.createBindGroupLayout({ label: 'post-a', entries: [tex(0), samp, uni] });
    this.layoutAB = device.createBindGroupLayout({ label: 'post-ab', entries: [tex(0), samp, tex(2), uni] });
    this.layoutComposite = device.createBindGroupLayout({
      label: 'post-composite',
      entries: [tex(0), samp, tex(2), tex(3), tex(5), uni],
    });
    this.layoutDof = device.createBindGroupLayout({
      label: 'post-dof',
      entries: [tex(0), samp, sampCompare, depthTex, uni],
    });
  }

  private buildPipelines(m: Record<string, GPUShaderModule>): void {
    const device = this.ctx.device;
    const opaque: GPUBlendState = {
      color: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'zero', operation: 'add' },
    };
    const rp = (
      label: string,
      module: GPUShaderModule,
      entry: string,
      layout: GPUBindGroupLayout,
      format: GPUTextureFormat
    ): Pass => ({
      pipeline: device.createRenderPipeline({
        label,
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module: this.vsModule, entryPoint: 'vs_fullscreen' },
        fragment: { module, entryPoint: entry, targets: [{ format, blend: opaque }] },
        primitive: { topology: 'triangle-list' },
      }),
    });

    this.pFade = rp('fade', m.fade, 'fs_fade', this.layoutA, HDR_FORMAT);
    this.pBright = rp('bright', m.bright, 'fs_bright', this.layoutA, HDR_FORMAT);
    this.pDown = rp('bloom-down', m.bloomDown, 'fs_down', this.layoutA, HDR_FORMAT);
    this.pUp = rp('bloom-up', m.bloomUp, 'fs_up', this.layoutAB, HDR_FORMAT);
    this.pGod = rp('godray', m.godray, 'fs_godray', this.layoutAB, HDR_FORMAT);
    this.pDofDown = rp('dof-down', m.dofDown, 'fs_dof_down', this.layoutDof, HDR_FORMAT);
    this.pDofGather = rp('dof-gather', m.dofGather, 'fs_dof', this.layoutAB, HDR_FORMAT);
    this.pComposite = rp('composite', m.composite, 'fs_composite', this.layoutComposite, HDR_FORMAT);
    this.pFxaa = rp('fxaa', m.fxaa, 'fs_fxaa', this.layoutA, this.ctx.format);
  }

  // -------------------------------------------------------------------------
  // Targets
  // -------------------------------------------------------------------------
  resize(width: number, height: number): void {
    const w = Math.max(8, Math.floor(width));
    const h = Math.max(8, Math.floor(height));
    if (w === this.width && h === this.height && this.sceneTex) return;
    this.width = w;
    this.height = h;
    this.destroyTargets();

    const device = this.ctx.device;
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const mk = (label: string, tw: number, th: number) =>
      device.createTexture({
        label,
        size: { width: Math.max(1, tw), height: Math.max(1, th) },
        format: HDR_FORMAT,
        usage,
      });

    this.sceneTex = mk('scene', w, h);
    this.previousTex = mk('previous', w, h);
    this.dofLowTex = mk('dofLow', w >> 1, h >> 1);
    this.litTex = mk('lit', w, h);
    this.depthTex = device.createTexture({
      label: 'depth',
      size: { width: w, height: h },
      format: 'depth32float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });

    // Two parallel bloom chains so the descent and the ascent never touch the
    // same texture twice in one pass. Level 0 is half resolution.
    this.bloomDown = [];
    this.bloomUp = [];
    for (let i = 0; i < this.levels; i++) {
      const sc = 1 << (i + 1);
      const bw = Math.max(2, Math.floor(w / sc));
      const bh = Math.max(2, Math.floor(h / sc));
      if (bw < 4 || bh < 4) break;
      this.bloomDown.push(mk(`bloomDown${i}`, bw, bh));
      this.bloomUp.push(mk(`bloomUp${i}`, bw, bh));
    }
    if (this.bloomDown.length < 2) {
      this.bloomDown = [mk('bloomDown0', w >> 1, h >> 1), mk('bloomDown1', w >> 2, h >> 2)];
      this.bloomUp = [mk('bloomUp0', w >> 1, h >> 1), mk('bloomUp1', w >> 2, h >> 2)];
    }
    this.levels = this.bloomDown.length;

    this.godrayTex = mk('godray', w, h);
    this.dofTex = mk('dof', w, h);
  }

  private texelOf(t: GPUTexture): { x: number; y: number } {
    return { x: 1 / t.width, y: 1 / t.height };
  }

  // -------------------------------------------------------------------------
  // Views used by the renderer
  // -------------------------------------------------------------------------
  get sceneView(): GPUTextureView {
    return this.sceneTex!.createView();
  }

  get depthView(): GPUTextureView {
    return this.depthTex!.createView();
  }

  get bloomLevelCount(): number {
    return this.levels;
  }

  get dimensions(): { width: number; height: number } {
    return { width: this.width, height: this.height };
  }

  /**
   * Publish every per-pass uniform block, then swap the scene/previous roles.
   * Must run before the particle pass is recorded, because the swap is what
   * tells the particle pass which texture is this frame's accumulation buffer.
   */
  prepareFrame(
    state: Parameters<PostWriter['write']>[0],
    trails: number,
    focusDepth: number,
    aperture: number
  ): void {
    if (!this.sceneTex || !this.previousTex || !this.godrayTex) return;
    const device = this.ctx.device;
    const full = this.texelOf(this.sceneTex);

    const patch = (slot: number, t: { x: number; y: number }, extra?: Partial<typeof state>) => {
      device.queue.writeBuffer(
        this.uniformBuffer,
        slot * SLOT_STRIDE,
        this.writer.write({ ...state, ...extra, texelX: t.x, texelY: t.y })
      );
    };

    // slot 0: fade. `trails` rides in the chroma slot because the fade shader is
    // the only consumer of that field at this point in the frame.
    patch(0, full, { chroma: trails, bloom: 0, dof: 0, volumetric: 0, grain: 0, vignette: 0 });
    // slot 1: bright pass, reading the full-resolution scene.
    patch(1, full);
    // slots 2..2+N-1: downsample. Slot i reads level i-1 (or the full-resolution
    // scene at i == 0), so the kernel must step in the source's texels.
    for (let i = 0; i < this.levels; i++) {
      const src = i === 0 ? this.sceneTex : this.bloomDown[i - 1];
      patch(2 + i, src ? this.texelOf(src) : full);
    }
    // slots 10..: upsample. The kernel reads level i and blends into level i-1,
    // so the step is one texel of level i.
    for (let i = 0; i < this.levels; i++) {
      const src = this.bloomDown[i];
      patch(2 + MAX_DOWN + i, src ? this.texelOf(src) : full);
    }
    const godraySlot = 2 + MAX_DOWN + MAX_UP;
    patch(godraySlot, this.texelOf(this.godrayTex));
    patch(godraySlot + 1, full, { focusDepth, focusRange: 1, dof: aperture });
    patch(godraySlot + 2, full);
    patch(godraySlot + 3, full);
    patch(godraySlot + 4, full);
  }

  /** After this call, the scene just written becomes next frame's `previous`. */
  commitFrame(): void {
    const tmp = this.sceneTex;
    this.sceneTex = this.previousTex;
    this.previousTex = tmp;
  }

  // -------------------------------------------------------------------------
  // Frame execution
  // -------------------------------------------------------------------------
  /** Decay the previous frame into this frame's accumulation buffer. */
  fadeScene(encoder: GPUCommandEncoder): void {
    const bind = this.ctx.device.createBindGroup({
      label: 'fade',
      layout: this.layoutA,
      entries: [
        { binding: 0, resource: this.previousTex!.createView() },
        { binding: 1, resource: this.samplerLinear },
        { binding: 4, resource: { buffer: this.uniformBuffer, size: POST_UNIFORM_BYTES } },
      ],
    });
    this.runPass(encoder, 'fade-scene', this.pFade, bind, this.sceneView, 0, 'clear');
  }

  run(encoder: GPUCommandEncoder, canvas: GPUTextureView): void {
    const device = this.ctx.device;
    const view = (t: GPUTexture) => t.createView();
    const ub = () => ({ buffer: this.uniformBuffer, size: POST_UNIFORM_BYTES });
    const scene = view(this.sceneTex!);
    const level = this.levels;
    const godraySlot = 2 + MAX_DOWN + MAX_UP;

    // --- bright pass -> bloomDown[0] --------------------------------------
    this.runPass(
      encoder,
      'bright',
      this.pBright,
      device.createBindGroup({
        label: 'bright',
        layout: this.layoutA,
        entries: [
          { binding: 0, resource: scene },
          { binding: 1, resource: this.samplerLinear },
          { binding: 4, resource: ub() },
        ],
      }),
      view(this.bloomDown[0]!),
      1,
      'clear'
    );

    // --- descent: each downsample reads the level above and writes its own --
    for (let i = 1; i < level; i++) {
      this.runPass(
        encoder,
        `bloom-down-${i}`,
        this.pDown,
        device.createBindGroup({
          label: `bloom-down-${i}`,
          layout: this.layoutA,
          entries: [
            { binding: 0, resource: view(this.bloomDown[i - 1]!) },
            { binding: 1, resource: this.samplerLinear },
            { binding: 4, resource: ub() },
          ],
        }),
        view(this.bloomDown[i]!),
        2 + i,
        'clear'
      );
    }

    // --- ascent: finer level + coarser up-chain, into this level's up-chain --
    // No seed pass for the coarsest level is needed: it is never read, because
    // level i-1 samples level i and the composite reads only bloomUp[0]. Seeding
    // it would spend a full pass producing output nothing consumes.
    for (let i = level - 2; i > 0; i--) {
      this.runPass(
        encoder,
        `bloom-up-${i}`,
        this.pUp,
        device.createBindGroup({
          label: `bloom-up-${i}`,
          layout: this.layoutAB,
          entries: [
            { binding: 0, resource: view(this.bloomUp[i]!) },
            { binding: 1, resource: this.samplerLinear },
            { binding: 2, resource: view(this.bloomDown[i - 1]!) },
            { binding: 4, resource: ub() },
          ],
        }),
        view(this.bloomUp[i - 1]!),
        2 + MAX_DOWN + (i - 1),
        'clear'
      );
    }

    // --- volumetric light into its own target ------------------------------
    if (this.hasVolumetric) {
      this.runPass(
        encoder,
        'godray',
        this.pGod,
        device.createBindGroup({
          label: 'godray',
          layout: this.layoutAB,
          entries: [
            { binding: 0, resource: view(this.bloomUp[0]!) },
            { binding: 1, resource: this.samplerLinear },
            { binding: 2, resource: scene },
            { binding: 4, resource: ub() },
          ],
        }),
        view(this.godrayTex!),
        godraySlot,
        'clear'
      );
    }

    // --- depth of field: scene -> dofTex -> sceneScratch -------------------
    if (this.hasDof) {
      this.runPass(
        encoder,
        'dof-down',
        this.pDofDown,
        device.createBindGroup({
          label: 'dof-down',
          layout: this.layoutDof,
          entries: [
            { binding: 0, resource: scene },
            { binding: 1, resource: this.samplerLinear },
            { binding: 2, resource: this.samplerDepth },
            { binding: 3, resource: view(this.depthTex!) },
            { binding: 4, resource: ub() },
          ],
        }),
        view(this.dofLowTex!),
        godraySlot + 1,
        'clear'
      );
      this.runPass(
        encoder,
        'dof-gather',
        this.pDofGather,
        this.tag(
          device.createBindGroup({
            label: 'dof-gather',
            layout: this.layoutAB,
            entries: [
              { binding: 0, resource: scene },
              { binding: 1, resource: this.samplerLinear },
              { binding: 2, resource: view(this.dofLowTex!) },
              { binding: 4, resource: ub() },
            ],
          }),
          'scene',
          'dof'
        ),
        // Half resolution: the bokeh disc is wide and smooth, so the gather runs
        // over a quarter of the pixels and the composite upsamples it.
        view(this.dofTex!),
        godraySlot + 2,
        'clear'
      );
    }

    // --- composite: four distinct inputs -> lit ----------------------------
    // The scene stays a pure source here, which is what lets the particle pass
    // of the *next* frame accumulate into it after the fade.
    this.runPass(
      encoder,
      'composite',
      this.pComposite,
      this.tag(
        device.createBindGroup({
          label: 'composite',
          layout: this.layoutComposite,
          entries: [
            { binding: 0, resource: scene },
            { binding: 1, resource: this.samplerLinear },
            { binding: 2, resource: view(this.bloomUp[0]!) },
            { binding: 3, resource: this.hasVolumetric ? view(this.godrayTex!) : view(this.bloomUp[0]!) },
            { binding: 5, resource: this.hasDof ? view(this.dofTex!) : scene },
            { binding: 4, resource: ub() },
          ],
        }),
        'scene',
        'bloom',
        this.hasVolumetric ? 'godray' : 'bloom',
        this.hasDof ? 'scratch' : 'scene'
      ),
      view(this.litTex!),
      godraySlot + 3,
      'clear'
    );

    // --- FXAA + present ---------------------------------------------------
    // The LDR composite is written to lit, then FXAA reads lit and writes the
    // canvas, so neither pass aliases its own source.
    this.runPass(
      encoder,
      'fxaa',
      this.pFxaa,
      this.tag(
        device.createBindGroup({
          label: 'fxaa',
          layout: this.layoutA,
          entries: [
            { binding: 0, resource: view(this.litTex!) },
            { binding: 1, resource: this.samplerLinear },
            { binding: 4, resource: ub() },
          ],
        }),
        'lit'
      ),
      canvas,
      godraySlot + 4,
      'clear'
    );
    void level;
  }

  /** Record which textures a bind group exposes (debug aid). */
  private tag(bind: GPUBindGroup, ...names: string[]): GPUBindGroup {
    (bind as unknown as { __reads?: string[] }).__reads = names;
    return bind;
  }

  private runPass(
    encoder: GPUCommandEncoder,
    label: string,
    pass: Pass,
    bind: GPUBindGroup,
    target: GPUTextureView,
    slot: number,
    load: 'clear' | 'load'
  ): void {
    const rp = encoder.beginRenderPass({
      label,
      colorAttachments: [
        { view: target, loadOp: load, storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } },
      ],
    });
    rp.setPipeline(pass.pipeline);
    rp.setBindGroup(0, bind, [slot * SLOT_STRIDE]);
    rp.draw(3);
    rp.end();
    // Opt-in audit: `?wgsl=1` collects a pass/target/binding table so a WebGPU
    // usage conflict can be attributed to a specific pass instead of guessed at.
    const audit = (window as unknown as { __AETHERIA_AUDIT?: string[] }).__AETHERIA_AUDIT;
    if (audit) {
      const reads = (bind as unknown as { __reads?: string[] }).__reads ?? [];
      audit.push(`${label} | target=${target.label || '(none)'} | reads=${reads.join(',') || '(untagged)'}`);
    }
  }

  setQuality(dof: boolean, volumetric: boolean): void {
    this.hasDof = dof;
    this.hasVolumetric = volumetric;
  }

  private destroyTargets(): void {
    this.sceneTex?.destroy();
    this.previousTex?.destroy();
    this.dofLowTex?.destroy();
    this.litTex?.destroy();
    this.depthTex?.destroy();
    for (const t of this.bloomDown) t.destroy();
    for (const t of this.bloomUp) t.destroy();
    this.bloomDown = [];
    this.bloomUp = [];
    this.godrayTex?.destroy();
    this.dofTex?.destroy();
    this.sceneTex = null;
    this.previousTex = null;
    this.dofLowTex = null;
    this.litTex = null;
    this.depthTex = null;
    this.godrayTex = null;
    this.dofTex = null;
  }

  estimateBytes(): number {
    const px = this.width * this.height;
    if (!px) return 0;
    let bytes = px * 8 * 5; // scene, previous, scratch, lit, godray (rgba16f full-res)
    bytes += px * 4; // depth32float
    for (const t of this.bloomDown) bytes += t.width * t.height * 8;
    for (const t of this.bloomUp) bytes += t.width * t.height * 8;
    bytes += this.dofLowTex ? this.dofLowTex.width * this.dofLowTex.height * 8 : 0;
    bytes += this.dofTex ? this.dofTex.width * this.dofTex.height * 8 : 0;
    return bytes;
  }

  dispose(): void {
    this.destroyTargets();
    this.uniformBuffer.destroy();
  }
}

void (0 as unknown as PlannedPass);
