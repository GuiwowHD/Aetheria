/**
 * Aetheria — GPU particle store.
 *
 * Four parallel vec4 arrays per particle (position, velocity, colour, lifecycle)
 * in two ping-pong sets. Because WGSL forbids aliasing writable storage bindings,
 * the "copy" step is an explicit blit pipeline with its own bind-group pair; a
 * simulation step followed by a blit therefore always leaves *both* sets
 * consistent, and the render pass can always read the ping set without tracking
 * which half of the ping-pong just ran.
 *
 * Memory: 88 bytes per simulated particle (8 x vec4 x 2 sets + headroom), which
 * is what allows 4M simulated particles inside a ~350 MB budget while still
 * *displaying* up to 10M via render multiplicity.
 */

import type { GpuContext } from './device';
import { compileChecked } from './device';
import {
  SIMULATE_WGSL,
  SEED_WGSL,
  COPY_WGSL,
  MAX_SHOCKS,
  PARTICLE_WORKGROUP,
  COPY_ENTRY,
  SEED_ENTRY,
  SIM_ENTRY,
} from './wgsl/simulate.wgsl';
import { PARTICLE_VERTEX_WGSL, PARTICLE_FRAGMENT_WGSL } from './wgsl/particle.wgsl';
import { SIM_UNIFORM_BYTES, PARTICLE_BYTES_PER_SIM } from './bridge';

export interface ParticleSystemOptions {
  capacity: number;
  maxParticles: number;
}

/** A supernova shell, owned and aged by the renderer. */
export interface ShockEvent {
  origin: [number, number, number];
  age: number;
  strength: number;
}

const SHOCK_SLOTS = MAX_SHOCKS;
const SHOCK_BYTES = SHOCK_SLOTS * 2 * 16;

export class ParticleSystem {
  /** Bind group layout for the simulate/seed/copy kernels. */
  readonly simulateLayout: GPUBindGroupLayout;
  /** Bind group layout for the read-only render path. */
  readonly renderLayout: GPUBindGroupLayout;

  count = 0;
  capacity = 0;
  activeShocks = 0;

  private readonly device: GPUDevice;
  private readonly uniformBuffer: GPUBuffer;
  private readonly maxParticles: number;

  private readonly simulatePipeline: GPUComputePipeline;
  private readonly seedPipeline: GPUComputePipeline;
  private readonly copyPipeline: GPUComputePipeline;

  private bufP: GPUBuffer[] = [];
  private bufV: GPUBuffer[] = [];
  private bufC: GPUBuffer[] = [];
  private bufM: GPUBuffer[] = [];
  private readonly shockBuf: GPUBuffer;

  private simGroups: GPUBindGroup[] = [];
  private copyGroups: GPUBindGroup[] = [];
  private renderGroups: GPUBindGroup[] = [];

  private readonly pipelineLayoutSim: GPUPipelineLayout;

  private readonly shockOrigins = new Float32Array(SHOCK_SLOTS * 4);
  private readonly shockStrength = new Float32Array(SHOCK_SLOTS);
  private readonly shockUpload = new Float32Array(SHOCK_SLOTS * 2 * 4);

  private constructor(
    ctx: GpuContext,
    uniformBuffer: GPUBuffer,
    opts: ParticleSystemOptions,
    simModule: GPUShaderModule,
    seedModule: GPUShaderModule,
    copyModule: GPUShaderModule,
    vsModule: GPUShaderModule,
    fsModule: GPUShaderModule
  ) {
    this.device = ctx.device;
    this.uniformBuffer = uniformBuffer;
    this.maxParticles = opts.maxParticles;
    const device = this.device;
    void vsModule;
    void fsModule;

    // The simulation layout uses 8 storage buffers + shock + uniform; the render
    // layout needs only the four live arrays.
    const storageRW = (binding: number): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: 'storage' as GPUBufferBindingType },
    });
    const storageRO = (binding: number): GPUBindGroupLayoutEntry => ({
      binding,
      visibility: GPUShaderStage.VERTEX,
      buffer: { type: 'read-only-storage' as GPUBufferBindingType },
    });

    this.simulateLayout = device.createBindGroupLayout({
      label: 'simulate-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ...Array.from({ length: 8 }, (_, i) => storageRW(1 + i)),
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });

    this.renderLayout = device.createBindGroupLayout({
      label: 'particle-render-layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        storageRO(1),
        storageRO(2),
        storageRO(3),
        storageRO(4),
        storageRO(9),
      ],
    });

    this.pipelineLayoutSim = device.createPipelineLayout({
      label: 'sim-layout',
      bindGroupLayouts: [this.simulateLayout],
    });

    this.simulatePipeline = device.createComputePipeline({
      label: 'simulate',
      layout: this.pipelineLayoutSim,
      compute: { module: simModule, entryPoint: SIM_ENTRY },
    });
    this.seedPipeline = device.createComputePipeline({
      label: 'seed',
      layout: this.pipelineLayoutSim,
      compute: { module: seedModule, entryPoint: SEED_ENTRY },
    });
    this.copyPipeline = device.createComputePipeline({
      label: 'copy',
      layout: this.pipelineLayoutSim,
      compute: { module: copyModule, entryPoint: COPY_ENTRY },
    });

    this.shockBuf = device.createBuffer({
      label: 'shocks',
      size: SHOCK_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.shockBuf, 0, this.shockUpload);

    this.allocate(opts.capacity);
  }

  static async create(
    ctx: GpuContext,
    uniformBuffer: GPUBuffer,
    opts: ParticleSystemOptions,
    onError: (msg: string) => void
  ): Promise<ParticleSystem> {
    const device = ctx.device;
    const sim = await compileChecked(device, SIMULATE_WGSL, 'simulate.wgsl');
    const seedMod = await compileChecked(device, SEED_WGSL, 'seed.wgsl');
    const copyMod = await compileChecked(device, COPY_WGSL, 'copy.wgsl');
    const vs = await compileChecked(device, PARTICLE_VERTEX_WGSL, 'particle.vert.wgsl');
    const fs = await compileChecked(device, PARTICLE_FRAGMENT_WGSL, 'particle.frag.wgsl');
    const errs = [...sim.errors, ...seedMod.errors, ...copyMod.errors, ...vs.errors, ...fs.errors];
    if (errs.length) onError(errs.join('\n'));
    return new ParticleSystem(
      ctx,
      uniformBuffer,
      opts,
      sim.module,
      seedMod.module,
      copyMod.module,
      vs.module,
      fs.module
    );
  }

  /** (Re)create all particle buffers at `count` particles and re-seed them. */
  allocate(count: number): void {
    const device = this.device;
    const n = Math.max(64, Math.min(Math.floor(count), this.maxParticles));
    if (n === this.capacity && this.bufP.length === 2) {
      this.count = n;
      return;
    }
    for (const b of [...this.bufP, ...this.bufV, ...this.bufC, ...this.bufM]) b.destroy();
    this.bufP = [];
    this.bufV = [];
    this.bufC = [];
    this.bufM = [];

    this.capacity = n;
    this.count = n;
    const bytes = n * 16;
    const usage =
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX;
    for (let i = 0; i < 2; i++) {
      this.bufP.push(device.createBuffer({ label: `pos${i}`, size: bytes, usage }));
      this.bufV.push(device.createBuffer({ label: `vel${i}`, size: bytes, usage }));
      this.bufC.push(device.createBuffer({ label: `col${i}`, size: bytes, usage }));
      this.bufM.push(device.createBuffer({ label: `meta${i}`, size: bytes, usage }));
    }

    this.buildBindGroups();
    this.seed();
  }

  private buildBindGroups(): void {
    const device = this.device;
    const u = this.uniformBuffer;
    const sim = (
      label: string,
      read: [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer],
      write: [GPUBuffer, GPUBuffer, GPUBuffer, GPUBuffer]
    ) =>
      device.createBindGroup({
        label,
        layout: this.simulateLayout,
        entries: [
          { binding: 0, resource: { buffer: u } },
          { binding: 1, resource: { buffer: read[0] } },
          { binding: 2, resource: { buffer: read[1] } },
          { binding: 3, resource: { buffer: read[2] } },
          { binding: 4, resource: { buffer: read[3] } },
          { binding: 5, resource: { buffer: write[0] } },
          { binding: 6, resource: { buffer: write[1] } },
          { binding: 7, resource: { buffer: write[2] } },
          { binding: 8, resource: { buffer: write[3] } },
          { binding: 9, resource: { buffer: this.shockBuf } },
        ],
      });

    // The render path always reads the ping set, so the ping set is also the
    // simulation *read* source and the pong set is always the write target.
    this.simGroups = [
      sim('sim-ping->pong', [this.bufP[0]!, this.bufV[0]!, this.bufC[0]!, this.bufM[0]!], [
        this.bufP[1]!,
        this.bufV[1]!,
        this.bufC[1]!,
        this.bufM[1]!,
      ]),
    ];
    this.copyGroups = [
      sim('copy-pong->ping', [this.bufP[1]!, this.bufV[1]!, this.bufC[1]!, this.bufM[1]!], [
        this.bufP[0]!,
        this.bufV[0]!,
        this.bufC[0]!,
        this.bufM[0]!,
      ]),
    ];
    this.renderGroups = [
      device.createBindGroup({
        label: 'render-ping',
        layout: this.renderLayout,
        entries: [
          { binding: 0, resource: { buffer: u } },
          { binding: 1, resource: { buffer: this.bufP[0]! } },
          { binding: 2, resource: { buffer: this.bufV[0]! } },
          { binding: 3, resource: { buffer: this.bufC[0]! } },
          { binding: 4, resource: { buffer: this.bufM[0]! } },
          { binding: 9, resource: { buffer: this.shockBuf } },
        ],
      }),
    ];
  }

  get renderBindGroup(): GPUBindGroup {
    return this.renderGroups[0]!;
  }

  /** Dispatch the seed kernel into the pong set, then blit to the ping set. */
  seed(): void {
    const device = this.device;
    const groups = Math.ceil(this.count / PARTICLE_WORKGROUP);
    const encoder = device.createCommandEncoder({ label: 'seed' });
    const pass = encoder.beginComputePass({ label: 'seed-pass' });
    pass.setPipeline(this.seedPipeline);
    pass.setBindGroup(0, this.simGroups[0]!);
    pass.dispatchWorkgroups(groups);
    pass.setPipeline(this.copyPipeline);
    pass.setBindGroup(0, this.copyGroups[0]!);
    pass.dispatchWorkgroups(groups);
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  setSimCount(count: number): void {
    const n = Math.max(64, Math.min(Math.floor(count), this.maxParticles));
    if (n === this.count) return;
    if (n > this.capacity) this.allocate(n);
    else this.count = n;
  }

  /**
   * Upload the shockwave ring buffer. The caller owns ageing and pruning; this
   * only packs two vec4 arrays (origins+ages, then strengths) in one transfer.
   */
  tickShocks(shocks: readonly ShockEvent[]): void {
    this.shockOrigins.fill(0);
    this.shockStrength.fill(0);
    let active = 0;
    for (let i = 0; i < SHOCK_SLOTS; i++) {
      const ev = shocks[i];
      if (!ev || ev.age < 0 || ev.strength <= 0) continue;
      this.shockOrigins[i * 4 + 0] = ev.origin[0];
      this.shockOrigins[i * 4 + 1] = ev.origin[1];
      this.shockOrigins[i * 4 + 2] = ev.origin[2];
      this.shockOrigins[i * 4 + 3] = ev.age;
      this.shockStrength[i] = ev.strength;
      active++;
    }
    this.activeShocks = active;
    this.shockUpload.set(this.shockOrigins, 0);
    this.shockUpload.set(this.shockStrength, SHOCK_SLOTS * 4);
    this.device.queue.writeBuffer(this.shockBuf, 0, this.shockUpload);
  }

  /** One simulation step plus the parity blit, recorded into `encoder`. */
  step(encoder: GPUCommandEncoder, timing?: GPUComputePassTimestampWrites): void {
    const groups = Math.ceil(this.count / PARTICLE_WORKGROUP);
    const pass = timing
      ? encoder.beginComputePass({ label: 'simulate-step', timestampWrites: timing })
      : encoder.beginComputePass({ label: 'simulate-step' });
    pass.setPipeline(this.simulatePipeline);
    pass.setBindGroup(0, this.simGroups[0]!);
    pass.dispatchWorkgroups(groups);

    // Restore parity: pong -> ping. Two dispatches of a 1M-particle kernel cost
    // roughly 0.25 ms on an RTX-class part and buy a completely stateless render
    // path, which is what keeps this renderer correct under live quality changes.
    pass.setPipeline(this.copyPipeline);
    pass.setBindGroup(0, this.copyGroups[0]!);
    pass.dispatchWorkgroups(groups);
    pass.end();
  }

  estimateBytes(): number {
    return this.capacity * PARTICLE_BYTES_PER_SIM + SHOCK_BYTES + SIM_UNIFORM_BYTES;
  }

  dispose(): void {
    for (const b of [...this.bufP, ...this.bufV, ...this.bufC, ...this.bufM]) b.destroy();
    this.bufP = [];
    this.bufV = [];
    this.bufC = [];
    this.bufM = [];
    this.shockBuf.destroy();
    this.simGroups = [];
    this.copyGroups = [];
    this.renderGroups = [];
  }
}
