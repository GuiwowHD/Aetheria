/**
 * Aetheria — per-pass GPU timing.
 *
 * A single timestamp query set is shared by every pass in the frame, with two
 * indices reserved per pass. Each index pair is bracketed with
 * `writeTimestamp()`, and one `resolveQuerySet` at the end of the frame reads the
 * whole thing back, so a frame costs exactly one resolve and one small copy.
 *
 * This exists because attribution has to be measured. Optimising a pass because
 * it *looks* expensive is how a renderer ends up with a fast bloom chain and a
 * 40 ms particle pass; the numbers decide.
 */

/** Indices per pass: entry and exit. */
const STRIDE = 2;

export interface PassTiming {
  label: string;
  ms: number;
}

export class PassTimer {
  private readonly device: GPUDevice;
  private readonly set: GPUQuerySet;
  private readonly resolveBuf: GPUBuffer;
  private readonly readBuf: GPUBuffer;
  private readonly labels: string[] = [];
  /** Labels of the frame currently being resolved; immune to reset(). */
  private resolvedLabels: string[] = [];
  private inFlight = false;
  private pendingResolve = false;
  private writeSupported: boolean | undefined;

  /** Set true for one frame when the caller wants numbers for that frame. */
  captureRequested = false;

  constructor(device: GPUDevice, maxPasses: number) {
    this.device = device;
    this.set = device.createQuerySet({ label: 'pass-timings', type: 'timestamp', count: maxPasses * STRIDE });
    this.resolveBuf = device.createBuffer({
      label: 'pass-timings-resolve',
      size: maxPasses * STRIDE * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    // A buffer with MAP_READ may only also carry COPY_DST, so the query resolves
    // into a GPU-side buffer and is copied into this staging one.
    this.readBuf = device.createBuffer({
      label: 'pass-timings-readback',
      size: maxPasses * STRIDE * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  get available(): boolean {
    return !this.inFlight;
  }

  /** Forget the previous frame's labels; called at the start of each frame. */
  reset(): void {
    this.labels.length = 0;
  }

  /**
   * Bracket a pass with two timestamp writes. The write indices are passed
   * separately from the pass descriptor so the caller can attach
   * `beginningOfPassWriteIndex` to the pass while the closing write happens on
   * the encoder after the pass ends.
   *
   * `GPUCommandEncoder.writeTimestamp` is a newer addition and is absent from
   * some implementations, so it is probed once and reported as unavailable
   * rather than assumed.
   */
  private canWrite(encoder: GPUCommandEncoder): boolean {
    if (this.writeSupported === undefined) {
      this.writeSupported = typeof (encoder as { writeTimestamp?: unknown }).writeTimestamp === 'function';
      if (!this.writeSupported) {
        console.warn('[aetheria] GPUCommandEncoder.writeTimestamp is unavailable; per-pass profiling is disabled');
      }
    }
    return this.writeSupported;
  }

  private write(encoder: GPUCommandEncoder, index: number): void {
    (encoder as unknown as { writeTimestamp(set: GPUQuerySet, i: number): void }).writeTimestamp(this.set, index);
  }

  /**
   * Open a bracket for one pass. Returns the descriptor to attach to
   * `beginComputePass`/`beginRenderPass`, or undefined when profiling is off or
   * the previous readback is still outstanding.
   */
  bracket(encoder: GPUCommandEncoder, label: string, capture: boolean): GPUComputePassTimestampWrites | undefined {
    if (!capture || this.inFlight || !this.canWrite(encoder)) return undefined;
    const slot = this.labels.length;
    if ((slot + 1) * STRIDE > this.set.count) return undefined;
    this.labels.push(label);
    const base = slot * STRIDE;
    this.write(encoder, base);
    return { querySet: this.set, beginningOfPassWriteIndex: base };
  }

  /** Close the bracket opened by the matching `bracket()` call. */
  close(encoder: GPUCommandEncoder): void {
    const slot = this.labels.length - 1;
    if (slot < 0 || !this.canWrite(encoder)) return;
    this.write(encoder, slot * STRIDE + 1);
  }

  /** Resolve and copy; call once, after the frame's own submission. */
  finish(): void {
    if (!this.labels.length || this.inFlight) return;
    // Snapshot the labels now. `reset()` clears the live list at the start of the
    // next frame, while `collect()` may still be waiting on the map, so reading
    // the live list there would pair this frame's timestamps with nothing.
    this.resolvedLabels = this.labels.slice();
    const count = this.labels.length * STRIDE;
    const enc = this.device.createCommandEncoder({ label: 'pass-timings' });
    enc.resolveQuerySet(this.set, 0, count, this.resolveBuf, 0);
    enc.copyBufferToBuffer(this.resolveBuf, 0, this.readBuf, 0, count * 8);
    this.device.queue.submit([enc.finish()]);
    this.pendingResolve = true;
  }

  /**
   * Read the resolved timestamps and hand back a per-pass breakdown. Resolves to
   * an empty array until the GPU has finished the frame the numbers belong to.
   */
  async collect(): Promise<PassTiming[]> {
    if (!this.pendingResolve || this.inFlight) return [];
    this.pendingResolve = false;
    const labels = this.resolvedLabels;
    if (!labels.length) return [];
    this.inFlight = true;
    try {
      await this.readBuf.mapAsync(GPUMapMode.READ);
      const data = new BigUint64Array(this.readBuf.getMappedRange().slice(0));
      this.readBuf.unmap();
      const out: PassTiming[] = [];
      for (let i = 0; i < labels.length; i++) {
        const start = data[i * STRIDE] ?? 0n;
        const end = data[i * STRIDE + 1] ?? 0n;
        const ms = end > start ? Number(end - start) / 1e6 : 0;
        out.push({ label: labels[i]!, ms });
      }
      return out;
    } catch {
      return [];
    } finally {
      this.inFlight = false;
    }
  }

  dispose(): void {
    this.set.destroy();
    this.resolveBuf.destroy();
    this.readBuf.destroy();
  }
}
