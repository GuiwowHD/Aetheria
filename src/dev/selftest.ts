/**
 * Aetheria — in-app self test.
 *
 * Headless Chrome presents a WebGPU canvas to a compositor surface that
 * `page.screenshot()` cannot capture, so verifying the render output from the
 * outside is unreliable. This module instead reads the pixels straight out of
 * the swap-chain texture with `copyTextureToBuffer`, on the GPU, before the
 * frame ends — which measures exactly what the renderer produced.
 *
 * It is opt-in (`?selftest=1`) and costs nothing otherwise: no extra buffers, no
 * readbacks, no allocation on the normal path.
 */

export interface SelfTestResult {  width: number;
  height: number;
  /** Base64 of the PNG-encoded frame, produced by the page's own encoder. */
  png: string;
  /** Mean linear-ish luma over the sampled grid, on the sRGB-encoded output. */
  mean: number;
  /** Peak luma, the best single indicator that HDR highlights survived. */
  max: number;
  /** Fraction of sampled pixels above 6/255, i.e. "not the void". */
  litRatio: number;
  /** Fraction above 0.95, i.e. blown out. A healthy frame is a few percent. */
  clippedRatio: number;
  /** Mean per channel, to catch a colour cast. */
  rgb: [number, number, number];
  /** Decoded pixel count actually sampled. */
  samples: number;
  /**
   * Luma statistics over a 40x18 tile grid. A single mean cannot distinguish a
   * nebula from a grey wash; the spread across tiles is what proves there is
   * structure with a dark sky around it, so it is computed here rather than by
   * re-decoding the PNG in the runner (which would need the page still alive).
   */
  grid: { mean: number; sd: number; p50: number; p95: number; max: number; min: number };
  /** `gl.getError()` after a WebGL2 readback; undefined on the WebGPU path. */
  glError?: number;
}

/**
 * Luma statistics over a 40x18 tile grid of the RGBA image. Separated from the
 * capture loop so the pixel scan and the structural summary stay readable.
 */
function tileStats(rgba: Uint8ClampedArray, width: number, height: number): SelfTestResult['grid'] {
  const GW = 40;
  const GH = 18;
  const tiles: number[] = [];
  for (let gy = 0; gy < GH; gy++) {
    for (let gx = 0; gx < GW; gx++) {
      const x0 = Math.floor((gx * width) / GW);
      const x1 = Math.max(x0 + 1, Math.floor(((gx + 1) * width) / GW));
      const y0 = Math.floor((gy * height) / GH);
      const y1 = Math.max(y0 + 1, Math.floor(((gy + 1) * height) / GH));
      let sum = 0;
      let count = 0;
      for (let y = y0; y < y1; y += 2) {
        for (let x = x0; x < x1; x += 2) {
          const i = (y * width + x) * 4;
          sum += 0.2126 * rgba[i]! + 0.7152 * rgba[i + 1]! + 0.0722 * rgba[i + 2]!;
          count++;
        }
      }
      tiles.push(sum / Math.max(1, count));
    }
  }
  const mean = tiles.reduce((a, b) => a + b, 0) / tiles.length;
  const sd = Math.sqrt(tiles.reduce((a, b) => a + (b - mean) * (b - mean), 0) / tiles.length);
  const sorted = [...tiles].sort((a, b) => a - b);
  return {
    mean,
    sd,
    min: sorted[0] ?? 0,
    p50: sorted[Math.floor(sorted.length / 2)] ?? 0,
    p95: sorted[Math.floor(sorted.length * 0.95)] ?? 0,
    max: sorted[sorted.length - 1] ?? 0,
  };
}

export class SelfTest {
  private pending = false;
  private inFlight = false;
  private callback: ((result: SelfTestResult) => void) | null = null;

  /**
   * Backend-agnostic capture entry point.
   *
   * WebGPU must copy the swap-chain texture inside its own command stream, so the
   * renderer calls `consume()` after submitting a frame. WebGL2's drawing buffer
   * is directly readable, so there is nothing to record and `attachGl()` can pull
   * the pixels as soon as the frame is drawn. Both funnel into `finish()`.
   */
  request(): Promise<SelfTestResult> {
    return new Promise((resolve) => {
      this.callback = resolve;
      this.pending = true;
    });
  }

  get armed(): boolean {
    return this.pending;
  }

  /**
   * WebGPU path: called by the renderer *after* the frame has been submitted, to
   * record the copy into its own command buffer. Copying inside the frame's own
   * encoder looks equivalent but is not: reading the mapped range then yields
   * zero bytes, because the mappable buffer is still being written.
   */
  consume(device: GPUDevice, canvas: GPUTexture): boolean {
    if (!this.pending || this.inFlight || canvas.width === 0) return false;
    this.pending = false;
    this.inFlight = true;

    const { width, height } = canvas;
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    const readBuf = device.createBuffer({
      label: 'selftest-readback',
      size: bytesPerRow * height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder({ label: 'selftest-copy' });
    encoder.copyTextureToBuffer({ texture: canvas }, { buffer: readBuf, bytesPerRow }, { width, height });
    device.queue.submit([encoder.finish()]);

    readBuf
      .mapAsync(GPUMapMode.READ)
      .then(() => {
        const range = readBuf.getMappedRange();
        const src = new Uint8Array(range);
        if (src.length === 0) throw new Error('mapped range was empty');
        // The swap chain is bgra8unorm, so the first byte is blue.
        this.finish(src, width, height, bytesPerRow, 'bgra');
        readBuf.unmap();
        readBuf.destroy();
      })
      .catch((err) => this.fail(err))
      .finally(() => {
        this.inFlight = false;
      });
    return true;
  }

  /**
   * WebGL2 path: read the default framebuffer directly into `src`, top-down, in
   * RGBA order. `gl.readPixels` returns rows bottom-up, so `flipY` is set.
   */
  attachGl(gl: WebGL2RenderingContext, width: number, height: number): boolean {
    if (!this.pending || this.inFlight || width === 0) return false;
    this.pending = false;
    this.inFlight = true;
    try {
      const src = new Uint8Array(width * height * 4);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, src);
      this.finish(src, width, height, width * 4, 'rgba', gl.getError());
    } catch (err) {
      this.fail(err);
    } finally {
      this.inFlight = false;
    }
    return true;
  }

  /**
   * Shared statistics + PNG encoding.
   *
   * `channelOrder` says which byte holds red. WebGPU's `bgra8unorm` swap chain
   * puts blue first; `gl.readPixels(..., gl.RGBA, ...)` puts red first. Getting
   * this wrong is not a subtle error — it produced a uniformly white frame during
   * development — so the two cases are named rather than passed as a boolean.
   */
  private finish(
    src: Uint8Array,
    width: number,
    height: number,
    bytesPerRow: number,
    channelOrder: 'rgba' | 'bgra',
    glError?: number
  ): void {
    const blueFirst = channelOrder === 'bgra';
    const image = new ImageData(width, height);
    const out = image.data;
    let sum = 0;
    let max = 0;
    let lit = 0;
    let clipped = 0;
    let sr = 0;
    let sg = 0;
    let sb = 0;
    let n = 0;
    const step = Math.max(1, Math.floor(width / 640));
    for (let y = 0; y < height; y += step) {
      const row = y * bytesPerRow;
      for (let x = 0; x < width; x += step) {
        const i0 = row + x * 4;
        const c0 = src[i0] ?? 0;
        const c1 = src[i0 + 1] ?? 0;
        const c2 = src[i0 + 2] ?? 0;
        const r = blueFirst ? c2 : c0;
        const g = c1;
        const b = blueFirst ? c0 : c2;
        const i = (y * width + x) * 4;
        out[i] = r;
        out[i + 1] = g;
        out[i + 2] = b;
        out[i + 3] = 255;
        const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        sum += l;
        if (l > max) max = l;
        if (l > 6) lit++;
        if (l > 242) clipped++;
        sr += r;
        sg += g;
        sb += b;
        n++;
      }
    }
    if (n === 0) {
      this.fail(new Error('readback produced no samples'));
      return;
    }

    let png = '';
    try {
      const off = document.createElement('canvas');
      off.width = width;
      off.height = height;
      const octx = off.getContext('2d');
      if (octx) {
        octx.putImageData(image, 0, 0);
        png = off.toDataURL('image/png').split(',')[1] ?? '';
      }
    } catch (err) {
      console.warn('[aetheria] PNG encoding unavailable:', err);
    }

    const result: SelfTestResult = {
      width,
      height,
      png,
      mean: sum / n,
      max,
      litRatio: lit / n,
      clippedRatio: clipped / n,
      rgb: [sr / n, sg / n, sb / n],
      samples: n,
      grid: tileStats(out, width, height),
      glError,
    };
    if (!Number.isFinite(result.mean)) {
      this.fail(new Error('readback produced a non-finite mean'));
      return;
    }
    const cb = this.callback;
    this.callback = null;
    cb?.(result);
  }

  private fail(err: unknown): void {
    // Surface the reason: a silent null here once hid a bad typed array and then
    // an empty mapped range, for several debug cycles.
    console.error('[aetheria] self-test capture failed:', err);
    this.callback = null;
  }
}
