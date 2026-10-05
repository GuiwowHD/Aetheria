/**
 * Aetheria — WebGPU device acquisition and diagnostics.
 *
 * Feature detection is deliberately pessimistic: every optional capability
 * (timestamp queries, float32 blending, max storage buffer size) is probed and
 * the renderer degrades instead of assuming. Shader compilation is *checked*
 * rather than trusted, so a WGSL error surfaces as a readable message in the
 * HUD instead of a black frame.
 */

import { BackendUnavailable } from '../core/types';

export interface GpuContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  format: GPUTextureFormat;
  context: GPUCanvasContext;
  canvas: HTMLCanvasElement;
  hasTimestamp: boolean;
  /** Max storage buffer binding size, used to validate the particle count. */
  maxStorageBufferBindingSize: number;
  maxBufferSize: number;
  maxComputeInvocations: number;
  limits: GPUSupportedLimits;
  features: Set<string>;
  info: { vendor: string; architecture: string; device: string; description: string };
}

export interface DeviceRequestOptions {
  onMessage?: (level: 'info' | 'warn' | 'error', text: string) => void;
}

/**
 * Limits we would *like*. Each is clamped to what the adapter reports, because
 * `requestDevice()` rejects a descriptor that asks for more than the
 * implementation offers — and silently falling back to default limits is what
 * would cap the universe at 1.5M particles.
 *
 * `maxStorageBufferBindingSize` is the one that decides the particle ceiling:
 * Chrome's default is 128 MB (about 1.49M particles at 88 B each), whereas the
 * hardware maximum on a discrete GPU is typically several GB.
 */
const DESIRED_LIMITS: Record<string, number> = {
  maxStorageBufferBindingSize: 1_500_000_000,
  maxBufferSize: 2_000_000_000,
  maxComputeWorkgroupsPerDimension: 65535,
  maxStorageBuffersPerShaderStage: 10,
  maxBindGroups: 4,
  maxTextureDimension2D: 8192,
};

export async function createGpuContext(
  canvas: HTMLCanvasElement,
  opts: DeviceRequestOptions = {}
): Promise<GpuContext> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new BackendUnavailable(
      'WebGPU is not available in this browser',
      'navigator.gpu is undefined. Chrome/Edge 113+ on desktop, or Safari 18+, expose it; otherwise the WebGL2 path is used.'
    );
  }

  // Request the highest limits we actually need, and retry with defaults if the
  // adapter refuses them (some integrated GPUs cap maxBufferSize lower).
  let adapter: GPUAdapter | null = null;
  try {
    adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
      forceFallbackAdapter: false,
    });
  } catch (err) {
    throw new BackendUnavailable('requestAdapter() failed', String(err));
  }
  if (!adapter) {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'low-power' });
  }
  if (!adapter) {
    adapter = await navigator.gpu.requestAdapter({ forceFallbackAdapter: true });
  }
  if (!adapter) {
    throw new BackendUnavailable(
      'No WebGPU adapter available',
      'The browser exposes WebGPU but no adapter could be created (GPU blocklisted, software rendering disabled, or driver reset).'
    );
  }

  const features = new Set<string>([...adapter.features]);
  const wantFeatures: GPUFeatureName[] = [];
  if (features.has('timestamp-query')) wantFeatures.push('timestamp-query');
  // Float32 blending is only needed if a driver ever refuses rgba16float; ask
  // for it opportunistically so the fallback is available.
  if (features.has('float32-blendable')) wantFeatures.push('float32-blendable');

  const desired: Record<string, number> = {};
  for (const [k, v] of Object.entries(DESIRED_LIMITS)) {
    const supported = (adapter.limits as unknown as Record<string, number>)[k];
    if (supported === undefined) continue;
    // Clamp to the adapter's own maximum: asking for more is a hard rejection.
    desired[k] = Math.min(v, supported);
  }

  const device = await requestDeviceWithFallback(adapter, wantFeatures, desired, opts);
  const errorsSeen: string[] = [];
  const hasTimestamp = features.has('timestamp-query') && device.features.has('timestamp-query');

  device.addEventListener('uncapturederror', (ev) => {
    const e = ev as GPUUncapturedErrorEvent;
    const message = e.error.message;
    // Keep the first few distinct validation messages verbatim: Chrome rate
    // limits repeated warnings, and the first one is the one that matters.
    errorsSeen.push(message);
    if (errorsSeen.filter((m) => m === message).length <= 2) {
      opts.onMessage?.('error', `[webgpu] ${message}`);
    }
  });

  const context = canvas.getContext('webgpu') as GPUCanvasContext | null;
  if (!context) {
    throw new BackendUnavailable('canvas.getContext("webgpu") returned null');
  }

  // Prefer a plain unorm format and do the sRGB encode ourselves in the final
  // shader: that keeps every intermediate buffer linear and gives exact control
  // over dithering *after* the transfer function.
  //
  // COPY_SRC is included so the in-app self test can read the presented frame
  // back with copyTextureToBuffer; getting an image out of a headless WebGPU
  // canvas any other way is not reliable.
  const format: GPUTextureFormat = 'bgra8unorm';
  context.configure({
    device,
    format,
    alphaMode: 'opaque',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
  });

  const info = adapter.info
    ? {
        vendor: adapter.info.vendor || '',
        architecture: adapter.info.architecture || '',
        device: adapter.info.device || '',
        description: adapter.info.description || '',
      }
    : { vendor: '', architecture: '', device: '', description: '' };

  return {
    adapter,
    device,
    format,
    context,
    canvas,
    hasTimestamp,
    maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
    maxBufferSize: device.limits.maxBufferSize,
    maxComputeInvocations: device.limits.maxComputeInvocationsPerWorkgroup,
    limits: device.limits,
    features,
    info,
  };
}

async function requestDeviceWithFallback(
  adapter: GPUAdapter,
  features: GPUFeatureName[],
  limits: Record<string, number>,
  opts: DeviceRequestOptions
): Promise<GPUDevice> {
  const attempts: GPUDeviceDescriptor[] = [
    { requiredFeatures: features, requiredLimits: limits, label: 'aetheria-device' },
    { requiredFeatures: features, label: 'aetheria-device-default-limits' },
    { label: 'aetheria-device-minimal' },
  ];
  let lastErr: unknown = null;
  for (const desc of attempts) {
    try {
      return await adapter.requestDevice(desc);
    } catch (err) {
      lastErr = err;
      opts.onMessage?.('warn', `requestDevice() rejected ${JSON.stringify(desc.requiredLimits ?? 'defaults')}: ${String(err)}`);
    }
  }
  throw new BackendUnavailable('requestDevice() failed for every limit configuration', String(lastErr));
}

/** How many particles fit under the adapter's storage binding ceiling (88 B each). */
export function maxParticlesForDevice(ctx: GpuContext): number {
  const byStorage = Math.floor(ctx.maxStorageBufferBindingSize / 88);
  const byBuffer = Math.floor(ctx.maxBufferSize / 88);
  return Math.max(0, Math.min(byStorage, byBuffer, 4_000_000));
}

/**
 * Compile a WGSL module and surface every diagnostic.
 * Returns the module plus the (possibly empty) list of error strings.
 */
export async function compileChecked(
  device: GPUDevice,
  code: string,
  label: string
): Promise<{ module: GPUShaderModule; errors: string[] }> {
  const module = device.createShaderModule({ code, label });
  const errors: string[] = [];
  if (typeof module.getCompilationInfo === 'function') {
    try {
      const info = await module.getCompilationInfo();
      for (const m of info.messages) {
        const where = `${m.lineNum}:${m.linePos}`;
        const text = `${label} ${m.type} @${where}: ${m.message}`;
        if (m.type === 'error') errors.push(text);
        else if (m.type === 'warning') console.warn(`[aetheria] ${text}`);
      }
    } catch {
      /* getCompilationInfo is best-effort */
    }
  }
  if (errors.length) {
    console.error(`[aetheria] WGSL errors in ${label}:\n` + errors.join('\n'));
  }
  return { module, errors };
}
