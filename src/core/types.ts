/**
 * Aetheria — backend-agnostic contracts shared by the WebGPU renderer and the
 * WebGL2 fallback. Both backends are driven exclusively through `Renderer`, so
 * the app shell, UI, audio DSP and input layer never branch on the API.
 */

import type { Params, DeviceProfile } from './config';

export type BackendKind = 'webgpu' | 'webgl2' | 'none';

export interface PointerRay {
  /** Normalised device coordinates of the click, -1..1. */
  ndcX: number;
  ndcY: number;
  /** World-space origin and direction of the unprojected ray. */
  origin: [number, number, number];
  dir: [number, number, number];
}

/** A supernova ignition request, in world space where possible. */
export interface ShockRequest {
  origin: [number, number, number, number];
  strength: number;
  radius: number;
}

export interface HudStats {
  fps: number;
  frameMs: number;
  cpuMs: number;
  gpuMs: number;
  /** GPU time for the simulation compute stage alone, 0 when unmeasurable. */
  computeMs: number;
  simCount: number;
  renderCount: number;
  drawVertices: number;
  renderScale: number;
  drawWidth: number;
  drawHeight: number;
  backend: BackendKind;
  deviceLabel: string;
  gpuTimingSupported: boolean;
  memoryEstimateMB: number;
  degraded: string[];
}

export interface RendererTelemetry {
  onTelemetry?: (stats: HudStats) => void;
  onFatal?: (message: string, detail?: string) => void;
  onDegrade?: (reason: string) => void;
}

export interface Renderer {
  readonly kind: BackendKind;
  readonly profile: DeviceProfile;
  /** Live parameters; the app mutates this object in place and the renderer
   *  picks the change up on the next frame. */
  readonly params: Params;
  readonly stats: HudStats;
  /** True once pipelines exist and the simulation holds live particles. */
  readonly ready: boolean;

  /** Resize the drawing buffer. `scale` folds in dynamic resolution. */
  resize(cssWidth: number, cssHeight: number, dpr: number, scale: number): void;
  /** Advance simulation + draw one frame. */
  frame(nowMs: number, deltaMs: number): void;

  /** Ignite a supernova at a normalised device coordinate. */
  shock(ndcX: number, ndcY: number, strength?: number): void;
  /** Feed the analyser bands, already smoothed, in 0..~2. */
  setAudio(low: number, mid: number, high: number, beat: number): void;
  /** Camera/4D orbit control. */
  orbit(dx: number, dy: number, plane?: 'xw' | 'yw' | 'zw'): void;
  zoom(delta: number): void;
  slice(delta: number): void;
  /** Reset camera, rotation and simulation to a fresh state. */
  reset(): void;
  /** Snapshot the current frame as a PNG blob (best effort, HDR tone mapped). */
  capture(): Promise<Blob | null>;
  /** Rolling performance hooks. */
  setQualityScale(scale: number): void;
  dispose(): void;
}

/** Thrown when WebGPU cannot be brought up; the shell falls back to WebGL2. */
export class BackendUnavailable extends Error {
  constructor(message: string, readonly detail?: string) {
    super(message);
    this.name = 'BackendUnavailable';
  }
}

/**
 * Estimate peak GPU memory for a particle system, used by the HUD and by the
 * budget guard that refuses to allocate beyond `HARD_MAX_SIM_PARTICLES`.
 */
export function estimateParticleMemory(simCount: number): number {
  return simCount * 88 * 2; // 2x (vec4 pos + vec4 vel + vec4 colour + vec4 meta) * 8 ping-pong
}
