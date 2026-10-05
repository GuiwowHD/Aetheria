/**
 * Aetheria — constants bridge.
 *
 * The WebGL2 fallback lives in its own directory and must not reach into
 * `src/gpu/**`, so everything it legitimately needs (sizes, counts, ranges) is
 * re-exported from here. Keeping one hop of indirection means the fallback can
 * never accidentally import a WebGPU type and break its build on a browser that
 * has no `navigator.gpu` at all.
 */

export {
  MAX_RENDER_PARTICLES,
  HARD_MAX_SIM_PARTICLES,
  PARTICLE_BYTES_PER_SIM,
  PARAM_RANGES,
  clampParam,
  defaultParams,
  detectProfile,
  type Params,
  type DeviceProfile,
  type QualityTier,
  type RangeKey,
} from '../core/config';

export { SIM_UNIFORM_BYTES, POST_UNIFORM_BYTES } from './wgsl/common.wgsl';

export { BackendUnavailable, estimateParticleMemory, type Renderer, type HudStats, type RendererTelemetry, type BackendKind } from '../core/types';

export {
  TAU,
  clamp,
  lerp,
  mix,
  smoothstep,
  rot4,
  cloneRot4,
  rotationMatrix4,
  rotate4,
  perspective,
  lookAt,
  mulMat4,
  invert4,
  xform4,
  projectPoint,
  type Rot4,
  type Vec4,
} from '../core/math4d';

export {
  SimWriter,
  PostWriter,
  SIM_OFF,
  POST_OFF,
  makeSimState,
  makePostState,
  type SimState,
  type PostState,
} from '../core/uniforms';
