/**
 * Aetheria — WebGL2 helper layer for the fallback backend.
 *
 * Small, allocation-conscious wrappers around the WebGL2 API:
 *   - shader compilation / program linking with *full* info logs collected into
 *     `BackendUnavailable.detail` (a WebGL2 shader failure must never be a
 *     mystery; the shell surfaces `detail` verbatim);
 *   - program reflection so uniform locations are resolved once, never per frame;
 *   - colour render target + optional sampleable depth texture creation with
 *     completeness checks;
 *   - float-renderability probing (`EXT_color_buffer_float` / half-float).
 *
 * Everything here is dependency-free and touches no other Aetheria module except
 * the `BackendUnavailable` error contract.
 */

import { BackendUnavailable } from '../core/types';

/** Reflected uniform locations. `undefined` means "not active in this program". */
export interface UniformBag {
  [name: string]: WebGLUniformLocation | null | undefined;
}

/** Uniform block binding points shared by every program in the fallback. */
export const SIM_BLOCK_BINDING = 0;
export const POST_BLOCK_BINDING = 1;

export const SIM_BLOCK_NAME = 'SimBlock';
export const POST_BLOCK_NAME = 'PostBlock';

/** Prefixes every line of a shader with its line number (error reporting only). */
export function describeSource(source: string): string {
  const lines = source.split('\n');
  const out: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    out.push(`${String(i + 1).padStart(4, ' ')} | ${lines[i]}`);
  }
  return out.join('\n');
}

export function compileShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
  label: string,
): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) {
    throw new BackendUnavailable('WebGL2: gl.createShader() returned null', label);
  }
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? '(no info log)';
    gl.deleteShader(shader);
    const stage = type === gl.VERTEX_SHADER ? 'vertex' : 'fragment';
    throw new BackendUnavailable(
      `WebGL2 ${stage} shader failed to compile: ${label}`,
      `${label}\n${log}\n--- source ---\n${describeSource(source)}`,
    );
  }
  // A non-empty log on success is a warning; keep it visible but do not fail.
  const warn = gl.getShaderInfoLog(shader);
  if (warn && warn.trim().length > 0) {
    console.error(`[webgl2] shader warning in ${label}: ${warn.trim()}`);
  }
  return shader;
}

/**
 * A linked program plus its reflected uniform locations. Uniform lookups are a
 * plain property read (no Map hashing, no `getUniformLocation` in `frame()`).
 */
export class GlProgram {
  readonly handle: WebGLProgram;
  readonly label: string;
  private readonly gl: WebGL2RenderingContext;
  private readonly uniforms: UniformBag;

  constructor(
    gl: WebGL2RenderingContext,
    handle: WebGLProgram,
    uniforms: UniformBag,
    label: string,
  ) {
    this.gl = gl;
    this.handle = handle;
    this.uniforms = uniforms;
    this.label = label;
  }

  /** Location of a uniform, or null when the compiler removed it. */
  loc(name: string): WebGLUniformLocation | null {
    return this.uniforms[name] ?? null;
  }

  /** Names of the active (non-removed) uniforms, for diagnostics. */
  activeNames(): string[] {
    return Object.keys(this.uniforms);
  }

  /**
   * Fail loudly when a uniform the pipeline depends on was optimised away —
   * this is the "silent black screen" guard.
   */
  require(names: readonly string[]): void {
    const missing: string[] = [];
    for (const name of names) {
      if ((this.uniforms[name] ?? null) === null) missing.push(name);
    }
    if (missing.length > 0) {
      throw new BackendUnavailable(
        `WebGL2 program '${this.label}' is missing ${missing.length} expected uniform(s)`,
        `label: ${this.label}\nmissing: ${missing.join(', ')}\nactive: ${this.activeNames().join(', ')}`,
      );
    }
  }

  dispose(): void {
    this.gl.deleteProgram(this.handle);
  }
}

/**
 * Compile + link a program. `tfVaryings`, when given, are captured in the exact
 * order supplied (SEPARATE_ATTRIBS: one buffer object per varying).
 */
export function createProgram(
  gl: WebGL2RenderingContext,
  vertexSource: string,
  fragmentSource: string,
  label: string,
  tfVaryings?: readonly string[],
): GlProgram {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertexSource, `${label}.vert`);
  const fs = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource, `${label}.frag`);
  const handle = gl.createProgram();
  if (!handle) {
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    throw new BackendUnavailable('WebGL2: gl.createProgram() returned null', label);
  }
  gl.attachShader(handle, vs);
  gl.attachShader(handle, fs);
  if (tfVaryings && tfVaryings.length > 0) {
    gl.transformFeedbackVaryings(handle, [...tfVaryings], gl.SEPARATE_ATTRIBS);
  }
  gl.linkProgram(handle);
  // Shaders stay alive while attached; detach + delete so the driver can free them.
  gl.detachShader(handle, vs);
  gl.detachShader(handle, fs);
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  if (!gl.getProgramParameter(handle, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(handle) ?? '(no info log)';
    gl.deleteProgram(handle);
    throw new BackendUnavailable(
      `WebGL2 program failed to link: ${label}`,
      `${label}\n${log}\n--- vertex ---\n${describeSource(vertexSource)}\n--- fragment ---\n${describeSource(fragmentSource)}`,
    );
  }

  const uniforms: UniformBag = {};
  const count = gl.getProgramParameter(handle, gl.ACTIVE_UNIFORMS) as number;
  for (let i = 0; i < count; i++) {
    const info = gl.getActiveUniform(handle, i);
    if (!info) continue;
    const name = info.name;
    uniforms[name] = gl.getUniformLocation(handle, name);
    // Arrays reflect as `name[0]`; also expose the bare name for convenience.
    if (name.endsWith('[0]')) {
      const bare = name.slice(0, -3);
      uniforms[bare] = uniforms[name] ?? null;
    }
  }

  bindStandardBlocks(gl, handle);
  return new GlProgram(gl, handle, uniforms, label);
}

/**
 * Wire `SimBlock`/`PostBlock` to their fixed binding points. Blocks that the
 * compiler removed report INVALID_INDEX and are skipped (binding a removed block
 * is an error).
 */
export function bindStandardBlocks(gl: WebGL2RenderingContext, program: WebGLProgram): void {
  const sim = gl.getUniformBlockIndex(program, SIM_BLOCK_NAME);
  if (sim !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, sim, SIM_BLOCK_BINDING);
  const post = gl.getUniformBlockIndex(program, POST_BLOCK_NAME);
  if (post !== gl.INVALID_INDEX) gl.uniformBlockBinding(program, post, POST_BLOCK_BINDING);
}

/** Bind a vec4 attribute array (stride 16) with an optional instance divisor. */
export function bindAttributeBuffer(
  gl: WebGL2RenderingContext,
  location: number,
  buffer: WebGLBuffer,
  divisor: number,
): void {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, 4, gl.FLOAT, false, 16, 0);
  gl.vertexAttribDivisor(location, divisor);
}

export interface RenderTargetOptions {
  width: number;
  height: number;
  /** Sized internal format, e.g. gl.RGBA16F / gl.RGBA8. */
  internalFormat: number;
  /** Pixel format matching `internalFormat`. */
  format: number;
  /** Pixel type matching `internalFormat`. */
  type: number;
  /** Bytes per pixel, used for the HUD memory estimate. */
  bytesPerPixel: number;
  filter: number;
  /** Attach a sampleable DEPTH_COMPONENT24 texture. */
  depth: boolean;
  label: string;
}

/** An FBO with one colour texture and (optionally) a sampleable depth texture. */
export interface RenderTarget {
  readonly fbo: WebGLFramebuffer;
  readonly color: WebGLTexture;
  readonly depth: WebGLTexture | null;
  width: number;
  height: number;
  readonly bytes: number;
  readonly label: string;
}

export function createRenderTarget(
  gl: WebGL2RenderingContext,
  options: RenderTargetOptions,
): RenderTarget {
  const w = Math.max(1, Math.floor(options.width));
  const h = Math.max(1, Math.floor(options.height));

  const color = gl.createTexture();
  const fbo = gl.createFramebuffer();
  if (!color || !fbo) {
    if (color) gl.deleteTexture(color);
    if (fbo) gl.deleteFramebuffer(fbo);
    throw new BackendUnavailable(`WebGL2: could not allocate render target '${options.label}'`);
  }

  gl.bindTexture(gl.TEXTURE_2D, color);
  gl.texImage2D(gl.TEXTURE_2D, 0, options.internalFormat, w, h, 0, options.format, options.type, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, options.filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, options.filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_BASE_LEVEL, 0);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAX_LEVEL, 0);

  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, color, 0);
  // Single colour attachment; be explicit rather than relying on default state.
  gl.drawBuffers([gl.COLOR_ATTACHMENT0]);

  let depth: WebGLTexture | null = null;
  if (options.depth) {
    depth = gl.createTexture();
    if (!depth) {
      gl.deleteTexture(color);
      gl.deleteFramebuffer(fbo);
      throw new BackendUnavailable(`WebGL2: could not allocate depth texture for '${options.label}'`);
    }
    gl.bindTexture(gl.TEXTURE_2D, depth);
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, w, h, 0, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null,
    );
    // Depth textures are only filterable with NEAREST; anything else is incomplete.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, depth, 0);
  }

  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteTexture(color);
    if (depth) gl.deleteTexture(depth);
    gl.deleteFramebuffer(fbo);
    throw new BackendUnavailable(
      `WebGL2: incomplete framebuffer '${options.label}' (${w}x${h})`,
      `status = 0x${status.toString(16)}`,
    );
  }

  const bytes = w * h * (options.bytesPerPixel + (options.depth ? 4 : 0));
  return { fbo, color, depth, width: w, height: h, bytes, label: options.label };
}

export function deleteRenderTarget(gl: WebGL2RenderingContext, target: RenderTarget | null): void {
  if (!target) return;
  gl.deleteFramebuffer(target.fbo);
  gl.deleteTexture(target.color);
  if (target.depth) gl.deleteTexture(target.depth);
}

export interface FloatProbe {
  /** True when RGBA16F is colour-renderable (HDR pipeline available). */
  readonly floatRenderable: boolean;
  readonly internalFormat: number;
  readonly type: number;
  readonly bytesPerPixel: number;
  /** Non-null when the pipeline had to fall back to RGBA8. */
  readonly degradedReason: string | null;
}

/**
 * Probe for HDR colour attachments. `EXT_color_buffer_float` covers RGBA16F;
 * `EXT_color_buffer_half_float` also makes RGBA16F renderable on some drivers.
 * RGBA16F (not 32F) is used because it is texture-filterable in core WebGL2.
 *
 * The extension check is backed by a real 4x4 framebuffer test: a driver that
 * advertises the extension but rejects the attachment must degrade, not throw.
 */
export function probeFloatColorBuffer(gl: WebGL2RenderingContext): FloatProbe {
  const fallback: FloatProbe = {
    floatRenderable: false,
    internalFormat: gl.RGBA8,
    type: gl.UNSIGNED_BYTE,
    bytesPerPixel: 4,
    degradedReason: 'no-float-hdr',
  };
  const advertised = gl.getExtension('EXT_color_buffer_float') ?? gl.getExtension('EXT_color_buffer_half_float');
  if (!advertised) return fallback;

  const texture = gl.createTexture();
  const fbo = gl.createFramebuffer();
  if (!texture || !fbo) {
    if (texture) gl.deleteTexture(texture);
    if (fbo) gl.deleteFramebuffer(fbo);
    return fallback;
  }
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, 4, 4, 0, gl.RGBA, gl.HALF_FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.bindTexture(gl.TEXTURE_2D, null);
  gl.deleteFramebuffer(fbo);
  gl.deleteTexture(texture);

  if (!ok) return fallback;
  return {
    floatRenderable: true,
    internalFormat: gl.RGBA16F,
    type: gl.HALF_FLOAT,
    bytesPerPixel: 8,
    degradedReason: null,
  };
}

/** Renderer string from `WEBGL_debug_renderer_info` when the driver exposes it. */
export function getDeviceLabel(gl: WebGL2RenderingContext): string {
  try {
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    if (dbg) {
      const name = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL);
      if (typeof name === 'string' && name.length > 0) return name;
    }
  } catch {
    /* privacy-restricted browsers throw here; the generic label is fine */
  }
  return 'WebGL2';
}

/**
 * An empty VAO for attribute-less full-screen passes: the 3 vertices come from
 * `gl_VertexID`, but WebGL2 is happiest with *some* VAO bound.
 */
export function createFullscreenVao(gl: WebGL2RenderingContext): WebGLVertexArrayObject {
  const vao = gl.createVertexArray();
  if (!vao) throw new BackendUnavailable('WebGL2: gl.createVertexArray() returned null');
  gl.bindVertexArray(vao);
  gl.bindVertexArray(null);
  return vao;
}
