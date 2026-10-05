/**
 * Aetheria — 4D math.
 *
 * Conventions used consistently across every shader and every backend:
 *   - Right-handed world space, +Y up, camera looks down -Z.
 *   - NDC depth in [0,1] (WebGPU convention).
 *   - A 4D rotation is the ordered composition of six plane rotations:
 *         R = Rzw · Ryw · Rxw · Ryz · Rxz · Rxy
 *     Angles are radians. Composing in a fixed order guarantees the WGSL
 *     compute shader, the WebGPU renderer and the WebGL2 fallback all produce
 *     byte-identical transforms for the same angle vector.
 */

export type Vec4 = [number, number, number, number];

/** Six independent rotation planes of SO(4). */
export interface Rot4 {
  /** Plane angles in radians, ordered: xy, xz, xw, yz, yw, zw. */
  angles: [number, number, number, number, number, number];
}

export function rot4(): Rot4 {
  return { angles: [0, 0, 0, 0, 0, 0] };
}

export function cloneRot4(r: Rot4): Rot4 {
  return { angles: [...r.angles] as Rot4['angles'] };
}

/**
 * 4x4 rotation in the plane spanned by axes `i` and `j` (column-major, GLSL/JS
 * friendly: `m[col*4 + row]`).
 */
export function planeRotation(i: number, j: number, a: number): Float64Array {
  const m = new Float64Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  const c = Math.cos(a);
  const s = Math.sin(a);
  m[i * 4 + i] = c;
  m[j * 4 + j] = c;
  m[i * 4 + j] = -s;
  m[j * 4 + i] = s;
  return m;
}

/** out = a · b for two column-major 4x4 matrices. */
export function mul4(a: Float64Array, b: Float64Array, out = new Float64Array(16)): Float64Array {
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

/**
 * Build the composed 4D rotation matrix. The order matches the shader helper
 * `rot4_from_angles()` in `src/gpu/wgsl/common.wgsl.ts`.
 */
export function rotationMatrix4(angles: readonly number[]): Float64Array {
  const [xy, xz, xw, yz, yw, zw] = angles as [number, number, number, number, number, number];
  let m = planeRotation(0, 1, xy);
  m = mul4(planeRotation(0, 2, xz), m);
  m = mul4(planeRotation(0, 3, xw), m);
  m = mul4(planeRotation(1, 2, yz), m);
  m = mul4(planeRotation(1, 3, yw), m);
  m = mul4(planeRotation(2, 3, zw), m);
  return m;
}

export function rotate4(m: Float64Array, v: readonly number[]): Vec4 {
  return [
    m[0] * v[0] + m[4] * v[1] + m[8] * v[2] + m[12] * v[3],
    m[1] * v[0] + m[5] * v[1] + m[9] * v[2] + m[13] * v[3],
    m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3],
    m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3],
  ];
}

/** Column-major perspective matrix with a [0,1] depth range (WebGPU). */
export function perspective(fovY: number, aspect: number, near: number, far: number, out = new Float32Array(16)): Float32Array {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  out.fill(0);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = far * nf;
  out[11] = -1;
  out[14] = far * near * nf;
  return out;
}

/** Column-major right-handed look-at with a [0,1] depth range (WebGPU). */
export function lookAt(eye: readonly number[], center: readonly number[], up: readonly number[], out = new Float32Array(16)): Float32Array {
  let zx = eye[0] - center[0];
  let zy = eye[1] - center[1];
  let zz = eye[2] - center[2];
  let len = Math.hypot(zx, zy, zz) || 1;
  zx /= len;
  zy /= len;
  zz /= len;

  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  len = Math.hypot(xx, xy, xz) || 1;
  xx /= len;
  xy /= len;
  xz /= len;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  out[0] = xx;
  out[1] = yx;
  out[2] = zx;
  out[3] = 0;
  out[4] = xy;
  out[5] = yy;
  out[6] = zy;
  out[7] = 0;
  out[8] = xz;
  out[9] = yz;
  out[10] = zz;
  out[11] = 0;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;
  return out;
}

/** out = a · b for two column-major 4x4 matrices (Float32). */
export function mulMat4(a: Float32Array, b: Float32Array, out = new Float32Array(16)): Float32Array {
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

export function projectPoint(mvp: Float32Array, x: number, y: number, z: number, out = new Float32Array(4)): Float32Array {
  out[0] = mvp[0] * x + mvp[4] * y + mvp[8] * z + mvp[12];
  out[1] = mvp[1] * x + mvp[5] * y + mvp[9] * z + mvp[13];
  out[2] = mvp[2] * x + mvp[6] * y + mvp[10] * z + mvp[14];
  out[3] = mvp[3] * x + mvp[7] * y + mvp[11] * z + mvp[15];
  return out;
}

/**
 * Invert a 4x4 matrix (column-major). Used to unproject the click ray into
 * world space so a supernova ignites where the pointer actually pointed.
 */
export function invert4(m: Float32Array, out = new Float32Array(16)): Float32Array {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) return out.set(m), out;
  det = 1 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

/** Transform a point by a column-major matrix, returning (x, y, z, w). */
export function xform4(m: Float32Array, x: number, y: number, z: number, w: number): Vec4 {
  return [
    m[0] * x + m[4] * y + m[8] * z + m[12] * w,
    m[1] * x + m[5] * y + m[9] * z + m[13] * w,
    m[2] * x + m[6] * y + m[10] * z + m[14] * w,
    m[3] * x + m[7] * y + m[11] * z + m[15] * w,
  ];
}

export const TAU = Math.PI * 2;
export const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);
export const smoothstep = (e0: number, e1: number, x: number) => {
  const t = clamp((x - e0) / (e1 - e0 || 1e-6), 0, 1);
  return t * t * (3 - 2 * t);
};
export const mix = lerp;
