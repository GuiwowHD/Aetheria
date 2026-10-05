/**
 * Aetheria -> WGSL common prelude.
 *
 * Prepended to every shader module. It carries:
 *   - the exact uniform layouts (kept in lockstep with `src/core/uniforms.ts`)
 *   - the particle storage layout (kept in lockstep with `PARTICLE_LAYOUT`)
 *   - hash / RNG helpers used for deterministic, allocation-free respawn
 *   - the 4D rotation algebra
 *   - an analytic, exactly divergence-free 4D curl-noise field
 *   - the palette + ACES transfer functions shared by render and post
 *
 * Everything here is dependency-free WGSL with no feature requirements beyond
 * `f32` arithmetic, so it compiles on every WebGPU implementation down to
 * SwiftShader.
 */

export const WGSL_CONSTANTS = /* wgsl */ `
const PI: f32 = 3.141592653589793;
const TAU: f32 = 6.283185307179586;
const INV_PI: f32 = 0.3183098861837907;

// Particle storage layout (structure-of-arrays, 6 arrays):
//   P[i] = vec4(x, y, z, w)                     position in R4
//   V[i] = vec4(vx, vy, vz, vw)                 velocity in R4
//   C[i] = vec4(r, g, b, energy)                linear HDR colour + scalar energy
//   M[i] = vec4(age, lifespan, seed, speed)     lifecycle + deterministic seed
const ARRAY_P = 0u;
const ARRAY_V = 1u;
const ARRAY_C = 2u;
const ARRAY_M = 3u;
`;

/**
 * ---------------------------------------------------------------------------
 * Uniform layout contract (std140 / WGSL-uniform compatible, 16B aligned).
 * ---------------------------------------------------------------------------
 * struct Sim {                 // offset  size
 *   vec4 uTime;                //   0     16   x=t, y=dt, z=frame, w=simCount
 *   vec4 uViewport;            //  16     16   x=w, y=h, z=1/w, w=1/h
 *   vec4 uCamera;              //  32     16   xyz=eye, w=zoom(projection distance)
 *   vec4 uSlice;               //  48     16   x=wSlice, y=4D fov, z=nearBlend, w=aspect
 *   vec4 uJulia;               //  64     16   (a,b,c,d) Julia quaternion
 *   vec4 uForce;               //  80     16   x=curl y=damping z=gravity w=confinement
 *   vec4 uAttract;             //  96     16   x=depth y=escape z=power w=warp
 *   vec4 uAudio;               // 112     16   x=low y=mid z=high w=beat
 *   vec4 uPalette;             // 128     16   x=hue y=saturation z=paletteMix w=exposure
 *   vec4 uRender;              // 144     16   x=sizeScale y=energyScale z=multiplicity w=speed
 *   mat4x4 uProj;              // 160     64
 *   mat4x4 uView;              // 224     64
 *   mat4x4 uRot4;              // 288     64
 *   vec4 uRot0;                // 352     16   plane angles (xy, xz, xw, yz)
 *   vec4 uRot1;                // 368     16   plane angles (yw, zw, tanHalfFov, time4D)
 *   vec4 uMisc;                // 384     16   x=shockGain y=bloom z=dpr w=quality
 *   vec4 uFocus;               // 400     16   x=focusDist y=aperture z=grain w=vignette
 *   vec4 uShockMisc;           // 416     16   x=activeShocks y=shockRadius z=thickness w=life
 *   vec4 uMisc2;               // 432     16   x=trailDecay y=paletteMix z=saturation w=reserved
 * }                            // total 448
 */
export const SIM_UNIFORM_BYTES = 448;

/**
 * ---------------------------------------------------------------------------
 * Uniform declarations only.
 *
 * Deliberately split from the storage bindings: WebGPU rejects any vertex or
 * fragment stage that pulls in `var<storage, read_write>` ("var with storage
 * address space and read_write access mode cannot be used by vertex pipeline
 * stage"), so the render path includes this plus WGSL_STORAGE_READONLY, while
 * the compute kernels include WGSL_STORAGE_BINDINGS.
 * ---------------------------------------------------------------------------
 */
export const WGSL_UNIFORMS = /* wgsl */ `
struct Sim {
  uTime     : vec4f,
  uViewport : vec4f,
  uCamera   : vec4f,
  uSlice    : vec4f,
  uJulia    : vec4f,
  uForce    : vec4f,
  uAttract  : vec4f,
  uAudio    : vec4f,
  uPalette  : vec4f,
  uRender   : vec4f,
  uProj     : mat4x4f,
  uView     : mat4x4f,
  uRot4     : mat4x4f,
  uRot0     : vec4f,
  uRot1     : vec4f,
  uMisc     : vec4f,
  uFocus    : vec4f,
  uShockMisc: vec4f,
  uMisc2    : vec4f,
};

@group(0) @binding(0) var<uniform> U : Sim;
`;

/** Compute-stage (writable) storage bindings: simulate, seed and copy only. */
export const WGSL_STORAGE_BINDINGS = /* wgsl */ `
@group(0) @binding(1) var<storage, read_write> arrPingP : array<vec4f>;
@group(0) @binding(2) var<storage, read_write> arrPingV : array<vec4f>;
@group(0) @binding(3) var<storage, read_write> arrPingC : array<vec4f>;
@group(0) @binding(4) var<storage, read_write> arrPingM : array<vec4f>;
@group(0) @binding(5) var<storage, read_write> arrPongP : array<vec4f>;
@group(0) @binding(6) var<storage, read_write> arrPongV : array<vec4f>;
@group(0) @binding(7) var<storage, read_write> arrPongC : array<vec4f>;
@group(0) @binding(8) var<storage, read_write> arrPongM : array<vec4f>;
@group(0) @binding(9) var<storage, read_write> shocks : array<vec4f>;
`;

/** Render-stage (read-only) storage bindings, matching the render bind layout. */
export const WGSL_STORAGE_READONLY = /* wgsl */ `
@group(0) @binding(1) var<storage, read> arrPingP : array<vec4f>;
@group(0) @binding(2) var<storage, read> arrPingV : array<vec4f>;
@group(0) @binding(3) var<storage, read> arrPingC : array<vec4f>;
@group(0) @binding(4) var<storage, read> arrPingM : array<vec4f>;
@group(0) @binding(9) var<storage, read> shocks : array<vec4f>;
`;

/**
 * ---------------------------------------------------------------------------
 * Post-process uniform layout.
 * ---------------------------------------------------------------------------
 * struct Post {
 *   vec4 pResolution;   //  0  x=w y=h z=1/w w=1/h
 *   vec4 pTexel;        // 16  xy = blur direction (per mip), zw = mip texel size
 *   vec4 pParams0;      // 32  x=exposure y=bloom z=bloomRadius w=threshold
 *   vec4 pParams1;      // 48  x=chroma y=grain z=vignette w=dof
 *   vec4 pParams2;      // 64  x=volumetric y=time z=aspect w=frame
 *   vec4 pFocus;        // 80  x=focusDepth y=focusRange z=maxCoC w=nearPlane
 *   vec4 pFar;          // 96  x=farPlane y=sourceLod z=upsampleRadius w=bloomLevels
 *   vec4 pColor;        //112  x=hue y=saturation z=quality w=sceneLuma
 * }                     // 128 bytes
 */
export const POST_UNIFORM_BYTES = 128;

export const WGSL_POST_UNIFORM = /* wgsl */ `
struct Post {
  pResolution : vec4f,
  pTexel      : vec4f,
  pParams0    : vec4f,
  pParams1    : vec4f,
  pParams2    : vec4f,
  pFocus      : vec4f,
  pFar        : vec4f,
  pColor      : vec4f,
};
`;

/**
 * ---------------------------------------------------------------------------
 * Math helpers that depend on NO binding: hashing, palette, tonemapping.
 *
 * Kept separate from WGSL_SIM_MATH on purpose. The post-processing shaders bind
 * @group(0) @binding(4) to the Post uniform, so pulling in the Sim-dependent
 * helpers (which reference U) would produce dangling bindings and fail module
 * creation. Bloom, DOF and composite include only this prelude plus the post
 * uniform, which is why it contains no reference to the Sim block at all.
 * ---------------------------------------------------------------------------
 */
export const WGSL_CORE_MATH = /* wgsl */ `
// ---------------------------------------------------------------------------
// Deterministic integer hashing. hash3 is a 3-round PCG-style mix: cheap
// (a handful of imad + xorshift) yet good enough that respawn positions show no
// lattice structure even at 4M particles.
// ---------------------------------------------------------------------------
fn hash3(p_in : vec3u) -> u32 {
  var p = p_in * 1664525u + 1013904223u;
  p.x = p.x + (p.y * p.z);
  p.y = p.y + (p.z * p.x);
  p.z = p.z + (p.x * p.y);
  // Component-wise right shift: WGSL has no vector-by-scalar shift operator.
  p = vec3u(p.x >> 16u, p.y >> 16u, p.z >> 16u) ^ p;
  p.x = p.x + (p.y * p.z);
  p.y = p.y + (p.z * p.x);
  p.z = p.z + (p.x * p.y);
  return p.x ^ p.y ^ p.z;
}

fn hash1(n : u32) -> u32 {
  return hash3(vec3u(n, n * 0x9E3779B9u, 0x85EBCA6Bu));
}

fn rnd1(seed : u32) -> f32 { return f32(hash1(seed)) * (1.0 / 4294967296.0); }

/** Uniform float in [lo, hi) from a scalar seed and a stream index. */
fn rndRange(seed : u32, stream : u32, lo : f32, hi : f32) -> f32 {
  return lo + (hi - lo) * rnd1(seed * 747796405u + stream * 2891336453u + 1u);
}

/** Uniform point on the unit 3-sphere embedded in R4 (Marsaglia-style). */
fn rndDir4(seed : u32, stream : u32) -> vec4f {
  let w = rndRange(seed, stream + 0u, -1.0, 1.0);
  let t = sqrt(max(0.0, 1.0 - w * w));
  let a = rndRange(seed, stream + 1u, 0.0, TAU);
  let b = rndRange(seed, stream + 2u, 0.0, TAU);
  return vec4f(t * cos(a), t * sin(a), t * sin(b) * 0.92, w * 0.86);
}

// ---------------------------------------------------------------------------
// Cosmology palette: magenta -> cyan -> gold -> violet, expressed in linear
// HDR. t is a wrapped 0..1 phase.
// ---------------------------------------------------------------------------
fn palette(t : f32) -> vec3f {
  let a = vec3f(0.62, 0.06, 0.94); // magenta/violet
  let b = vec3f(0.05, 0.86, 0.98); // cyan
  let c = vec3f(1.00, 0.72, 0.16); // gold
  let d = vec3f(0.48, 0.16, 0.92); // deep purple
  let x = fract(t) * 4.0;
  if (x < 1.0) { return mix(a, b, smoothstep(0.0, 1.0, x)); }
  if (x < 2.0) { return mix(b, c, smoothstep(0.0, 1.0, x - 1.0)); }
  if (x < 3.0) { return mix(c, d, smoothstep(0.0, 1.0, x - 2.0)); }
  return mix(d, a, smoothstep(0.0, 1.0, x - 3.0));
}

/** Hue rotation around the luma axis (Rodrigues), applied in linear space. */
fn hueRotate(rgb : vec3f, turns : f32) -> vec3f {
  let a = turns * TAU;
  let k = vec3f(0.57735026919);
  let ca = cos(a);
  let sa = sin(a);
  return rgb * ca + cross(k, rgb) * sa + k * dot(k, rgb) * (1.0 - ca);
}

fn luma(c : vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }

/** ACES filmic approximation (Stephen Hill's fit). */
fn acesFilm(x_in : vec3f) -> vec3f {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x_in * (a * x_in + b)) / (x_in * (c * x_in + d) + e), vec3f(0.0), vec3f(1.0));
}

/** ACES RRT+ODT fit evaluated through the ACEScg matrices: richer shadows. */
fn acesApprox(x : vec3f) -> vec3f {
  let m1 = mat3x3f(
    vec3f(0.59719, 0.07600, 0.02840),
    vec3f(0.35458, 0.90834, 0.13383),
    vec3f(0.04823, 0.01566, 0.83777)
  );
  let m2 = mat3x3f(
    vec3f( 1.60475, -0.10208, -0.00327),
    vec3f(-0.53108,  1.10813, -0.07276),
    vec3f(-0.07367, -0.00605,  1.07602)
  );
  let v0 = m1 * x;
  let a = v0 * (v0 + 0.0245786) - 0.000090537;
  let b = v0 * (0.983729 * v0 + 0.4329510) + 0.238081;
  return clamp(m2 * (a / b), vec3f(0.0), vec3f(1.0));
}

fn srgbEncode(c : vec3f) -> vec3f {
  let lo = c * 12.92;
  let hi = 1.055 * pow(max(c, vec3f(1e-5)), vec3f(1.0 / 2.4)) - 0.055;
  return select(hi, lo, c <= vec3f(0.0031308));
}

/** Interleaved gradient noise: the classic 1-sample-per-pixel dither source. */
fn ign(p : vec2f) -> f32 {
  return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

fn bayer8(p : vec2i) -> f32 {
  let x = u32(p.x) & 7u;
  let y = u32(p.y) & 7u;
  var v = 0u;
  // 8x8 ordered dither built by bit interleaving: no lookup table needed.
  for (var i = 0u; i < 3u; i = i + 1u) {
    v = v | (((x >> i) & 1u) << (2u * i)) | (((y >> i) & 1u) << (2u * i + 1u));
  }
  return f32(v) * (1.0 / 64.0);
}

fn softCircle(d : vec2f, radius : f32, hardness : f32) -> f32 {
  let r = length(d) / max(radius, 1e-6);
  return 1.0 - smoothstep(hardness, 1.0, r);
}
`;

/**
 * ---------------------------------------------------------------------------
 * Math helpers that read the Sim uniform (`U`): the 4D rotation algebra, the
 * analytic 4D curl-noise field and the fractal attractors. Only the particle
 * simulation and particle rasterisation may include this.
 * ---------------------------------------------------------------------------
 */
export const WGSL_SIM_MATH = /* wgsl */ `
// ---------------------------------------------------------------------------
// 4D rotation: rotation in each of the six planes, composed in a fixed order
// that matches rotationMatrix4() in src/core/math4d.ts and the exact
// column-major matrix uploaded in uRot4.
// ---------------------------------------------------------------------------
fn rot4_plane(v : vec4f, i : u32, j : u32, ang : f32) -> vec4f {
  let c = cos(ang);
  let s = sin(ang);
  let a = v[i];
  let b = v[j];
  var o = v;
  o[i] = a * c - b * s;
  o[j] = a * s + b * c;
  return o;
}

/** Apply all six plane rotations: R = Rzw . Ryw . Rxw . Ryz . Rxz . Rxy */
fn rot4_apply(v : vec4f) -> vec4f {
  var o = v;
  o = rot4_plane(o, 0u, 1u, U.uRot0.x); // xy
  o = rot4_plane(o, 0u, 2u, U.uRot0.y); // xz
  o = rot4_plane(o, 0u, 3u, U.uRot0.z); // xw
  o = rot4_plane(o, 1u, 2u, U.uRot0.w); // yz
  o = rot4_plane(o, 1u, 3u, U.uRot1.x); // yw
  o = rot4_plane(o, 2u, 3u, U.uRot1.y); // zw
  return o;
}

// ---------------------------------------------------------------------------
// Analytic 4D curl noise.
//
// Take the vector potential A = (s0, s1, s2, s0*s1*s2) with
// s_k = sin(dot(w_k, x) + phase_k). The result is *provably* divergence-free
// (div curl A == 0 identically), which is what gives the nebula its
// incompressible, silk-like filamentary flow instead of noisy jitter.
// ---------------------------------------------------------------------------
fn curl4(p : vec4f, t : f32, gain : f32, warp : f32) -> vec4f {
  var curl = vec4f(0.0);
  var amp = 1.0;
  var freq = 1.0;
  for (var k = 0u; k < 3u; k = k + 1u) {
    let f = freq;
    let a = amp;
    // Three axis-tilted wave vectors keep the field anisotropic (filaments,
    // not blobs) without breaking the divergence-free property.
    let w0 = vec4f(0.83, 1.31, 0.57, 1.07) * f;
    let w1 = vec4f(1.21, 0.49, 1.37, 0.73) * f;
    let w2 = vec4f(0.61, 1.13, 0.91, 1.49) * f;
    let ph = t * (0.55 + 0.31 * f) + warp;

    let s0 = sin(dot(w0, p) + ph * 1.13);
    let s1 = sin(dot(w1, p) + ph * 0.87);
    let s2 = sin(dot(w2, p) + ph * 1.41);
    let c0 = cos(dot(w0, p) + ph * 1.13) * a;
    let c1 = cos(dot(w1, p) + ph * 0.87) * a;
    let c2 = cos(dot(w2, p) + ph * 1.41) * a;

    let g0 = c0 * w0;
    let g1 = c1 * w1;
    let g2 = c2 * w2;
    let g3 = (c0 * w0 * s1 * s2 + c1 * w1 * s0 * s2 + c2 * w2 * s0 * s1);
    let g = g0 + g1 + g2 + g3;

    curl = curl + vec4f(
      (g.y - g.z) + (g.w * 0.5),
      (g.z - g.w) + (g.x * 0.5),
      (g.w - g.x) + (g.y * 0.5),
      (g.x - g.y) + (g.z * 0.5)
    );

    amp = amp / 1.7;
    freq = freq * 2.03;
  }
  return curl * gain;
}

// ---------------------------------------------------------------------------
// 4D quaternion Julia force field. Returns a displacement in R4.
// ---------------------------------------------------------------------------
fn julia4(z_in : vec4f, c : vec4f, iterations : i32) -> vec4f {
  var z = z_in;
  var acc = vec4f(0.0);
  for (var i = 0; i < 16; i = i + 1) {
    if (i >= iterations) { break; }
    // Quaternion square: (w + xi + yj + zk)^2
    let w = z.w;
    let x = z.x;
    let y = z.y;
    let zz = z.z;
    let nz = vec4f(
      2.0 * (x * w - y * zz),
      2.0 * (y * w + x * zz),
      2.0 * (zz * w - x * y),
      w * w - x * x - y * y - zz * zz
    );
    z = nz + c;
    acc = acc + z * exp(-0.85 * f32(i));
    if (length(z) > 6.0) { break; }
  }
  return acc;
}

/**
 * 4D Mandelbulb deformation: returns a direction + magnitude in R4 that pushes
 * particles along the iso-surface of the deformed bulb. power is driven by the
 * "fractal dimension" slider, which is why that slider visibly changes the
 * topology of the nebula rather than just its brightness.
 */
fn mandelbulb4(p : vec4f, power : f32, warp : f32) -> vec4f {
  let r = length(p) + 1e-5;
  let theta = acos(clamp(p.w / r, -1.0, 1.0));
  let phi = atan2(p.y, p.x);
  let psi = atan2(p.z, sqrt(p.x * p.x + p.y * p.y) + 1e-6);
  let zr = pow(r, power);
  let nt = theta * power;
  let np = phi * power;
  let ns = psi * power;
  let q = vec4f(
    zr * sin(nt) * cos(np) * sin(ns),
    zr * sin(nt) * sin(np) * sin(ns),
    zr * sin(nt) * cos(ns),
    zr * cos(nt)
  );
  let d = q - p;
  return d / (1.0 + dot(d, d)) * warp;
}
`;

export const WGSL_MATH = WGSL_CORE_MATH + WGSL_SIM_MATH;
