/**
 * Aetheria — GLSL ES 3.00 sources for the WebGL2 fallback backend.
 *
 * Every shader is a template literal so that shader source and the uniform
 * vocabulary that feeds it (`src/core/uniforms.ts`) live in the same repository
 * and cannot drift apart. The math helpers below are line-by-line ports of the
 * WGSL prelude in `src/gpu/wgsl/common.wgsl.ts`; when that file changes, these
 * must change with it (hash, 4D rotation, curl field, Julia/Mandelbulb forces,
 * palette, ACES, sRGB and interleaved-gradient noise are all shared vocabulary).
 *
 * Layout rules honoured by every source in this file:
 *   - `#version 300 es` is the very first characters of the source;
 *   - `precision highp float; precision highp int;` immediately follow it;
 *   - no `varying`/`attribute`/`texture2D`/`gl_FragColor` legacy names;
 *   - explicit `float(...)`/`int(...)` conversions, no implicit int->float;
 *   - all loops have constant bounds (dynamic `break` only);
 *   - uniform blocks are std140 and byte-compatible with `SimWriter`/`PostWriter`.
 */

/** Version + precision header. Must stay first in every concatenated source. */
const HEAD = `#version 300 es
precision highp float;
precision highp int;
`;

/** std140 mirror of `SimState` (see SIM_OFF in src/core/uniforms.ts). */
export const SIM_BLOCK_GLSL = `
layout(std140) uniform SimBlock {
  vec4 uTime;      // x=t y=dt z=frame w=simCount
  vec4 uViewport;  // x=w y=h z=1/w w=1/h
  vec4 uCamera;    // xyz=eye w=zoom
  vec4 uSlice;     // x=wSlice y=fov4 z=nearBlend w=aspect
  vec4 uJulia;     // quaternion Julia constant
  vec4 uForce;     // x=curl y=damping z=gravity w=confinement
  vec4 uAttract;   // x=depth y=escape z=power w=warp
  vec4 uAudio;     // x=low y=mid z=high w=beat
  vec4 uPalette;   // x=hue y=saturation z=paletteMix w=exposure
  vec4 uRender;    // x=sizeScale y=energyScale z=multiplicity w=trailDecay
  mat4 uProj;
  mat4 uView;
  mat4 uRot4;
  vec4 uRot0;      // xy, xz, xw, yz plane angles
  vec4 uRot1;      // yw, zw plane angles, tanHalfFov, time4D
  vec4 uMisc;      // x=shockGain y=bloom z=dpr w=quality
  vec4 uFocus;     // x=focusDist y=aperture z=grain w=vignette
  vec4 uShockMisc; // x=activeShocks y=radius z=thickness w=life
} U;
`;

/** std140 mirror of `PostState` (see POST_OFF in src/core/uniforms.ts). */
export const POST_BLOCK_GLSL = `
layout(std140) uniform PostBlock {
  vec4 pResolution; // x=w y=h z=1/w w=1/h  (destination)
  vec4 pTexel;      // xy=direction zw=source texel size
  vec4 pParams0;    // x=exposure y=bloom z=bloomRadius w=threshold
  vec4 pParams1;    // x=chroma y=grain z=vignette w=dof
  vec4 pParams2;    // x=volumetric y=time z=aspect w=frame
  vec4 pFocus;      // x=focusDepth y=focusRange z=maxCoC w=nearPlane
  vec4 pFar;        // x=farPlane y=sourceLod z=upsampleRadius w=bloomLevels
  vec4 pColor;      // x=hue y=saturation z=quality w=sceneLuma
} P;
`;

/** Constants, deterministic PCG-style hashing and respawn RNG. */
const GLSL_HASH = `
const float PI = 3.141592653589793;
const float TAU = 6.283185307179586;
const float INV_PI = 0.3183098861837907;

uint hash3(uvec3 p_in) {
  uvec3 p = p_in * 1664525u + 1013904223u;
  p.x = p.x + (p.y * p.z);
  p.y = p.y + (p.z * p.x);
  p.z = p.z + (p.x * p.y);
  p = p ^ (p >> 16u);
  p.x = p.x + (p.y * p.z);
  p.y = p.y + (p.z * p.x);
  p.z = p.z + (p.x * p.y);
  return p.x ^ p.y ^ p.z;
}

uint hash1(uint n) {
  return hash3(uvec3(n, n * 0x9E3779B9u, 0x85EBCA6Bu));
}

float rnd1(uint seed) {
  return float(hash1(seed)) * (1.0 / 4294967296.0);
}

float rndRange(uint seed, uint stream, float lo, float hi) {
  return lo + (hi - lo) * rnd1(seed * 747796405u + stream * 2891336453u + 1u);
}

/** Uniform point on S^3 (Marsaglia / Hopf decomposition), as in the WGSL prelude. */
vec4 rndDir4(uint seed, uint stream) {
  float a = rndRange(seed, stream + 0u, 0.0, TAU);
  float b = rndRange(seed, stream + 1u, -1.0, 1.0);
  float c = rndRange(seed, stream + 2u, 0.0, TAU);
  float s = sqrt(max(0.0, 1.0 - b * b));
  return vec4(s * cos(a), s * sin(a), s * cos(c), b);
}
`;

/** 4D rotation + the analytic force field (curl, Julia, Mandelbulb, Thomas). */
const GLSL_FIELD = `
// Six plane rotations in the fixed order R = Rzw . Ryw . Rxw . Ryz . Rxz . Rxy,
// byte-identical to rotationMatrix4() in src/core/math4d.ts.
vec4 rot4PlaneXY(vec4 v, float a) {
  float c = cos(a); float s = sin(a);
  return vec4(v.x * c - v.y * s, v.x * s + v.y * c, v.z, v.w);
}
vec4 rot4PlaneXZ(vec4 v, float a) {
  float c = cos(a); float s = sin(a);
  return vec4(v.x * c - v.z * s, v.y, v.x * s + v.z * c, v.w);
}
vec4 rot4PlaneXW(vec4 v, float a) {
  float c = cos(a); float s = sin(a);
  return vec4(v.x * c - v.w * s, v.y, v.z, v.x * s + v.w * c);
}
vec4 rot4PlaneYZ(vec4 v, float a) {
  float c = cos(a); float s = sin(a);
  return vec4(v.x, v.y * c - v.z * s, v.y * s + v.z * c, v.w);
}
vec4 rot4PlaneYW(vec4 v, float a) {
  float c = cos(a); float s = sin(a);
  return vec4(v.x, v.y * c - v.w * s, v.z, v.y * s + v.w * c);
}
vec4 rot4PlaneZW(vec4 v, float a) {
  float c = cos(a); float s = sin(a);
  return vec4(v.x, v.y, v.z * c - v.w * s, v.z * s + v.w * c);
}

vec4 rot4Apply(vec4 v) {
  vec4 o = rot4PlaneXY(v, U.uRot0.x);
  o = rot4PlaneXZ(o, U.uRot0.y);
  o = rot4PlaneXW(o, U.uRot0.z);
  o = rot4PlaneYZ(o, U.uRot0.w);
  o = rot4PlaneYW(o, U.uRot1.x);
  o = rot4PlaneZW(o, U.uRot1.y);
  return o;
}

// ---------------------------------------------------------------------------
// Analytic divergence-free 4D curl noise: 3 octaves of curl(A) with
// A = (s0, s1, s2, s0*s1*s2). Identical structure to curl4() in common.wgsl.ts.
// ---------------------------------------------------------------------------
vec4 curl4(vec4 p, float t, float gain, float warp) {
  vec4 curl = vec4(0.0);
  float amp = 1.0;
  float freq = 1.0;
  for (int k = 0; k < 3; k++) {
    float f = freq;
    float a = amp;
    vec4 w0 = vec4(0.83, 1.31, 0.57, 1.07) * f;
    vec4 w1 = vec4(1.21, 0.49, 1.37, 0.73) * f;
    vec4 w2 = vec4(0.61, 1.13, 0.91, 1.49) * f;
    float ph = t * (0.55 + 0.31 * f) + warp;

    float s0 = sin(dot(w0, p) + ph * 1.13);
    float s1 = sin(dot(w1, p) + ph * 0.87);
    float s2 = sin(dot(w2, p) + ph * 1.41);
    float c0 = cos(dot(w0, p) + ph * 1.13) * a;
    float c1 = cos(dot(w1, p) + ph * 0.87) * a;
    float c2 = cos(dot(w2, p) + ph * 1.41) * a;

    vec4 g = c0 * w0 + c1 * w1 + c2 * w2
           + (c0 * w0 * s1 * s2 + c1 * w1 * s0 * s2 + c2 * w2 * s0 * s1);

    curl = curl + vec4(
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

// 4D quaternion Julia accumulation. escape is the bail-out radius.
vec4 julia4(vec4 z_in, vec4 c, int iterations, float escape) {
  vec4 z = z_in;
  vec4 acc = vec4(0.0);
  for (int i = 0; i < 16; i++) {
    if (i >= iterations) { break; }
    float w = z.w;
    float x = z.x;
    float y = z.y;
    float zz = z.z;
    z = vec4(
      2.0 * (x * w - y * zz),
      2.0 * (y * w + x * zz),
      2.0 * (zz * w - x * y),
      w * w - x * x - y * y - zz * zz
    ) + c;
    acc = acc + z * exp(-0.85 * float(i));
    if (length(z) > escape) { break; }
  }
  return acc;
}

// 4D Mandelbulb triplex deformation: direction + magnitude pushing a particle
// along the iso-surface of the deformed bulb.
vec4 mandelbulb4(vec4 p, float power, float warp) {
  float r = length(p) + 1e-5;
  float theta = acos(clamp(p.w / r, -1.0, 1.0));
  float phi = atan(p.y, p.x);
  float psi = atan(p.z, sqrt(p.x * p.x + p.y * p.y) + 1e-6);
  float zr = pow(r, power);
  float nt = theta * power;
  float np = phi * power;
  float ns = psi * power;
  float st = sin(nt);
  vec4 q = vec4(
    zr * st * cos(np) * sin(ns),
    zr * st * sin(np) * sin(ns),
    zr * st * cos(ns),
    zr * cos(nt)
  );
  vec4 d = q - p;
  return d / (1.0 + dot(d, d)) * warp;
}

// Thomas' cyclically symmetric attractor, lifted to R4.
vec4 thomas4(vec4 q) {
  return vec4(
    sin(q.y) - 0.19 * q.x,
    sin(q.z) - 0.19 * q.y,
    sin(q.x) - 0.19 * q.z,
    sin(q.w) - 0.19 * q.w
  );
}
`;

/** Cosmology palette, hue rotation, filmic tonemap, sRGB and dither noise. */
const GLSL_COLOR = `
vec3 palette(float t) {
  vec3 a = vec3(0.62, 0.06, 0.94); // magenta
  vec3 b = vec3(0.05, 0.86, 0.98); // cyan
  vec3 c = vec3(1.00, 0.72, 0.16); // gold
  vec3 d = vec3(0.48, 0.16, 0.92); // deep purple
  float x = fract(t) * 4.0;
  if (x < 1.0) { return mix(a, b, smoothstep(0.0, 1.0, x)); }
  if (x < 2.0) { return mix(b, c, smoothstep(0.0, 1.0, x - 1.0)); }
  if (x < 3.0) { return mix(c, d, smoothstep(0.0, 1.0, x - 2.0)); }
  return mix(d, a, smoothstep(0.0, 1.0, x - 3.0));
}

/** Rodrigues rotation of a colour about the luma axis k = normalize(vec3(1)). */
vec3 hueRotate(vec3 rgb, float turns) {
  float a = turns * TAU;
  vec3 k = vec3(0.57735026919);
  float ca = cos(a);
  float sa = sin(a);
  return rgb * ca + cross(k, rgb) * sa + k * dot(k, rgb) * (1.0 - ca);
}

float luma(vec3 c) {
  return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

/** ACES filmic approximation (Stephen Hill's fit). */
vec3 acesFilm(vec3 x) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}

vec3 srgbEncode(vec3 c) {
  vec3 lo = c * 12.92;
  vec3 hi = 1.055 * pow(max(c, vec3(1e-5)), vec3(1.0 / 2.4)) - 0.055;
  return mix(hi, lo, step(c, vec3(0.0031308)));
}

/** Interleaved gradient noise: one ordered sample per pixel. */
float ign(vec2 p) {
  return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}
`;

/**
 * ---------------------------------------------------------------------------
 * Simulation pass — transform feedback, one step per frame.
 * Captures outPos / outVel / outMeta, in that order, into three separate
 * buffer objects (SEPARATE_ATTRIBS).
 * ---------------------------------------------------------------------------
 */
export const SIM_VS = `${HEAD}${SIM_BLOCK_GLSL}${GLSL_HASH}${GLSL_FIELD}
uniform vec4 uShocks[8];      // xyz = origin, w = strength
uniform float uShockAges[8];  // seconds since ignition
uniform int uJuliaIter;       // 6..12, from DeviceProfile.juliaIterations

layout(location = 0) in vec4 aPos;
layout(location = 1) in vec4 aVel;
layout(location = 2) in vec4 aMeta; // (age, lifespan, seed, phase)

out vec4 outPos;
out vec4 outVel;
out vec4 outMeta;

// Force gains: this block is the shared vocabulary with the WebGPU compute
// shader. Keep these numbers identical in both backends.
const float JULIA_GAIN  = 0.06;
const float BULB_GAIN   = 0.85;
const float THOMAS_GAIN = 0.15;
const float CURL_SCALE  = 0.09;

/** Expanding supernova shells (3D part of R4). */
vec3 shockForce(vec3 wp) {
  vec3 f = vec3(0.0);
  // The name is "nShock", not "active": GLSL ES 3.00 reserves the latter, and
  // using it makes the whole vertex shader fail to compile.
  int nShock = int(U.uShockMisc.x + 0.5);
  for (int i = 0; i < 8; i++) {
    if (i >= nShock) { break; }
    float strength = uShocks[i].w;
    if (strength <= 0.0) { continue; }
    float age = uShockAges[i];
    vec3 d = wp - uShocks[i].xyz;
    float r = length(d);
    vec3 dir = d / max(r, 1e-4);
    float shell = 0.6 + age * 2.3;
    float band = 0.32 + age * 0.5;
    float q = (r - shell) / band;
    float env = exp(-q * q) * strength * exp(-age * 1.6);
    f += dir * (env * 2.2);
    f += cross(dir, vec3(0.35, 0.6, 0.2)) * (env * 0.4);
  }
  return f;
}

void main() {
  float dt = clamp(U.uTime.y, 1.0 / 240.0, 1.0 / 30.0);

  vec4 p = aPos;
  vec4 v = aVel;
  float age = aMeta.x;
  float lifespan = max(aMeta.y, 0.05);
  uint seed = uint(max(aMeta.z, 0.0));
  float phase = aMeta.w;
  // meta.w doubles as a deterministic per-particle time scale.
  float spd = 0.75 + 0.5 * phase;

  // Sample the field in the rotated frame (six plane rotations first).
  vec4 rp = rot4Apply(p);

  vec4 acc = julia4(rp, U.uJulia, uJuliaIter, U.uAttract.y) * (JULIA_GAIN * (0.5 + U.uAttract.x));
  acc += mandelbulb4(rp, U.uAttract.z, U.uAttract.w) * BULB_GAIN;
  acc += thomas4(rp) * THOMAS_GAIN;
  acc += curl4(rp, U.uTime.x, U.uForce.x * (1.0 + U.uAudio.x * 0.8), U.uRot1.w) * CURL_SCALE;

  float rr = length(rp);
  vec4 radial = rp / max(rr, 1e-5);
  acc -= radial * (U.uForce.z / (0.6 + rr * rr));            // inverse-square pull
  acc -= rp * (U.uForce.w * smoothstep(2.0, 5.0, rr));       // confinement spring
  // Shocks act on the *rendered* (rotated) position so they line up with clicks.
  acc.xyz += shockForce(rp.xyz) * U.uMisc.x;

  v += acc * dt;
  v *= exp(-U.uForce.y * dt);
  p += v * (dt * spd);

  age += dt;
  if (age > lifespan) {
    uint ns = hash1(seed ^ (uint(U.uTime.z) * 747796405u));
    p = rndDir4(ns, 0u) * rndRange(ns, 4u, 0.9, 2.6);
    v = rndDir4(ns, 11u) * rndRange(ns, 14u, 0.05, 0.4);
    lifespan = rndRange(ns, 5u, 2.5, 7.5);
    phase = rndRange(ns, 7u, 0.0, 1.0);
    seed = ns & 0x00FFFFFFu;
    // age = 0 hides the birth behind the render-side fade-in.
    age = 0.0;
  }

  outPos = p;
  outVel = v;
  outMeta = vec4(age, lifespan, float(seed), phase);
  // Rasterization is discarded, but a defined position keeps every driver happy.
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
}
`;

/** Present only so the program is linkable everywhere; rasterization is discarded. */
export const SIM_FS = `${HEAD}
void main() {
}
`;

/**
 * ---------------------------------------------------------------------------
 * Particle billboards — instanced quads, expanded in view space.
 * ---------------------------------------------------------------------------
 */
export const PARTICLE_VS = `${HEAD}${SIM_BLOCK_GLSL}${GLSL_HASH}${GLSL_COLOR}
uniform float uMultiplicityPass; // 0 = the true particle, 1..3 = scatter copies
uniform float uAudioPulse;       // 0..~1 size pulse, decays after each beat

layout(location = 0) in vec4 aPos;
layout(location = 1) in vec4 aVel;
layout(location = 2) in vec4 aMeta;
layout(location = 3) in vec4 aColor; // static birth colour + energy

out vec2 vUv;
out vec3 vColor;

void main() {
  // Two triangles per billboard: (-1,-1) (1,-1) (-1,1) / (-1,1) (1,-1) (1,1)
  int vid = gl_VertexID;
  vec2 corner = vec2(
    (vid == 1 || vid == 4 || vid == 5) ? 1.0 : -1.0,
    (vid == 2 || vid == 3 || vid == 5) ? 1.0 : -1.0
  );

  float k = uMultiplicityPass;
  // Guard against a stale pass index if showCount shrank this frame.
  if (k >= max(U.uRender.z, 1.0)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vUv = vec2(0.0);
    vColor = vec3(0.0);
    return;
  }

  vec4 jitter = vec4(0.0);
  if (k > 0.0) {
    // Deterministic sub-particle scatter, hash-derived from seed + copy index.
    uint s = hash3(uvec3(
      uint(max(aMeta.z, 0.0)) + uint(k) * 7919u,
      uint(max(gl_InstanceID, 0)) + 31u,
      uint(k) + 17u
    ));
    jitter = vec4(rnd1(s), rnd1(s + 101u), rnd1(s + 202u), rnd1(s + 303u)) * 2.0 - 1.0;
    jitter = jitter * (0.012 * k);
  }

  // 4D rotation (the matrix is the composed plane rotations) then the
  // hyperplane slice: w maps to depth through the 4D fov term.
  vec4 rp = U.uRot4 * aPos + jitter;
  float wRel = rp.w - U.uSlice.x;
  vec3 world = vec3(rp.x, rp.y, rp.z + wRel * U.uSlice.y);

  vec4 viewPos = U.uView * vec4(world, 1.0);
  float depth = max(-viewPos.z, 0.02);

  // Depth cue: near = brighter/warmer/larger, far = darker/bluer/smaller.
  float focus = max(U.uCamera.w, 0.1);
  float near01 = 1.0 - smoothstep(focus * 0.55, focus * 1.25, depth);

  float lf = max(aMeta.y, 0.05);
  float lifeT = aMeta.x / lf;
  float lifeFade = smoothstep(0.0, 0.12, lifeT) * (1.0 - smoothstep(0.75, 1.0, lifeT));
  float nearFade = smoothstep(U.uSlice.z * 0.25, max(U.uSlice.z, 0.02), depth);

  float size = U.uRender.x
             * (0.014 + 0.022 * aColor.a)
             * (1.0 + uAudioPulse)
             * mix(0.70, 1.18, near01);

  // World size -> apparent pixel size -> back to a view-space half extent.
  float fovTan = max(U.uRot1.z, 0.05);
  float att = 1.0 / max(depth, 0.2);
  float pixelSize = size * U.uViewport.y * 0.5 / fovTan * att;
  float halfView = pixelSize * 2.0 * depth * fovTan / max(U.uViewport.y, 1.0);
  viewPos.xy += corner * halfView;

  gl_Position = U.uProj * viewPos;
  // [0,1] NDC depth (WebGPU convention) -> GL clip depth, so the depth texture
  // sampled by the post chain reads back as the same [0,1] value.
  gl_Position.z = 2.0 * gl_Position.z - gl_Position.w;

  // Palette phase: seed phase + slow drift + hue + audio mid band + 4th axis.
  float phase = aMeta.w
              + U.uTime.x * 0.03
              + U.uPalette.x
              + U.uAudio.y * 0.35
              + rp.w * 0.12
              + U.uRot1.w * 0.02;
  vec3 base = palette(phase);
  vec3 vivid = mix(vec3(luma(base)), base, clamp(U.uPalette.y, 0.0, 1.0));
  vec3 col = mix(base, vivid, clamp(U.uPalette.z, 0.0, 1.0));
  col = hueRotate(col, U.uPalette.x);
  // Static birth colour adds per-particle variation without leaving the palette.
  vec3 tint = aColor.rgb * (1.0 / max(luma(aColor.rgb), 0.35));
  col = col * mix(vec3(1.0), tint, 0.22) * (0.65 + 0.7 * aColor.a);

  vec3 deepTint = mix(col, vec3(0.35, 0.55, 1.0), 0.42) * 0.55;
  vec3 nearTint = mix(col, vec3(1.00, 0.80, 0.42), 0.30) * 1.18;
  col = mix(deepTint, nearTint, near01);

  vUv = corner;
  vColor = col * (U.uRender.y * lifeFade * nearFade * (1.0 + U.uAudio.x * 0.5));
}
`;

export const PARTICLE_FS = `${HEAD}
uniform float uLdr; // 1 when the HDR pipeline degraded to RGBA8

in vec2 vUv;
in vec3 vColor;
out vec4 fragColor;

void main() {
  float r = length(vUv);
  float t = max(1.0 - r, 0.0);
  // Soft circular core times a squared halo term; ~unit integral for energy=3.
  float alpha = (1.0 - smoothstep(0.35, 1.0, r)) * t * t * 3.0;
  vec3 c = vColor * alpha;
  if (uLdr > 0.5) {
    // RGBA8 accumulation: compress, then blend with ONE_MINUS_SRC_COLOR.
    c = c / (1.0 + c);
  }
  fragColor = vec4(c, 0.0);
}
`;

/**
 * Depth prepass — identical geometry, colour writes masked off. Only depth is
 * written, and only when depth of field needs a real depth buffer.
 */
export const DEPTH_VS = `${HEAD}${SIM_BLOCK_GLSL}
uniform float uMultiplicityPass;
uniform float uAudioPulse;

layout(location = 0) in vec4 aPos;
layout(location = 3) in vec4 aColor;

void main() {
  int vid = gl_VertexID;
  vec2 corner = vec2(
    (vid == 1 || vid == 4 || vid == 5) ? 1.0 : -1.0,
    (vid == 2 || vid == 3 || vid == 5) ? 1.0 : -1.0
  );

  if (uMultiplicityPass >= max(U.uRender.z, 1.0)) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    return;
  }

  vec4 rp = U.uRot4 * aPos;
  float wRel = rp.w - U.uSlice.x;
  vec3 world = vec3(rp.x, rp.y, rp.z + wRel * U.uSlice.y);

  vec4 viewPos = U.uView * vec4(world, 1.0);
  float depth = max(-viewPos.z, 0.02);
  float focus = max(U.uCamera.w, 0.1);
  float near01 = 1.0 - smoothstep(focus * 0.55, focus * 1.25, depth);

  float size = U.uRender.x
             * (0.014 + 0.022 * aColor.a)
             * (1.0 + uAudioPulse)
             * mix(0.70, 1.18, near01);

  float fovTan = max(U.uRot1.z, 0.05);
  float att = 1.0 / max(depth, 0.2);
  float pixelSize = size * U.uViewport.y * 0.5 / fovTan * att;
  float halfView = pixelSize * 2.0 * depth * fovTan / max(U.uViewport.y, 1.0);
  viewPos.xy += corner * halfView;

  gl_Position = U.uProj * viewPos;
  gl_Position.z = 2.0 * gl_Position.z - gl_Position.w;
}
`;

export const DEPTH_FS = `${HEAD}
void main() {
}
`;

/**
 * ---------------------------------------------------------------------------
 * Post chain. All intermediates are HDR linear; sRGB encoding happens once, in
 * COMPOSITE_FS, on the way to the 8-bit LDR buffer.
 * ---------------------------------------------------------------------------
 */

/** Attribute-less full-screen triangle (positions from gl_VertexID). */
export const FULLSCREEN_VS = `${HEAD}
out vec2 vUv;

void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

/** History feedback: previous HDR scene scaled by params.trails. */
/**
 * Frame-history feedback.
 *
 * The decay is luma-weighted, exactly as in the WebGPU fade pass: dim pixels are
 * released faster than bright ones. A flat multiplier feeds the whole frame back
 * at the same rate, so the accumulated haze reaches a steady state well above the
 * per-frame emission and the picture saturates — measured at a centre luma of 255
 * with trails on versus 171 with trails off, for identical particles.
 */
export const TRAIL_FS = `${HEAD}${SIM_BLOCK_GLSL}
uniform sampler2D uPrev;

in vec2 vUv;
out vec4 fragColor;

void main() {
  vec3 prev = texture(uPrev, vUv).rgb;
  float k = clamp(U.uRender.w, 0.0, 0.97);
  float lum = dot(prev, vec3(0.2126, 0.7152, 0.0722));
  float decay = k * mix(0.45, 1.0, smoothstep(0.0, 2.0, lum));
  // Ceiling on the history buffer. A feedback loop whose gain stays above the
  // emission it is fed will climb without bound, and because every later stage
  // (bloom, DOF, the grade) then sees a saturated buffer, *exposure stops having
  // any effect* — which is exactly the symptom this clamp removes.
  vec3 fed = min(prev * decay, vec3(8.0));
  fragColor = vec4(fed, 0.0);
}
`;

/** Soft-knee bright pass at half resolution. */
export const BRIGHT_FS = `${HEAD}${POST_BLOCK_GLSL}
uniform sampler2D uSrc;

in vec2 vUv;
out vec4 fragColor;

void main() {
  vec3 c = texture(uSrc, vUv).rgb;
  float br = max(c.r, max(c.g, c.b));
  float threshold = P.pParams0.w;
  float knee = 0.6;
  float soft = clamp(br - threshold + knee, 0.0, 2.0 * knee);
  soft = (soft * soft) / (4.0 * knee + 1e-4);
  float weight = max(soft, br - threshold) / max(br, 1e-4);
  fragColor = vec4(c * weight, 1.0);
}
`;

/** 13-tap COD-style downsample. */
export const BLOOM_DOWN_FS = `${HEAD}${POST_BLOCK_GLSL}
uniform sampler2D uSrc;

in vec2 vUv;
out vec4 fragColor;

void main() {
  vec2 t = P.pTexel.zw;

  vec3 a = texture(uSrc, vUv + t * vec2(-2.0, -2.0)).rgb;
  vec3 b = texture(uSrc, vUv + t * vec2( 0.0, -2.0)).rgb;
  vec3 c = texture(uSrc, vUv + t * vec2( 2.0, -2.0)).rgb;
  vec3 d = texture(uSrc, vUv + t * vec2(-2.0,  0.0)).rgb;
  vec3 e = texture(uSrc, vUv).rgb;
  vec3 f = texture(uSrc, vUv + t * vec2( 2.0,  0.0)).rgb;
  vec3 g = texture(uSrc, vUv + t * vec2(-2.0,  2.0)).rgb;
  vec3 h = texture(uSrc, vUv + t * vec2( 0.0,  2.0)).rgb;
  vec3 i = texture(uSrc, vUv + t * vec2( 2.0,  2.0)).rgb;
  vec3 j = texture(uSrc, vUv + t * vec2(-1.0, -1.0)).rgb;
  vec3 k = texture(uSrc, vUv + t * vec2( 1.0, -1.0)).rgb;
  vec3 l = texture(uSrc, vUv + t * vec2(-1.0,  1.0)).rgb;
  vec3 m = texture(uSrc, vUv + t * vec2( 1.0,  1.0)).rgb;

  vec3 o = e * 0.125;
  o += (a + c + g + i) * 0.03125;
  o += (b + d + f + h) * 0.0625;
  o += (j + k + l + m) * 0.125;
  fragColor = vec4(o, 1.0);
}
`;

/** 9-tap tent upsample, additively accumulated into the finer level. */
export const BLOOM_UP_FS = `${HEAD}${POST_BLOCK_GLSL}
uniform sampler2D uSrc;

in vec2 vUv;
out vec4 fragColor;

void main() {
  vec2 t = P.pTexel.zw * (1.0 + clamp(P.pParams0.z, 0.0, 1.0));

  vec3 o = texture(uSrc, vUv + t * vec2(-1.0, -1.0)).rgb * 1.0;
  o += texture(uSrc, vUv + t * vec2( 0.0, -1.0)).rgb * 2.0;
  o += texture(uSrc, vUv + t * vec2( 1.0, -1.0)).rgb * 1.0;
  o += texture(uSrc, vUv + t * vec2(-1.0,  0.0)).rgb * 2.0;
  o += texture(uSrc, vUv).rgb * 4.0;
  o += texture(uSrc, vUv + t * vec2( 1.0,  0.0)).rgb * 2.0;
  o += texture(uSrc, vUv + t * vec2(-1.0,  1.0)).rgb * 1.0;
  o += texture(uSrc, vUv + t * vec2( 0.0,  1.0)).rgb * 2.0;
  o += texture(uSrc, vUv + t * vec2( 1.0,  1.0)).rgb * 1.0;

  fragColor = vec4(o * (0.0625 * 0.6), 1.0);
}
`;

/** 24-tap radial god rays marching from the pixel toward the nebula core. */
/**
 * Volumetric light accumulation.
 *
 * Includes both uniform blocks on purpose: it projects the world origin with
 * `uProj`/`uView` from SimBlock to place the light, and reads the strength from
 * PostBlock. GLSL ES 3.00 gives the two blocks distinct names (`SimBlock`,
 * `PostBlock`) but a single instance name `U`, so referring to `U.pParams2.x`
 * after both are declared resolves against the merged member set, which compiles
 * cleanly — but it does mean neither block may be included twice.
 */
export const VOLUMETRIC_FS = `${HEAD}${SIM_BLOCK_GLSL}${POST_BLOCK_GLSL}
uniform sampler2D uSrc;

in vec2 vUv;
out vec4 fragColor;

void main() {
  vec4 clip = U.uProj * U.uView * vec4(0.0, 0.0, 0.0, 1.0);
  vec2 light = vec2(0.5);
  if (clip.w > 1e-4) {
    light = clip.xy / clip.w * 0.5 + 0.5;
  }
  light = clamp(light, vec2(-0.25), vec2(1.25));

  vec2 delta = (light - vUv) / 24.0;
  vec2 p = vUv;
  vec3 acc = vec3(0.0);
  float decay = 1.0;
  for (int i = 0; i < 24; i++) {
    p += delta;
    acc += texture(uSrc, clamp(p, vec2(0.0), vec2(1.0))).rgb * decay;
    decay *= 0.94;
  }
  acc *= (1.0 / 24.0) * 1.6;

  float mask = 1.0 - smoothstep(0.15, 1.2, length(vUv - light));
  fragColor = vec4(acc * clamp(P.pParams2.x, 0.0, 3.0) * mask, 1.0);
}
`;

/** Circle-of-confusion prefilter: half-res colour with CoC packed in alpha. */
export const DOF_PREPARE_FS = `${HEAD}${POST_BLOCK_GLSL}
uniform sampler2D uColor;
uniform sampler2D uDepth;

in vec2 vUv;
out vec4 fragColor;

void main() {
  vec2 t = P.pTexel.zw;
  vec3 c = texture(uColor, vUv + t * vec2(-0.5, -0.5)).rgb;
  c += texture(uColor, vUv + t * vec2( 0.5, -0.5)).rgb;
  c += texture(uColor, vUv + t * vec2(-0.5,  0.5)).rgb;
  c += texture(uColor, vUv + t * vec2( 0.5,  0.5)).rgb;
  c *= 0.25;

  // Depth is stored as [0,1] NDC depth (see the vertex shaders' z remap).
  float d = texture(uDepth, vUv).r;
  float nearPlane = P.pFocus.w;
  float farPlane = P.pFar.x;
  float viewZ = -(nearPlane * farPlane) / max(farPlane - d * (farPlane - nearPlane), 1e-4);
  float focus = max(P.pFocus.x, 1e-3);
  float coc = clamp(abs(viewZ - focus) / focus * max(P.pFocus.y, 0.0), 0.0, 1.0);

  fragColor = vec4(c, coc);
}
`;

/** 13-tap golden-angle bokeh gather, upsampled to full resolution. */
export const DOF_GATHER_FS = `${HEAD}${POST_BLOCK_GLSL}
uniform sampler2D uSrc;

in vec2 vUv;
out vec4 fragColor;

const float GOLDEN_ANGLE = 2.39996323;

void main() {
  vec4 center = texture(uSrc, vUv);
  float coc = center.a;
  vec2 radius = vec2(coc * P.pFocus.z) * P.pResolution.zw;

  vec3 sum = center.rgb;
  float wsum = 1.0;
  for (int i = 0; i < 12; i++) {
    float fi = float(i) + 0.5;
    float r = sqrt(fi / 12.0);
    float a = fi * GOLDEN_ANGLE;
    vec2 off = vec2(cos(a), sin(a)) * r * radius;
    vec4 s = texture(uSrc, vUv + off);
    // Depth-aware weighting keeps in-focus highlights from smearing.
    float w = 1.0 / (1.0 + abs(s.a - coc) * 8.0);
    sum += s.rgb * w;
    wsum += w;
  }
  fragColor = vec4(sum / wsum, coc);
}
`;

/** Final look: bloom + volumetric + DOF, chroma, ACES, grain, vignette, dither. */
/**
 * Final grade. Needs GLSL_COLOR for `luma`, `acesFilm`, `srgbEncode` and `ign`;
 * POST_BLOCK_GLSL alone supplies the uniforms, not the transfer functions.
 */
export const COMPOSITE_FS = `${HEAD}${POST_BLOCK_GLSL}${GLSL_HASH}${GLSL_COLOR}
uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform sampler2D uVolumetric;
uniform sampler2D uDof;
uniform float uAces; // 1 = HDR (ACES), 0 = already compressed by the LDR path

in vec2 vUv;
out vec4 fragColor;

vec3 sceneBox(vec2 uv) {
  vec2 t = P.pTexel.zw;
  vec3 a = texture(uScene, uv + t * vec2(-0.5, -0.5)).rgb;
  vec3 b = texture(uScene, uv + t * vec2( 0.5, -0.5)).rgb;
  vec3 c = texture(uScene, uv + t * vec2(-0.5,  0.5)).rgb;
  vec3 d = texture(uScene, uv + t * vec2( 0.5,  0.5)).rgb;
  return (a + b + c + d) * 0.25;
}

vec3 mixedAt(vec2 uv) {
  float dofMix = clamp(P.pParams1.w, 0.0, 1.0);
  return mix(sceneBox(uv), texture(uDof, uv).rgb, dofMix);
}

void main() {
  vec2 texel = P.pResolution.zw;
  float aspect = max(P.pParams2.z, 1e-3);
  float chroma = clamp(P.pParams1.x, 0.0, 2.0);

  vec3 col = mixedAt(vUv);

  // Diagnostic taps, enabled by setting pColor.y to 1..4. Used to find which
  // stage of the HDR chain was saturating; costs one uniform compare otherwise.
  if (P.pColor.y > 0.5) {
    if (P.pColor.y < 1.5) { fragColor = vec4(texture(uScene, vUv).rgb, 1.0); return; }
    if (P.pColor.y < 2.5) { fragColor = vec4(texture(uBloom, vUv).rgb, 1.0); return; }
    if (P.pColor.y < 3.5) { fragColor = vec4(texture(uVolumetric, vUv).rgb, 1.0); return; }
    fragColor = vec4(texture(uDof, vUv).rgb, 1.0);
    return;
  }

  if (chroma > 0.001) {
    // Per-channel radial offset growing with r^2 (lateral chromatic aberration).
    vec2 d = vUv - 0.5;
    vec2 off = d * dot(d, d) * chroma * 4.0 * texel;
    float dofMix = clamp(P.pParams1.w, 0.0, 1.0);
    col.r = mix(sceneBox(vUv + off).r, texture(uDof, vUv + off).r, dofMix);
    col.b = mix(sceneBox(vUv - off).b, texture(uDof, vUv - off).b, dofMix);
  }

  col += texture(uBloom, vUv).rgb * clamp(P.pParams0.y, 0.0, 4.0);
  col += texture(uVolumetric, vUv).rgb * clamp(P.pParams2.x, 0.0, 3.0);

  col *= max(P.pParams0.x, 0.0);                       // HDR exposure, pre-tonemap
  // Particle-count pre-exposure, matched to the WebGPU composite: additive
  // emission integrates with the number of emitters, so without a 1/sqrt(N)
  // normalisation a 1.5M-particle cloud clips to white (it did, at a mean luma of
  // 241/255) while a 200k one is nearly black. The count arrives in pColor.w
  // because this stage must not pull in SimBlock: GLSL ES 3.00 forbids two
  // interface blocks from sharing an instance name, and PostBlock is already U.
  col *= 0.62 / sqrt(max(P.pColor.w, 1.0));
  col = mix(clamp(col, 0.0, 1.0), acesFilm(col), uAces);

  // Luma-weighted film grain: shadows and highlights stay clean.
  float l = luma(col);
  float grainWeight = smoothstep(0.01, 0.18, l) * (1.0 - 0.5 * smoothstep(0.6, 1.0, l));
  float n = ign(gl_FragCoord.xy + vec2(P.pParams2.w * 13.7, P.pParams2.w * 7.3)) - 0.5;
  col += n * (clamp(P.pParams1.y, 0.0, 1.0) * 0.10) * grainWeight;

  // Smooth radial vignette, normalised so the corners reach 1.
  vec2 v = (vUv - 0.5) * vec2(aspect, 1.0);
  float vd = length(v) / (0.5 * length(vec2(aspect, 1.0)));
  col *= mix(1.0, smoothstep(1.25, 0.25, vd), clamp(P.pParams1.z, 0.0, 1.0));

  col += (ign(gl_FragCoord.xy) - 0.5) / 255.0;         // 1-LSB ordered dither
  fragColor = vec4(srgbEncode(max(col, vec3(0.0))), 1.0);
}
`;

/**
 * FXAA-lite: 5-tap luma edge detection, then a 4-tap directional refinement.
 * Runs on the LDR (already sRGB-encoded) image so edge detection is perceptual.
 */
export const FXAA_FS = `${HEAD}${POST_BLOCK_GLSL}
uniform sampler2D uSrc;

in vec2 vUv;
out vec4 fragColor;

float fxaaLuma(vec3 c) {
  return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
  vec2 texel = P.pTexel.zw;

  vec3 rgbM = texture(uSrc, vUv).rgb;
  vec3 rgbN = texture(uSrc, vUv + vec2(0.0, texel.y)).rgb;
  vec3 rgbS = texture(uSrc, vUv - vec2(0.0, texel.y)).rgb;
  vec3 rgbW = texture(uSrc, vUv - vec2(texel.x, 0.0)).rgb;
  vec3 rgbE = texture(uSrc, vUv + vec2(texel.x, 0.0)).rgb;

  float lM = fxaaLuma(rgbM);
  float lN = fxaaLuma(rgbN);
  float lS = fxaaLuma(rgbS);
  float lW = fxaaLuma(rgbW);
  float lE = fxaaLuma(rgbE);

  float lMin = min(lM, min(min(lN, lS), min(lW, lE)));
  float lMax = max(lM, max(max(lN, lS), max(lW, lE)));
  if (lMax - lMin < max(0.04, lMax * 0.125)) {
    fragColor = vec4(rgbM, 1.0);
    return;
  }

  vec2 dir = vec2(-((lN + lS) - 2.0 * lM), ((lE + lW) - 2.0 * lM));
  float reduce = max((lN + lS + lE + lW) * 0.03125, 1.0 / 128.0);
  float rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
  dir = clamp(dir * rcp, vec2(-8.0), vec2(8.0)) * texel;

  vec3 rgbA = 0.5 * (
    texture(uSrc, vUv + dir * (1.0 / 3.0 - 0.5)).rgb +
    texture(uSrc, vUv + dir * (2.0 / 3.0 - 0.5)).rgb
  );
  vec3 rgbB = rgbA * 0.5 + 0.25 * (
    texture(uSrc, vUv - dir * 0.5).rgb +
    texture(uSrc, vUv + dir * 0.5).rgb
  );

  float lB = fxaaLuma(rgbB);
  bool outside = (lB < lMin) || (lB > lMax);
  fragColor = vec4(outside ? rgbA : rgbB, 1.0);
}
`;

/**
 * Transform feedback capture order. MUST match the `bindBufferBase` order in
 * `WebGL2Renderer.stepSimulation()` (pos, vel, meta) and the declaration order of
 * `outPos`/`outVel`/`outMeta` in SIM_VS.
 */
export const SIM_TF_VARYINGS: readonly string[] = ['outPos', 'outVel', 'outMeta'];
