/**
 * Aetheria -- the hand-written HDR post-processing chain.
 *
 * Pipeline (every stage linear/HDR until the very last pixel):
 *
 *   scene(RGBA16F) --> brightPass --> bloom[0] --> downsample xN --> bloom[N]
 *                                                                      |
 *                                                                      |
 *   canvas <-- FXAA <-- composite <-- dofGather <-- upsample(tent) xN <-+
 *                        ^              ^
 *                        |              +-- dofDown: colour + max CoC, half res
 *                        +-- godray: 24 radial taps, additive into bloom[0]
 *
 * Design notes that matter for image quality:
 *   - Bloom uses a 13-tap Call-of-Duty style downsample with a Karis average on
 *     the first level, which removes the fireflies that would otherwise flicker
 *     as particles are born and die.
 *   - The bright pass uses a soft knee, so the transition into bloom is smooth
 *     instead of a hard threshold line drawn across the nebula.
 *   - DOF uses a golden-angle bokeh kernel on a half-resolution colour+CoC pair:
 *     a wide, round, noise-free blur for a fraction of a full-rate gather.
 *   - Dithering is applied *after* tone mapping with 1 LSB amplitude, which is
 *     exactly enough to kill Mach banding in the dark nebula without adding
 *     visible noise.
 */

import { WGSL_CONSTANTS, WGSL_CORE_MATH, WGSL_POST_UNIFORM } from './common.wgsl';

/** Fullscreen triangle from vertex_index -> no vertex buffer, no index buffer. */
export const FULLSCREEN_VS = /* wgsl */ `
struct FSOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
  @location(1) ndc : vec2f,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid : u32) -> FSOut {
  // Oversized triangle: (0,0) (2,0) (0,2) covers the clip square exactly once.
  var p = vec2f(f32((vid << 1u) & 2u), f32(vid & 2u));
  var out : FSOut;
  out.uv = p;
  out.ndc = p * 2.0 - 1.0;
  out.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  return out;
}
`;

/**
 * Single-input prelude (fade, bright pass, bloom downsample, FXAA).
 * WebGPU validates that a pipeline layout covers *exactly* the bindings a
 * shader declares, so each pass family gets its own minimal prelude + layout
 * instead of one fat shared layout.
 */
export const POST_PRELUDE = /* wgsl */ `
${WGSL_CONSTANTS}
${WGSL_POST_UNIFORM}
${WGSL_CORE_MATH}

struct FSOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
  @location(1) ndc : vec2f,
};

@group(0) @binding(0) var texA : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
@group(0) @binding(4) var<uniform> P : Post;

fn sampleA(uv : vec2f) -> vec3f { return textureSampleLevel(texA, samp, uv, 0.0).rgb; }

/** Karis average: weights samples by inverse luminance to kill fireflies. */
fn karis(w : vec4f) -> vec3f {
  let l = 1.0 / (1.0 + luma(w.rgb));
  return w.rgb * l;
}
`;

/**
 * Two-input prelude (bloom upsample, DOF gather). The sampler sits on binding 1
 * and both textures follow it, so every pass in this family has an identical,
 * unambiguous layout.
 */
export const POST_PRELUDE_AB = /* wgsl */ `
${WGSL_CONSTANTS}
${WGSL_POST_UNIFORM}
${WGSL_CORE_MATH}

struct FSOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
  @location(1) ndc : vec2f,
};

@group(0) @binding(0) var texA : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
@group(0) @binding(2) var texB : texture_2d<f32>;
@group(0) @binding(4) var<uniform> P : Post;

fn sampleA(uv : vec2f) -> vec3f { return textureSampleLevel(texA, samp, uv, 0.0).rgb; }
fn sampleB(uv : vec2f) -> vec3f { return textureSampleLevel(texB, samp, uv, 0.0).rgb; }

/** Karis average: weights samples by inverse luminance to kill fireflies. */
fn karis(w : vec4f) -> vec3f {
  let l = 1.0 / (1.0 + luma(w.rgb));
  return w.rgb * l;
}
`;

/**
 * Four-input prelude for the final composite: the scene (texA), the bloom
 * pyramid's finest level (texB), the volumetric streaks (texC) and the gathered
 * depth-of-field result (texD).
 *
 * All four exist as separate textures because WebGPU forbids a pass from
 * sampling a texture it renders into. Keeping the inputs distinct is what lets
 * every earlier stage write to its own target; the alternative — accumulating
 * bloom and god rays into one texture — would force one of them to read and
 * write the same resource.
 */
export const POST_PRELUDE_COMPOSITE = /* wgsl */ `
${WGSL_CONSTANTS}
${WGSL_POST_UNIFORM}
${WGSL_CORE_MATH}

struct FSOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
  @location(1) ndc : vec2f,
};

@group(0) @binding(0) var texA : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
@group(0) @binding(2) var texB : texture_2d<f32>;
@group(0) @binding(3) var texC : texture_2d<f32>;
@group(0) @binding(5) var texD : texture_2d<f32>;
@group(0) @binding(4) var<uniform> P : Post;

fn sampleA(uv : vec2f) -> vec3f { return textureSampleLevel(texA, samp, uv, 0.0).rgb; }
fn sampleB(uv : vec2f) -> vec3f { return textureSampleLevel(texB, samp, uv, 0.0).rgb; }
fn sampleC(uv : vec2f) -> vec3f { return textureSampleLevel(texC, samp, uv, 0.0).rgb; }
fn sampleD(uv : vec2f) -> vec3f { return textureSampleLevel(texD, samp, uv, 0.0).rgb; }
`;

/**
 * Volumetric-light prelude: the god-ray pass accumulates into the same bloom[0]
 * texture it samples (load-preserving additive blend), so ping-ponging is
 * unnecessary and it needs no extra target.
 */
export const POST_PRELUDE_GOD = /* wgsl */ `
${WGSL_CONSTANTS}
${WGSL_POST_UNIFORM}
${WGSL_CORE_MATH}

struct FSOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
  @location(1) ndc : vec2f,
};

@group(0) @binding(0) var texA : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
@group(0) @binding(2) var texB : texture_2d<f32>;
@group(0) @binding(4) var<uniform> P : Post;

fn sampleA(uv : vec2f) -> vec3f { return textureSampleLevel(texA, samp, uv, 0.0).rgb; }
fn sampleB(uv : vec2f) -> vec3f { return textureSampleLevel(texB, samp, uv, 0.0).rgb; }
fn lumaA(c : vec3f) -> f32 { return dot(c, vec3f(0.2126, 0.7152, 0.0722)); }
fn clampedA(uv : vec2f) -> vec3f { return textureSampleLevel(texA, samp, clamp(uv, vec2f(0.0), vec2f(1.0)), 0.0).rgb; }
`;

/** Depth-aware prelude for the DOF tile downsample: colour (filtered) + depth. */
export const POST_PRELUDE_ABD = /* wgsl */ `
${WGSL_CONSTANTS}
${WGSL_POST_UNIFORM}
${WGSL_CORE_MATH}

struct FSOut {
  @builtin(position) pos : vec4f,
  @location(0) uv : vec2f,
  @location(1) ndc : vec2f,
};

@group(0) @binding(0) var texA : texture_2d<f32>;
@group(0) @binding(1) var samp : sampler;
@group(0) @binding(2) var depthSamp : sampler_comparison;
@group(0) @binding(3) var texC : texture_depth_2d;
@group(0) @binding(4) var<uniform> P : Post;

fn sampleA(uv : vec2f) -> vec3f { return textureSampleLevel(texA, samp, uv, 0.0).rgb; }

/**
 * Depth fetch.
 *
 * A depth texture cannot be sampled with textureSampleLevel at all: WGSL only
 * offers textureSampleCompare / textureSampleCompareLevel (which return a
 * comparison result, not a depth) plus textureLoad. So the depth is read
 * directly by texel coordinate with textureLoad, which is exact and also avoids
 * any filtering across a depth discontinuity.
 */
fn sampleDepth(uv : vec2f) -> f32 {
  let dim = vec2f(textureDimensions(texC));
  let coord = vec2i(clamp(uv * dim, vec2f(0.0), dim - vec2f(1.0)));
  return textureLoad(texC, coord, 0);
}
`;

export const BRIGHT_PASS_WGSL = /* wgsl */ `
${POST_PRELUDE}

/**
 * Half-resolution bright pass with a soft knee. Soft knee matters: thousands of
 * individual particle sprites cross the threshold every frame, and a hard cut
 * turns that into visible twinkling at the bloom boundary.
 */
@fragment
fn fs_bright(in : FSOut) -> @location(0) vec4f {
  let texel = P.pTexel.zw;
  var sum = vec3f(0.0);
  // 4-tap box keeps the smallest bloom mip stable under motion.
  sum = sum + sampleA(in.uv + texel * vec2f(-0.5, -0.5));
  sum = sum + sampleA(in.uv + texel * vec2f( 0.5, -0.5));
  sum = sum + sampleA(in.uv + texel * vec2f(-0.5,  0.5));
  sum = sum + sampleA(in.uv + texel * vec2f( 0.5,  0.5));
  sum = sum * 0.25;

  let thr = P.pParams0.w;
  let knee = max(thr * 0.62, 1e-3);
  let br = max(max(sum.r, sum.g), sum.b);
  var soft = clamp(br - thr + knee, 0.0, 2.0 * knee);
  soft = soft * soft / (4.0 * knee);
  let contrib = max(soft, br - thr) / max(br, 1e-4);

  let c = sum * contrib;
  return vec4f(c, 1.0);
}
`;

export const BLOOM_DOWN_WGSL = /* wgsl */ `
${POST_PRELUDE}

/**
 * 13-tap downsample (the Jimenez / COD filter). The wider kernel removes the
 * pulsing that a naive bilinear chain produces when the source resolution
 * halves, which is very visible on thin particle filaments.
 */
fn downsample13(uv : vec2f, texel : vec2f) -> vec3f {
  var a = karis(textureSampleLevel(texA, samp, uv + texel * vec2f(-2.0, -2.0), 0.0));
  var b = karis(textureSampleLevel(texA, samp, uv + texel * vec2f( 0.0, -2.0), 0.0));
  var c = karis(textureSampleLevel(texA, samp, uv + texel * vec2f( 2.0, -2.0), 0.0));
  var d = karis(textureSampleLevel(texA, samp, uv + texel * vec2f(-1.0, -1.0), 0.0));
  var e = karis(textureSampleLevel(texA, samp, uv + texel * vec2f( 1.0, -1.0), 0.0));
  var f = karis(textureSampleLevel(texA, samp, uv + texel * vec2f(-2.0,  0.0), 0.0));
  var g = karis(textureSampleLevel(texA, samp, uv, 0.0));
  var h = karis(textureSampleLevel(texA, samp, uv + texel * vec2f( 2.0,  0.0), 0.0));
  var i = karis(textureSampleLevel(texA, samp, uv + texel * vec2f(-1.0,  1.0), 0.0));
  var j = karis(textureSampleLevel(texA, samp, uv + texel * vec2f( 1.0,  1.0), 0.0));
  var k = karis(textureSampleLevel(texA, samp, uv + texel * vec2f(-2.0,  2.0), 0.0));
  var l = karis(textureSampleLevel(texA, samp, uv + texel * vec2f( 0.0,  2.0), 0.0));
  var m = karis(textureSampleLevel(texA, samp, uv + texel * vec2f( 2.0,  2.0), 0.0));

  let group = (a + b + c + d + e + f + g + h + i + j + k + l + m) * (1.0 / 13.0);
  // Weighted ring emphasises the 3x3 core while still gathering a 5x5 support.
  let core = (d + e + i + j) * 0.125 + (b + f + h + l) * 0.0625 + g * 0.125;
  return mix(group, core * 1.6, 0.55);
}

@fragment
fn fs_down(in : FSOut) -> @location(0) vec4f {
  return vec4f(downsample13(in.uv, P.pTexel.zw), 1.0);
}
`;

export const BLOOM_UP_WGSL = /* wgsl */ `
${POST_PRELUDE_AB}

/**
 * 9-tap tent upsample.
 *
 * The kernel is parameterised: the exponent raises the tent weights to a power,
 * which narrows the effective footprint without changing the tap count or the
 * sample positions. The widest bloom mips span a large fraction of the frame, so
 * an un-narrowed tent turns a bright core into a halo that reaches the corners
 * and reads as a grey wash rather than a glow.
 */
fn tent9(uv : vec2f, texel : vec2f, k : f32) -> vec3f {
  var o = vec3f(0.0);
  let w: array<f32, 5> = array<f32, 5>(1.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0);
  let ox: array<f32, 5> = array<f32, 5>(-2.0, -1.0, 0.0, 1.0, 2.0);
  let p = clamp(k, 1.0, 6.0);
  var wsum = 0.0;
  for (var y = 0u; y < 5u; y = y + 1u) {
    for (var x = 0u; x < 5u; x = x + 1u) {
      let ww = pow(w[x] * w[y], p);
      let d = vec2f(ox[x], ox[y]);
      o = o + sampleA(uv + texel * d) * ww;
      wsum = wsum + ww;
    }
  }
  // Re-normalising keeps total energy constant, so the control changes the
  // halo's *size* rather than its brightness.
  return o * (1.0 / max(wsum, 1e-5));
}

@fragment
fn fs_up(in : FSOut) -> @location(0) vec4f {
  // pParams0.z is the bloom-radius control: 0 gives the widest halo, 1 the
  // tightest, and the kernel sharpens as it rises.
  let sharp = 1.0 + P.pParams0.z * 3.2;
  let hi = sampleB(in.uv);
  let lo = tent9(in.uv, P.pTexel.zw, sharp);
  let radius = mix(0.55, 1.05, P.pParams0.z);
  return vec4f(hi + lo * radius, 1.0);
}
`;

export const GODRAY_WGSL = /* wgsl */ `
${POST_PRELUDE_GOD}

/**
 * Volumetric light: 24-tap radial accumulation from the brightest region of the
 * bloom pyramid toward the frame centre. Implemented as a gather (not scatter)
 * so it is deterministic and does not need an atomic or ping-pong target.
 */
@fragment
fn fs_godray(in : FSOut) -> @location(0) vec4f {
  let strength = P.pParams2.x;
  if (strength <= 0.001) { return vec4f(0.0, 0.0, 0.0, 1.0); }

  let aspect = P.pParams2.z;
  let res = P.pResolution.xy;
  let uv = in.uv;
  // Aspect-corrected offset from the frame centre, in units where 1.0 is half
  // the frame's shorter axis. Without this, taps march orthogonally across the
  // wide axis on an ultrawide frame while barely moving vertically.
  let rel = (uv - vec2f(0.5)) * vec2f(max(aspect, 1.0) / max(min(aspect, 1.0), 1e-3), 1.0);
  // A dark frame edge stops the accumulator from dragging the bright core out to
  // the corners, where a radial march covers several screen widths.
  let edgeFade = smoothstep(2.4, 0.75, length(rel));
  if (edgeFade <= 0.001) { return vec4f(0.0, 0.0, 0.0, 1.0); }

  // The god-ray pass inspects the scene luminance to find where the light is
  // coming from, then streaks the *bloom* energy toward the frame centre. This is
  // what makes bright particle clusters read as light through dust.
  let guide = sampleB(uv);
  let sceneLum = lumaA(guide);
  let brightBias = smoothstep(0.12, 1.4, sceneLum);
  let dir = (vec2f(0.5) - uv) * (0.22 + 0.18 * brightBias);

  var acc = vec3f(0.0);
  var wsum = 0.0;
  let N = 24;
  var w = 1.0;
  for (var i = 0; i < N; i = i + 1) {
    let fi = f32(i) / f32(N);
    w = w * 0.94;
    let s = clamp(uv + dir * fi * 0.85, vec2f(0.0), vec2f(1.0));
    acc = acc + clampedA(s) * w;
    wsum = wsum + w;
  }
  acc = acc / max(wsum, 1e-4);
  // Only the brightest structures should streak; bias hard against the haze.
  let dens = max(acc.r, max(acc.g, acc.b));
  let streak = acc * smoothstep(0.25, 1.1, dens) * strength * 1.6;
  return vec4f(streak * (0.6 + 0.4 * brightBias) * edgeFade, 1.0);
}
`;

export const DOF_DOWN_WGSL = /* wgsl */ `
${POST_PRELUDE_ABD}

/**
 * Depth-aware downsample. We store the *maximum* circle of confusion in the
 * tile while averaging colour, which prevents a sharp foreground pixel from
 * being blended into a blurred background plateau (the classic "halo around
 * in-focus particles" artefact).
 */
@fragment
fn fs_dof_down(in : FSOut) -> @location(0) vec4f {
  let texel = P.pTexel.zw;
  let focus = max(P.pFocus.x, 1e-3);
  let aperture = P.pFocus.y;
  let maxCoc = P.pFocus.z;

  var col = vec3f(0.0);
  var maxC = 0.0;
  for (var y = -1; y <= 1; y = y + 1) {
    for (var x = -1; x <= 1; x = x + 1) {
      let off = vec2f(f32(x), f32(y));
      let uv = in.uv + off * texel;
      let c = sampleA(uv);
      let d = sampleDepth(uv);
      // Perceptual circle of confusion: zero at the focal distance, rising
      // with relative defocus in both directions.
      let coc = clamp(abs(d - focus) / focus * aperture * 4.0, 0.0, 1.0);
      col = col + c;
      maxC = max(maxC, coc);
    }
  }
  return vec4f(col / 9.0, clamp(maxC * maxCoc, 0.0, 1.0));
}
`;

export const DOF_GATHER_WGSL = /* wgsl */ `
${POST_PRELUDE_AB}

/**
 * Half-resolution 13-tap golden-angle bokeh gather.
 *
 * The gather runs at half resolution deliberately: at full resolution it was the
 * single most expensive stage in the chain (about 4.6 ms at 1600x900), because
 * every tap costs two texture reads — colour plus its circle of confusion — and
 * the blur disc is large enough that half resolution loses nothing visible.
 *
 * The kernel steps in full-resolution texels (pTexel.zw) so the disc radius stays
 * expressed in display pixels, while the colour and CoC inputs are the
 * half-resolution tile buffers. Sampling the full-res scene with a bilinear
 * filter at each disc tap is what keeps the bokeh smooth rather than blocky.
 */
const GA : f32 = 2.399963229728653;

@fragment
fn fs_dof(in : FSOut) -> @location(0) vec4f {
  let amount = P.pParams1.w;
  let centre = sampleA(in.uv);
  // texB carries the tile-max circle of confusion in its alpha channel.
  let coc = textureSampleLevel(texB, samp, in.uv, 0.0).a;
  if (amount <= 0.001 || coc <= 0.002) { return vec4f(centre, 1.0); }

  let texel = P.pTexel.zw;
  let radius = P.pFar.z;
  let px = radius * pow(coc, 0.85) * min(P.pResolution.y, 1440.0) * 0.045;

  var acc = centre;
  var wsum = 1.0;
  for (var i = 0; i < 12; i = i + 1) {
    let fi = f32(i);
    let r = sqrt((fi + 0.5) / 12.0);
    let a = fi * GA;
    let off = vec2f(cos(a), sin(a)) * r * px;
    let s = sampleA(in.uv + off * texel);
    // Weight neighbours by their own CoC so sharp detail never bleeds outward.
    let sc = textureSampleLevel(texB, samp, in.uv + off * texel, 0.0).a;
    let w = mix(0.35, 1.0, sc);
    acc = acc + s * w;
    wsum = wsum + w;
  }
  return vec4f(acc / wsum, 1.0);
}
`;

export const COMPOSITE_WGSL = /* wgsl */ `
${POST_PRELUDE_COMPOSITE}

/**
 * Final grade. Order of operations is deliberate:
 *   HDR combine -> chromatic aberration (before tone map, so highlights fringe
 *   like real glass) -> exposure -> ACES -> grain -> vignette -> dither.
 */
@fragment
fn fs_composite(in : FSOut) -> @location(0) vec4f {
  let res = P.pResolution.xy;
  let centre = vec2f(0.5);
  var uv = in.uv;

  // --- barrel distortion + chromatic aberration -------------------------
  // Both offsets grow with radius, and at the frame corners the combined
  // displacement reaches several percent of the image. Sampling outside 0..1
  // clamps to the edge texel, which paints a bright frame around the picture —
  // so the *result* is clamped rather than the offset, keeping the sampling
  // continuous while guaranteeing nothing reads past the border.
  let rel = uv - centre;
  let r2 = dot(rel, rel);
  let lens = 1.0 + r2 * 0.012 * P.pParams1.x * 0.5;
  uv = centre + rel * lens;

  let chroma = P.pParams1.x * 0.0022;
  let ca = rel * (r2 * chroma * 3.0 + chroma);
  let uvR = clamp(uv + ca, vec2f(0.0), vec2f(1.0));
  let uvG = clamp(uv, vec2f(0.0), vec2f(1.0));
  let uvB = clamp(uv - ca, vec2f(0.0), vec2f(1.0));
  let sceneR = sampleA(uvR);
  let sceneG = sampleA(uvG);
  let sceneB = sampleA(uvB);
  var scene = vec3f(sceneR.r, sceneG.g, sceneB.b);

  // --- depth of field: swap in the gathered bokeh where it exists --------
  let dofMix = P.pParams1.w;
  if (dofMix > 0.001) {
    let dof = sampleD(uvG);
    // Fade the blend near the frame edge, where the half-resolution gather is
    // least accurate; a soft transition hides the resolution mismatch.
    let edge = smoothstep(1.25, 0.35, length(rel * vec2f(res.x / max(res.y, 1.0), 1.0)));
    scene = mix(scene, dof, clamp(dofMix * edge, 0.0, 1.0));
  }

  // --- bloom and volumetric light, accumulated in HDR --------------------
  scene = scene + sampleB(clamp(uv, vec2f(0.0), vec2f(1.0))) * P.pParams0.y;
  scene = scene + sampleC(clamp(uv, vec2f(0.0), vec2f(1.0))) * P.pParams2.x;

  // --- film grain (luma-weighted: shadows stay clean) -------------------
  let grain = P.pParams1.y;
  if (grain > 0.001) {
    let n = ign(in.pos.xy + vec2f(P.pParams2.w * 17.137, P.pParams2.w * 7.331));
    let n2 = ign(in.pos.yx * 1.37 + vec2f(P.pParams2.w * 3.71, 0.0));
    let g = (n + n2 - 1.0) * grain * 0.085;
    scene = scene + g * (0.35 + 0.65 * smoothstep(0.0, 0.5, luma(scene)));
  }

  // --- exposure + tone map ---------------------------------------------
  // pColor.z carries a particle-count pre-exposure: a firefly-lit additive
  // renderer integrates proportional to the number of emitters, so without a
  // 1/sqrt(N) normalisation a 4M-particle cloud would be four times brighter
  // than a 250k one and every such frame would clip to white.
  let pre = P.pColor.z;
  let exposed = max(scene, vec3f(0.0)) * (P.pParams0.x * pre);
  var col = acesApprox(exposed);
  col = mix(col, acesFilm(exposed), 0.35);

  // --- hue/saturation master grade -------------------------------------
  col = hueRotate(col, P.pColor.x);
  let l = luma(col);
  col = mix(vec3f(l), col, P.pColor.y);

  // --- dynamic vignette -------------------------------------------------
  // Aspect-corrected radius. The floor is deliberately deep (0.06): a wide bloom
  // halo from the core spreads light past the frame corners, and only a strong
  // vignette keeps the picture's edge reading as empty space rather than a glow.
  let vig = P.pParams1.z;
  let vd = length(rel * vec2f(res.x / max(res.y, 1.0), 1.0));
  let v = 1.0 - smoothstep(0.35, 1.25, vd);
  col = col * mix(1.0, mix(0.06, 1.0, v), vig);

  // --- ordered + interleaved dither, 1 LSB, kills banding ---------------
  let d = (bayer8(vec2i(in.pos.xy)) - 0.5) + (ign(in.pos.xy * 1.618) - 0.5);
  col = col + d * (1.0 / 255.0);

  return vec4f(clamp(srgbEncode(col), vec3f(0.0), vec3f(1.0)), 1.0);
}
`;

export const FXAA_WGSL = /* wgsl */ `
${POST_PRELUDE}

/**
 * FXAA-lite: luma edge detection with directional 3-tap blur. Runs last, on the
 * already sRGB-encoded image, and is skipped entirely where no edge exists, so
 * it costs one extra texture read on the ~85% of pixels that are flat haze.
 */
fn lumaAt(uv : vec2f) -> f32 { return luma(textureSampleLevel(texA, samp, uv, 0.0).rgb); }

@fragment
fn fs_fxaa(in : FSOut) -> @location(0) vec4f {
  let texel = P.pTexel.xy;
  let rgbM = textureSampleLevel(texA, samp, in.uv, 0.0);
  let lM = luma(rgbM.rgb);

  let lNW = lumaAt(in.uv + vec2f(-1.0, -1.0) * texel);
  let lNE = lumaAt(in.uv + vec2f( 1.0, -1.0) * texel);
  let lSW = lumaAt(in.uv + vec2f(-1.0,  1.0) * texel);
  let lSE = lumaAt(in.uv + vec2f( 1.0,  1.0) * texel);

  let lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
  let lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
  let range = lMax - lMin;
  // Threshold scales with local contrast so dark sky is left completely alone.
  if (range < max(0.028, lMax * 0.115)) { return rgbM; }

  let dir = vec2f(-((lNW + lNE) - (lSW + lSE)), ((lNW + lSW) - (lNE + lSE)));
  let dirReduce = max((lNW + lNE + lSW + lSE) * 0.25 * 0.06, 1.0 / 128.0);
  let rcpMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
  let d = clamp(dir * rcpMin, vec2f(-8.0), vec2f(8.0)) * texel;

  let rgbA = (textureSampleLevel(texA, samp, in.uv + d * (1.0 / 3.0 - 0.5), 0.0).rgb
            + textureSampleLevel(texA, samp, in.uv + d * (2.0 / 3.0 - 0.5), 0.0).rgb) * 0.5;
  let rgbB = rgbA * 0.5 + (textureSampleLevel(texA, samp, in.uv + d * -0.5, 0.0).rgb
            + textureSampleLevel(texA, samp, in.uv + d * 0.5, 0.0).rgb) * 0.25;
  let lB = luma(rgbB);
  return vec4f(select(rgbB, rgbA, lB < lMin || lB > lMax), 1.0);
}
`;

/**
 * Trail feedback shader. Multiplies the previous frame by the persistence
 * factor, which is what turns a swarm of points into flowing silk: each frame
 * leaves a fraction of its energy behind, so fast particles draw their own
 * motion blur without a velocity buffer or a second geometry pass.
 *
 * The decay is applied in HDR linear space, so the trail of a 12.0-luminance
 * star stays visible far longer than the trail of a 0.3-luminance dust mote -> * exactly the behaviour of a real long-exposure photograph.
 */
export const FADE_WGSL = /* wgsl */ `
${POST_PRELUDE}

@fragment
fn fs_fade(in : FSOut) -> @location(0) vec4f {
  let c = sampleA(in.uv);
  // Decay slightly faster for dim pixels: prevents the nebula from turning into
  // a uniform grey wash at high persistence settings.
  let k = P.pParams1.x; // trails persistence, 0..0.97
  let lum = luma(c);
  let decay = k * mix(0.55, 1.0, smoothstep(0.0, 1.2, lum));
  return vec4f(c * decay, 1.0);
}
`;

