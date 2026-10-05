/**
 * Aetheria — particle rasterisation.
 *
 * Six vertices per instance expand an axis-aligned quad in *pixel* space, which
 * keeps the billboard circular regardless of aspect ratio or FOV. The centre
 * comes from the same storage buffers the compute kernel wrote, so the render
 * pass and the simulation can never disagree about where a particle is.
 *
 * 4D → 3D: the CPU uploads the composed SO(4) matrix; the vertex shader rotates
 * the 4-position, uses the resulting hyperplane coordinate as a perspective
 * divide, and mixes it into the depth output. That single term is what makes
 * the 4th dimension read as volume rather than a flat re-shuffle: particles
 * with large |w| defocus and dim, exactly like out-of-focus depth.
 */

import {
  WGSL_CONSTANTS,
  WGSL_UNIFORMS,
  WGSL_STORAGE_READONLY,
  WGSL_MATH,
  WGSL_CORE_MATH,
} from './common.wgsl';

export const PARTICLE_VERTEX_WGSL = /* wgsl */ `
${WGSL_CONSTANTS}
${WGSL_UNIFORMS}
${WGSL_STORAGE_READONLY}
${WGSL_MATH}

struct VSOut {
  @builtin(position) clip : vec4f,
  @location(0) @interpolate(linear) uv : vec2f,
  @location(1) @interpolate(linear) color : vec3f,
  @location(2) @interpolate(flat) sizePx : f32,
  @location(3) @interpolate(linear) energy : f32,
  @location(4) @interpolate(linear) spark : f32,
};

fn hsl2rgb(h : f32, s : f32, l : f32) -> vec3f {
  let c = (1.0 - abs(2.0 * l - 1.0)) * s;
  let hp = fract(h) * 6.0;
  let x = c * (1.0 - abs(fract(hp) - 1.0));
  var rgb = vec3f(0.0);
  let i = u32(hp) % 6u;
  if (i == 0u) { rgb = vec3f(c, x, 0.0); }
  else if (i == 1u) { rgb = vec3f(x, c, 0.0); }
  else if (i == 2u) { rgb = vec3f(0.0, c, x); }
  else if (i == 3u) { rgb = vec3f(0.0, x, c); }
  else if (i == 4u) { rgb = vec3f(x, 0.0, c); }
  else { rgb = vec3f(c, 0.0, x); }
  return rgb + (l - 0.5 * c);
}

@vertex
fn vs_main(
  @builtin(vertex_index) vid : u32,
  @builtin(instance_index) inst : u32
) -> VSOut {
  let corner = vec2f(f32(vid & 1u), f32((vid >> 1u) & 1u)) * 2.0 - 1.0;
  let sub = f32(vid >> 2u); // 0 for the primary image, 0..3 for the scatter copies

  let particleCount = u32(U.uTime.w);
  let primary = inst % max(particleCount, 1u);

  // ---------------------------------------------------------------------
  // Multiplicity: draw the same simulated particle several times with a
  // deterministic sub-offset. This is how "10M displayed particles" is
  // honoured without paying 10M×88 bytes of simulation memory — the copies are
  // visually indistinguishable from extra particles but cost only vertices.
  // ---------------------------------------------------------------------
  let multipl = max(U.uRender.z, 1.0);
  let copy = min(floor(f32(inst) / f32(max(particleCount, 1u))), multipl - 1.0);
  var jitter = vec3f(0.0);
  if (copy > 0.0 || sub > 0.0) {
    let h = hash1(inst * 2654435761u + vid * 40503u + 7u);
    let r = 0.028 * (0.4 + 0.6 * f32(copy));
    jitter = vec3f(
      rndRange(h, 0u, -r, r),
      rndRange(h, 1u, -r, r),
      rndRange(h, 2u, -r, r)
    );
  }

  let p4 = arrPingP[primary];
  let v4 = arrPingV[primary];
  let c4 = arrPingC[primary];
  let m4 = arrPingM[primary];

  let age = m4.x;
  let lifespan = max(m4.y, 1e-3);
  let lifeT = clamp(age / lifespan, 0.0, 1.0);
  // Symmetric fade: no particle is ever born or killed at full brightness.
  let lifeFade = smoothstep(0.0, 0.085, lifeT) * (1.0 - smoothstep(0.78, 1.0, lifeT));

  // ---------------------------------------------------------------------
  // 4D transform + hyperplane slice + perspective divide from w.
  // ---------------------------------------------------------------------
  var q = rot4_apply(vec4f(p4.xyz + jitter, p4.w + U.uSlice.x));
  let wPersp = 1.0 / (1.0 + q.w * U.uSlice.y);
  q = vec4f(q.xyz * wPersp, q.w);

  var viewPos = U.uView * vec4f(q.xyz, 1.0);
  // Fold the hyperplane coordinate into the depth channel so DOF and fog both
  // treat 4D distance as distance.
  let wDepth = q.w * 0.34 * wPersp;
  viewPos = vec4f(viewPos.xyz, viewPos.w - wDepth);
  let clip = U.uProj * viewPos;

  // Defensive: never let a bad particle produce NaN geometry.
  var safeClip = clip;
  if (!(abs(clip.w) > 1e-6)) { safeClip = vec4f(0.0, 0.0, 2.0, 1.0); }

  let depth = max(-viewPos.z, 1e-3);

  // ---------------------------------------------------------------------
  // Size: perspective attenuation, 4D proximity, lifecycle and audio pulse.
  // ---------------------------------------------------------------------
  let baseSize = U.uRender.x * (0.85 + 0.9 * c4.w) * 3.6;
  let pixelScale = U.uViewport.y * 0.5 * U.uRot1.z;
  let sizeWorld = baseSize / (0.85 + 0.35 * depth);
  var sizePx = sizeWorld * pixelScale / max(safeClip.w, 0.05);
  sizePx = clamp(sizePx, 0.55, 90.0);
  sizePx = sizePx * mix(0.72, 1.0, lifeFade) * (1.0 + U.uAudio.x * 0.25 + U.uAudio.w * 0.22);

  // Persistent glow for a handful of "hero" stars, so the frame never looks
  // uniformly granular — it reads as a photograph with real highlights.
  let hero = smoothstep(0.86, 1.0, c4.w);
  sizePx = sizePx * (1.0 + hero * 2.4);

  let px = sizePx * 0.5 * corner * U.uViewport.zw;
  let outPos = vec4f(safeClip.xy / safeClip.w + px, safeClip.z, safeClip.w);

  // ---------------------------------------------------------------------
  // Colour: palette phase (seeded + hue + audio mid), then temperature grade
  // by depth (near warm/gold, far cool/blue), then luma shaping.
  // ---------------------------------------------------------------------
  let phase = m4.w + U.uPalette.x + U.uAudio.y * 0.30 + q.w * 0.10;
  let speedT = clamp(length(v4.xyz) * 1.4, 0.0, 1.0);

  var col = palette(phase) * mix(0.55, 1.35, speedT);
  // Violet-shift the coldest particles for extra palette depth.
  col = col * mix(vec3f(1.0), vec3f(1.06, 0.94, 1.12), 1.0 - lifeFade);

  let nearMix = smoothstep(6.5, 1.2, depth);
  col = mix(col * vec3f(0.48, 0.62, 1.12), col * vec3f(1.14, 1.02, 0.78), nearMix);
  col = col * mix(1.18, 0.72, smoothstep(3.0, 22.0, depth));

  // Saturation control keeps the master hue rotation from going pastel.
  let l = luma(col);
  col = mix(vec3f(l), col, U.uMisc2.z) * (1.0 + hero * 1.6);

  var out : VSOut;
  out.clip = outPos;
  out.uv = corner;
  out.color = col * c4.xyz * 2.2;
  out.sizePx = sizePx;
  out.energy = c4.w * lifeFade * U.uRender.y;
  out.spark = hero * U.uAudio.z;
  return out;
}
`;

export const PARTICLE_FRAGMENT_WGSL = /* wgsl */ `
${WGSL_CONSTANTS}
${WGSL_CORE_MATH}

struct VSOut {
  @builtin(position) clip : vec4f,
  @location(0) @interpolate(linear) uv : vec2f,
  @location(1) @interpolate(linear) color : vec3f,
  @location(2) @interpolate(flat) sizePx : f32,
  @location(3) @interpolate(linear) energy : f32,
  @location(4) @interpolate(linear) spark : f32,
};

/**
 * Soft, slightly anamorphic sprite. The core is a tight gaussian for a crisp
 * centre (so the eye reads individual stars) and the halo is a wide falloff
 * that blooms into its neighbours. Both are analytic — no texture lookups — so
 * the pass is bandwidth-bound on the storage reads only.
 */
@fragment
fn fs_main(in : VSOut) -> @location(0) vec4f {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }

  // Core-to-halo balance decides whether the nebula reads as glowing gas or as a
  // washed-out haze. With a halo as strong as the core, a million overlapping
  // sprites sum to an evenly lit rectangle; biasing the energy into a tight core
  // leaves dark space between filaments, so the additive pass shows structure
  // instead of a uniform floor.
  let core = exp(-r2 * 12.0);
  let halo = pow(max(1.0 - sqrt(r2), 0.0), 3.4) * 0.22;
  // Cross-shaped diffraction spikes on the brightest stars only.
  let ax = in.uv.x * in.uv.x;
  let ay = in.uv.y * in.uv.y;
  let spike = (exp(-ay * 220.0) + exp(-ax * 220.0)) * pow(max(1.0 - r2, 0.0), 3.0) * in.spark * 0.9;

  let alpha = (core + halo + spike) * in.energy;
  if (alpha <= 0.0009) { discard; }

  // A hint of blue at the sprite rim mimics lens dispersion on real optics.
  let rim = smoothstep(0.35, 1.0, sqrt(r2));
  let tint = mix(in.color, in.color * vec3f(0.72, 0.86, 1.25), rim * 0.55);

  return vec4f(tint * alpha, alpha);
}
`;
