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

@vertex
fn vs_main(
  @builtin(vertex_index) vid : u32,
  @builtin(instance_index) inst : u32
) -> VSOut {
  let corner = vec2f(f32(vid & 1u), f32((vid >> 1u) & 1u)) * 2.0 - 1.0;
  let sub = f32(vid >> 2u); // 0 for the primary image, 0..3 for the scatter copies

  // ---------------------------------------------------------------------
  // Instance -> particle mapping.
  //
  // The obvious form, an integer modulo against the count, is the single most
  // expensive thing this shader can do: unsigned integer modulo is not a hardware
  // op on current GPUs, so it expands into a reciprocal-multiply sequence with a
  // long dependency chain, executed once per vertex - millions of times a frame.
  //
  // It is replaced by a float division. The count is well under 2^24 and the
  // instance index is under 2^32, so the quotient is exact in f32, and
  // (inst - count * floor(inst / count)) recovers the remainder exactly while
  // emitting no long-latency integer division at all.
  //
  // This keeps the count an arbitrary number rather than a power of two, which
  // matters: the frame budget has to be honoured at whatever value it computes,
  // and rounding to the next power of two could overshoot it by nearly 2x.
  //
  // Copies come first and primary particles last, so the drawn range is a whole
  // number of copies and every copy covers every live particle.
  let count = max(U.uMisc.w, 1.0);
  let copy = floor(f32(inst) / count);
  let primary = u32(f32(inst) - count * copy);
  let copies = max(U.uRender.z, 1.0);
  // Clamp so a draw range larger than count * copies cannot index out of bounds.
  var jitter = vec3f(0.0);
  let copyClamped = min(copy, copies - 1.0);
  if (copyClamped > 0.0 || sub > 0.0) {
    let h = hash1(inst * 2654435761u + vid * 40503u + 7u);
    let r = 0.028 * (0.4 + 0.6 * copyClamped);
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

  // Ablation hook, enabled only by the profiling tools: uMisc2.w scales the sprite
  // radius. Crushing it to a pixel or two removes almost all fill work while
  // leaving the per-instance vertex work untouched, which is what separates a
  // fill-bound pass from a vertex-bound one.
  sizePx = sizePx * max(U.uMisc2.w, 0.01);

  // The quad is sized to what the sprite actually draws, not to a fixed multiple
  // of it. The core falloff is exp(-12 r^2), so the radius at which it becomes
  // negligible is sqrt(-ln(T)/12); a quad of that extent already covers every
  // fragment that survives the alpha test. Sizing it generously instead spends
  // rasteriser bandwidth on fragments that are discarded a few instructions later.
  //
  // The visible radius is derived from the same falloff the fragment stage uses:
  // exp(-12 r^2) = T  =>  r = sqrt(-ln(T)/12). A halo term extends it slightly
  // for the few bright sprites where it is noticeable.
  let coreRadius = sqrt(0.3838); // sqrt(-ln(0.01)/12)
  let haloExtent = 1.0 + 0.35 * (1.0 - exp(-sizePx * 0.5));
  let quadScale = coreRadius * haloExtent;

  let px = sizePx * quadScale * corner * U.uViewport.zw;
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
  // The quad now matches the visible sprite, so this test rejects almost nothing;
  // it stays as a guard for the thin ring where the halo has faded out.
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
