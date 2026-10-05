/**
 * Aetheria — particle lifecycle shader fragment.
 *
 * Extracted so the seed kernel and the per-frame simulation kernel share one
 * definition of "what a particle's birth looks like". Duplicating it would let
 * the two drift, and a drifting birth function shows up as a visible flicker
 * whenever a particle crosses its lifespan boundary.
 *
 * It is a plain string rather than a module because it depends on the hash,
 * palette and constant helpers in `common.wgsl.ts`, and on the uniform block.
 */

export const WGSL_LIFECYCLE = /* wgsl */ `
struct Birth { p : vec4f, v : vec4f, c : vec4f, m : vec4f };

/**
 * Deterministic respawn. The epoch is derived from the particle's generation so
 * successive lives of the same particle never repeat, yet the whole universe is
 * reproducible from (index, epoch) alone: no per-particle RNG state, no
 * allocation, no readback.
 *
 * Layout of the returned record:
 *   p.xyz  position in R3 (the shell)   p.w   position along the W axis
 *   v.xyz  tangential + radial velocity v.w   drift along W
 *   c.rgb  linear HDR colour            c.w   per-particle emission weight
 *   m.x    age (always 0)               m.y   lifespan in seconds
 *   m.z    seed in 0..1                 m.w   palette phase
 */
fn birth(seed : u32, epoch : u32) -> Birth {
  var b : Birth;
  let s = seed * 2654435761u + epoch * 40503u + 1u;

  // Uniform direction on S^3 (Marsaglia), flattened toward the W axis so the
  // cloud reads as a lens rather than a hypersphere when it is sliced.
  let w = rndRange(s, 0u, -1.0, 1.0);
  let t = sqrt(max(0.0, 1.0 - w * w));
  let angA = rndRange(s, 1u, 0.0, TAU);
  let angB = rndRange(s, 2u, 0.0, TAU);
  let dir = vec4f(t * cos(angA), t * sin(angA), t * sin(angB) * 0.92, w * 0.86);

  // Shell radius. The exponent shapes the radial density profile: below 1 the
  // population piles up at the outer edge, at 1 it is uniform through the volume,
  // and above 1 it concentrates toward the core. A concentrated profile is what
  // gives a bright nucleus with a sparse halo, instead of a uniformly lit hull.
  let u = rnd1(s * 7u + 13u);
  let r = mix(0.32, 2.9, pow(u, 1.85));
  let pos = normalize(dir + vec4f(1e-4)) * r;

  // Tangential birth velocity keeps the cloud spinning instead of collapsing.
  let tangent = normalize(vec4f(-pos.y, pos.x, -pos.w, pos.z) + vec4f(1e-5));
  let speed = rndRange(s, 3u, 0.05, 0.22);
  let radial = rndRange(s, 4u, -0.05, 0.12);

  let phase = rndRange(s, 5u, 0.0, 1.0);
  var col = palette(phase);
  // Emissive weighting: most particles are cool dust, a minority are hot stars.
  // The sixth power is what produces a sparse, believable population of
  // highlights instead of a uniformly glowing fog.
  let hot = pow(rnd1(s * 31u + 7u), 6.0);
  col = mix(col * 0.55, col * 3.4 + vec3f(0.35, 0.28, 0.16), hot);

  b.p = vec4f(pos.x, pos.y, pos.z, rndRange(s, 6u, -1.2, 1.2));
  // Velocity is assembled from a vec3 plus a scalar: WGSL has no (vec4, f32)
  // constructor, so the W component cannot simply be appended.
  b.v = vec4f(
    tangent.xyz * speed + normalize(pos.xyz + vec3f(1e-4)) * radial,
    rndRange(s, 7u, -0.08, 0.08)
  );
  b.c = vec4f(col, mix(0.35, 1.0, pow(rnd1(s * 17u + 3u), 2.2)));
  b.m = vec4f(0.0, mix(3.2, 9.5, pow(rnd1(s * 23u + 11u), 1.35)), f32(seed) * (1.0 / 4294967296.0), phase);
  return b;
}
`;
