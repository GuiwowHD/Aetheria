/**
 * Aetheria — WGSL particle universe.
 *
 * One workgroup per 64 particles; each invocation evolves a single particle in
 * R⁴ and writes it to the opposite ping-pong buffer. All state lives in four
 * vec4 arrays (position, velocity, colour, lifecycle) so the whole step is
 * coalesced 16-byte traffic.
 *
 * Force model, in evaluation order:
 *   1. 4D quaternion-Julia attractor (the "fractal skeleton")
 *   2. 4D Mandelbulb triplex deformation (topology driven by fractalDim)
 *   3. Thomas-attractor swirl (large-scale rotation)
 *   4. analytic divergence-free 4D curl noise (turbulence / silk)
 *   5. inverse-square gravity + confinement shell (bounded nebula)
 *   6. supernova shockwaves (up to MAX_SHOCKS expanding shells)
 *   7. audio: low → turbulence gain, mid → fractal warp, high → sparkle jitter
 *
 * Stability comes from three guards: a semi-implicit integrator, a
 * tanh-based soft velocity limiter (never a hard clamp, so motion stays smooth
 * at the boundary), and force saturation on the fractal terms which may
 * legitimately blow up inside the Julia set.
 */

import { WGSL_CONSTANTS, WGSL_UNIFORMS, WGSL_STORAGE_BINDINGS, WGSL_MATH } from './common.wgsl';
import { WGSL_LIFECYCLE } from './lifecycle.wgsl';

export const MAX_SHOCKS = 8;

export const PARTICLE_WORKGROUP = 64;

export const SIMULATE_WGSL = /* wgsl */ `
${WGSL_CONSTANTS}
${WGSL_UNIFORMS}
${WGSL_STORAGE_BINDINGS}
${WGSL_MATH}
${WGSL_LIFECYCLE}

const MAX_SHOCKS : u32 = ${MAX_SHOCKS}u;

fn softLimitAccel(a : vec3f, limit : f32) -> vec3f {
  let m = length(a);
  if (m < 1e-6) { return a; }
  // tanh saturation: linear near zero (so small forces are exact), asymptotic
  // at the limit (so nothing explodes). Branch-free and C1-continuous.
  return a * (limit * tanh(m / limit) / m);
}

fn softLimitVec4(a : vec4f, limit : f32) -> vec4f {
  let m = length(a);
  if (m < 1e-6) { return a; }
  return a * (limit * tanh(m / limit) / m);
}

@compute @workgroup_size(${PARTICLE_WORKGROUP})
fn simulate(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  let count = u32(U.uTime.w);
  if (i >= count) { return; }

  let dt = clamp(U.uTime.y, 1.0 / 240.0, 1.0 / 30.0);
  let t = U.uTime.x;

  var pos = arrPingP[i];
  var vel = arrPingV[i];
  var col = arrPingC[i];
  var md = arrPingM[i];

  // ---------------------------------------------------------------------
  // Lifecycle. Respawn is metered: each particle is allowed to transition
  // only once per ~0.25 s slice of the global clock, which staggers births
  // into a continuous drizzle instead of a synchronised flash.
  // ---------------------------------------------------------------------
  var age = md.x + dt;
  md.y = md.y + dt * 1e-4; // tiny lifespan drift keeps lifetimes coprime-ish

  if (age > md.y || age < 0.0) {
    let epoch = u32(t * 3.7) + u32(md.x * 1000.0);
    if (fract(abs(md.z) * 137.0 + t * 0.37) < 0.16) {
      let b = birth(u32(md.z * 4294967295.0) ^ (i * 747796405u), epoch);
      // Cross-fade the new life in by keeping a fraction of the old velocity,
      // so a respawn reads as a smooth drift rather than a teleport.
      pos = mix(pos, b.p, 0.86);
      vel = mix(vel, b.v, 0.86);
      col = mix(col, b.c, 0.9);
      age = 0.0;
      md = vec4f(0.0, b.m.y, b.m.z, b.m.w);
      // Seed the colour carried in col.w directly.
      col.w = b.c.w;
    } else {
      // Not this particle's turn: hold it in a long tail so it does not vanish.
      age = md.y * 0.02;
      md.y = md.y * 1.6 + 1.0;
    }
  }

  // ---------------------------------------------------------------------
  // Shared field frame: one rotation, one radius, reused by every term.
  // ---------------------------------------------------------------------
  let rp = rot4_apply(pos);
  let rp3 = rp.xyz;
  let r3 = length(rp3) + 1e-5;

  let low = U.uAudio.x;
  let mid = U.uAudio.y;
  let high = U.uAudio.z;
  let beat = U.uAudio.w;

  // ---------------------------------------------------------------------
  // 1 + 2. Fractal attractors. fractalDim drives both the bulb exponent and
  // the Julia iteration depth, which is why that slider changes topology.
  // ---------------------------------------------------------------------
  let dim = clamp(U.uAttract.x, 0.0, 1.0);
  let iters = i32(clamp(4.0 + dim * 12.0 + mid * 3.0, 4.0, 16.0));
  let innerScale = mix(0.72, 0.36, dim);

  let jf = julia4(rp * innerScale, U.uJulia, iters);
  let bulbPower = mix(2.35, 6.5, dim);
  let bulbWarp = mix(0.22, 0.85, dim) * (1.0 + mid * 0.6);
  let bf = mandelbulb4(rp * 0.55, bulbPower, bulbWarp);

  let fractalForce = softLimitAccel(jf.xyz, 26.0) * (0.30 + dim * 0.42) + softLimitAccel(bf.xyz, 14.0) * 0.5;

  // ---------------------------------------------------------------------
  // 3. Thomas attractor: a bounded chaotic flow that gives the cloud its
  // large-scale spiral arms.
  // ---------------------------------------------------------------------
  let thomas = vec3f(sin(rp3.y) - 0.19 * rp3.x, sin(rp3.z) - 0.19 * rp3.y, sin(rp3.x) - 0.19 * rp3.z);
  let thomasForce = thomas * 0.14;

  // ---------------------------------------------------------------------
  // 4. Divergence-free curl noise. Low frequencies inflate its gain, so bass
  // makes the nebula breathe and churn.
  // ---------------------------------------------------------------------
  let curlGain = U.uForce.x * (1.0 + low * 1.35 + beat * 0.35);
  let curlForce = curl4(rp, t * 0.7, curlGain * 0.62, U.uAttract.w + mid * 0.9).xyz;

  // ---------------------------------------------------------------------
  // 5. Radial terms: a core attractor plus a bounding shell.
  //
  // The attractive term is what makes the nebula *read* as a nebula. With only a
  // weak pull, particles coast out to the confinement radius at roughly uniform
  // density, and because the pass is additive that produces an evenly glowing
  // rectangle rather than a bright core inside a dark void. A power-law pull
  // keeps the mass concentrated near the centre and lets a sparse halo fall off
  // outward, which is the density profile the eye expects.
  // ---------------------------------------------------------------------
  let rn = rp3 / r3;
  // Pull grows toward the centre (r^-1.6) instead of falling off, which is what
  // concentrates the population without needing an unphysically large gain.
  let gravity = -rn * (U.uForce.z / (0.35 + r3 * r3 * 1.1) * (2.2 / (0.55 + r3)));
  let confine = -rp3 * (U.uForce.w * smoothstep(1.5, 3.6, r3));

  var acc = fractalForce + thomasForce + curlForce + gravity + confine;

  // ---------------------------------------------------------------------
  // 6. Supernova shockwaves: N expanding, decaying shells.
  // ---------------------------------------------------------------------
  let nShock = u32(U.uShockMisc.x);
  for (var s = 0u; s < MAX_SHOCKS; s = s + 1u) {
    if (s >= nShock) { break; }
    let ev = shocks[s];
    let shockAge = ev.w;
    let strength = shocks[s + MAX_SHOCKS].x; // second half holds the strengths
    if (strength <= 0.0 || shockAge < 0.0) { continue; }
    let o = ev.xyz;
    let d = rp3 - o;
    let dist = length(d) + 1e-5;
    let dir = d / dist;
    let shell = 0.35 + shockAge * 2.6;
    let band = 0.30 + shockAge * 0.72;
    let x = (dist - shell) / band;
    let env = exp(-x * x) * exp(-shockAge * 1.45) * strength;
    let shear = cross(dir, vec3f(0.31, 0.62, 0.19));
    acc = acc + (dir * 3.1 + shear * 0.85) * env;
  }

  // ---------------------------------------------------------------------
  // 7. Audio-driven sparkle: high frequencies jitter bright particles.
  // ---------------------------------------------------------------------
  let sparkle = high * col.w * 0.9;
  acc = acc + vec3f(
    sin(t * 41.0 + md.z * 91.0),
    sin(t * 47.3 + md.z * 57.0),
    sin(t * 53.7 + md.z * 73.0)
  ) * sparkle;

  // ---------------------------------------------------------------------
  // Integrate: semi-implicit Euler + exponential damping. speed scales dt
  // rather than forces so the motion stays physical at any speed setting.
  // ---------------------------------------------------------------------
  // uRender.w carries the user "speed" multiplier; scaling dt (not forces)
  // keeps trajectories identical in shape, just traversed faster.
  let dtS = dt * clamp(U.uRender.w, 0.0, 3.0);
  let step = acc * dtS;
  var v = (vel + vec4f(step, 0.0)) * exp(-U.uForce.y * dtS);

  // Soft velocity ceiling in R4: 3.2 units/s keeps the cloud coherent.
  let vm = length(v);
  if (vm > 3.2) { v = v * (3.2 * tanh(vm / 3.2) / vm); }

  var p = pos + v * dtS;

  // Hard safety net: nothing may escape the well, ever (nan/inf guards).
  let pr = length(p);
  if (!(pr < 12.0)) {
    let b = birth(u32(md.z * 4294967295.0) ^ (i * 2246822519u), u32(t * 11.0));
    p = b.p; v = b.v; col = b.c;
    md = vec4f(0.0, b.m.y, b.m.z, b.m.w);
  } else if (pr > 6.5) {
    p = p * (6.5 / pr);
    v = v - p * (dot(v, p) / dot(p, p));
  }

  // Fade energy with the audio spectrum so the nebula visibly "sings".
  let lifeFade = smoothstep(0.0, 0.10, age / max(md.y, 1e-3)) * (1.0 - smoothstep(0.82, 1.0, age / max(md.y, 1e-3)));
  let energy = mix(0.32, 1.0, lifeFade) * (1.0 + low * 0.55 + beat * 0.75 + sparkle * 0.4);
  col = vec4f(col.xyz * mix(0.86, 1.0, lifeFade * 0.65), energy);
  md.x = age;

  arrPongP[i] = p;
  arrPongV[i] = v;
  arrPongC[i] = col;
  arrPongM[i] = md;
}
`;

/**
 * Initial seeding kernel.
 *
 * Runs once per particle at startup (and whenever the store is reallocated),
 * writing into the pong set; the caller then blits pong -> ping so frame zero is
 * already consistent and renderable. Ages are scattered across the full lifespan
 * so the very first frames contain dying, living and newborn particles rather
 * than a synchronised flash of births.
 */
export const SEED_WGSL = /* wgsl */ `
${WGSL_CONSTANTS}
${WGSL_UNIFORMS}
${WGSL_STORAGE_BINDINGS}
${WGSL_MATH}
${WGSL_LIFECYCLE}

@compute @workgroup_size(${PARTICLE_WORKGROUP})
fn seed(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= u32(U.uTime.w)) { return; }

  let sd = hash1(i * 2654435761u + 17u);
  let age0 = rnd1(sd ^ 0xB5297A4Du) * 9.5;
  let b = birth(sd, 0u);
  arrPongP[i] = b.p;
  arrPongV[i] = b.v;
  arrPongC[i] = b.c;
  // A longer-than-natural first lifespan keeps the population stable while the
  // respawn drizzle establishes itself.
  arrPongM[i] = vec4f(age0, b.m.y + 2.0, b.m.z, b.m.w);
}
`;

/**
 * Parity blit (pong -> ping).
 *
 * WGSL forbids a dispatch from reading and writing the same storage binding, so
 * instead of swapping bind groups every frame the freshly simulated state is
 * mirrored back into the ping set. The render pass can then always read
 * arrPing*, which removes a whole class of "which half is live" bugs whenever the
 * particle count or quality changes mid-flight.
 */
export const COPY_WGSL = /* wgsl */ `
${WGSL_CONSTANTS}
${WGSL_UNIFORMS}
${WGSL_STORAGE_BINDINGS}
${WGSL_MATH}

@compute @workgroup_size(${PARTICLE_WORKGROUP})
fn copy(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= u32(U.uTime.w)) { return; }
  arrPongP[i] = arrPingP[i];
  arrPongV[i] = arrPingV[i];
  arrPongC[i] = arrPingC[i];
  arrPongM[i] = arrPingM[i];
}
`;

/** Entry-point names, exported so the pipeline descriptors cannot drift. */
export const SIM_ENTRY = 'simulate';
export const SEED_ENTRY = 'seed';
export const COPY_ENTRY = 'copy';
