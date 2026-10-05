# Aetheria

Aetheria is a real-time particle universe that lives in four spatial dimensions. Every frame, up to four million particles are integrated through a 4D quaternion-Julia attractor, a 4D Mandelbulb deformation, a Thomas-attractor swirl and an analytic divergence-free 4D curl-noise field, then projected to the screen through a rotating hyperplane slice. The simulation runs in a hand-written WGSL compute kernel; the image is assembled by a hand-written HDR post chain — soft-knee bloom, volumetric god rays, golden-angle depth of field, ACES tone mapping, film grain, a dynamic vignette and a 1-LSB dither. Audio (a microphone or a built-in procedural drone) drives four control signals that modulate turbulence, fractal iteration depth, palette and sparkle. There are no runtime dependencies, no shader plugins and no asset files: the application is TypeScript and shader source in a single page.

**What makes it hard:** four million particles, a 448-byte uniform block and a divergence-free vector field have to be integrated and post-processed inside a 16.6 ms frame with no runtime dependencies and no CPU-side sorting.

## Quick start

```bash
pnpm install          # dev dependencies only: typescript, vite, @webgpu/types
pnpm dev              # Vite dev server on http://127.0.0.1:5173
pnpm build            # tsc --noEmit && vite build  → dist/
pnpm preview          # static preview on http://127.0.0.1:4173 (--strictPort)
```

`pnpm build:fast` skips the type check, `pnpm typecheck` runs it alone, and `pnpm verify` runs the headless verification runner (see [Verified performance report](#verified-performance-report)).

**Browser requirement.** WebGPU is required for the primary path: Chrome or Edge 113+ on desktop, or Safari 18+ where available. If `navigator.gpu` is missing, the app falls back to a WebGL2 transform-feedback renderer (see [Degradation and fallbacks](#degradation-and-fallbacks)). With neither API, the boot overlay reports the failure and nothing else renders.

The application is a single page (`index.html` plus one ES module). `package.json` declares no runtime dependencies, and the Vite build inlines dynamic imports and disables CSS code splitting, so the deployable output is one self-contained bundle that fetches nothing at runtime; the only GPU resources are compute buffers and render targets created at startup.

## Controls

Every gesture is funnelled through `InputController` (`src/input/controls.ts`) into six verbs — orbit, zoom, slice, ignite, key, activity — plus the panel (`src/ui/panel.ts`).

| Input | Action | Detail |
| --- | --- | --- |
| Drag (mouse, pen, single finger) | 4D rotate | Default planes: `xw += dx·0.0055`, `yz += dy·0.0055`, `xy += dx·0.0016`, `yw += dy·0.0016` (`GpuRenderer.orbit`) |
| Drag with a plane locked (X/Y/Z) | 4D rotate in that plane | `xw`: `rot[2] += dx·0.006`, `rot[3] += dy·0.006`; `yw`: `rot[4] += dx·0.006`, `rot[5] += dy·0.006`; `zw`: `rot[5] += dx·0.006`, `rot[0] += dy·0.006` |
| Click / tap | Supernova | Only if the pointer moved under 6 px in total and was released within 400 ms, and no plane is locked. Unprojected through the inverse MVP; the shell ignites at `clamp(focusDist, 1.2, 8)` along the ray. No-op when `params.shock <= 0` |
| Wheel | Zoom | `zoom *= exp(deltaY · 0.0012)`, clamped to 1.3 – 14 |
| Shift + wheel | Hyperplane slice | `wSlice += deltaY · 0.0016`, clamped to ±3.2 |
| Two-finger pinch | Zoom + slice | `zoom((Δdistance) · 2.4)` and `slice(−(ΔcentroidY) · 3.2)` simultaneously |
| Two-finger touchstart | Suppressed | `preventDefault()` stops iOS page scroll/zoom under the canvas |
| Context menu, `gesturestart` | Suppressed | Right-click and Safari trackpad-pinch gestures never fire |
| `W` / `↑` | Orbit up | Step 11, or 26 with Shift; `Alt` forces the `yw` plane |
| `S` / `↓` | Orbit down | A plain `S` (no Alt/Shift/Ctrl) exports a PNG instead; any modifier makes it orbit down |
| `A` / `←`, `D` / `→` | Orbit left / right | `Alt` forces the `xw` plane |
| `Q` / `E` | Slice out / in | `slice(−40)` / `slice(+40)` |
| `+` or `=` / `−` or `_` | Zoom in / out | `zoom(−90)` / `zoom(+90)` |
| `Space` | Pause / resume | `preventDefault()` so the page does not scroll |
| `R` | Reset | Camera, six plane angles, `wSlice`, shocks, elapsed time, and re-seeds every particle |
| `H` | Hide / show UI | Toggles `body.ui-hidden`; the button's `aria-label` flips with it |
| `S` | Save PNG | `canvas.toBlob('image/png')` → `aetheria-YYYYMMDD-HHMMSS.png` |
| `V` | Record 10 s | `canvas.captureStream(60)` + `MediaRecorder` (VP9 → VP8 → WebM), 12 Mbit/s, auto-stops after 10 s |
| `X` / `Y` / `Z` | Lock the drag plane | Sticky lock to `xw` / `yw` / `zw`; pressing the active key releases it |
| `?` or `F1` | Help | Toast with the short binding list; `preventDefault()` |
| `Shift` + keyboard | Larger step | Orbit step 11 → 26 |
| Double-click canvas | Focus mode | Toggles the minimal HUD (`body.panel-idle`) |
| Device orientation | Orbit | Enabled by the panel's `Gyro` button; iOS 13+ prompts first. `orbit(Δgamma · 0.012, Δbeta · 0.012)` |

Keyboard events whose target is an `INPUT`, `TEXTAREA` or `SELECT` are ignored, so typing in a panel field never drives the camera.

| Panel control | Type | Range / behaviour |
| --- | --- | --- |
| Particles | slider | `simCount` 100 000 – 4 000 000, step 50 000 (simulated, in GPU storage buffers) |
| Displayed | slider | `showCount` 100 000 – 10 000 000, step 250 000 (extra copies are scattered sub-particles) |
| Speed | slider | 0 – 3, step 0.01 (simulation time scale) |
| Fractal dim. | slider | 0 – 1, step 0.01 (Mandelbulb exponent + Julia iteration depth) |
| Curl noise | slider | 0 – 2, step 0.01 |
| Damping | slider | 0.02 – 1.5, step 0.01 |
| Gravity | slider | 0 – 0.6, step 0.01 |
| Confinement | slider | 0 – 1.5, step 0.01 |
| Supernova | slider | 0 – 3, step 0.05 (shockwave impulse strength) |
| Trails | slider | 0 – 0.97, step 0.01 (frame-history persistence) |
| Exposure | slider | 0.2 – 3, step 0.01 (linear, applied before ACES) |
| Bloom | slider | 0 – 2, step 0.01 |
| Bloom size | slider | 0 – 1, step 0.01 (kernel radius bias) |
| Depth of field | slider | 0 – 1.5, step 0.01 |
| Chromatic | slider | 0 – 2, step 0.01 |
| Volumetric | slider | 0 – 1.5, step 0.01 |
| Film grain | slider | 0 – 1, step 0.01 |
| Vignette | slider | 0 – 1, step 0.01 |
| Hue | slider | 0 – 1, step 0.005 (palette rotation in turns) |
| Audio gain | slider | 0 – 3, step 0.01 |
| Resolution | slider | 0.4 – 1.4, step 0.02 (multiplied by the profile's `maxPostScale`) |
| Silent / Drone / Mic | buttons | Audio source, reflected in `aria-pressed` |
| Gyro | button | Device orientation on/off |
| Pause / Resume, Reset, Focus mode, Save PNG, Record 10s, Fullscreen | buttons | Mirrors of the key bindings above |
| Hide interface (◡) | icon button | Same as `H` |

The panel fades to 16 % opacity after 6 s without activity and returns on hover or `focus-within`; on narrow screens it docks to the bottom as a sheet (`max-height: 62vh`).

Two deviations between the advertised hints and the code are worth knowing. `index.html` says "shift+drag hyperplanes", but the drag path consults only the sticky X/Y/Z plane lock — Shift is read exclusively by the wheel handler (slice) and by the keyboard step size. The panel's Focus mode button is titled "(F)", but no `F` binding exists in `controls.ts`; focus mode is reachable from the button or the double-click.

## Architecture

One `requestAnimationFrame` tick does all CPU work, records one command encoder and submits it once. The diagram lists the passes in recorded order with the texture each one actually binds.

```text
 CPU (single thread, per frame)
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │ audio.update(dt) → low, mid, high, beat, rms                                                  │
 │ gather 6 plane angles, idle auto-rotate, camera eye/zoom, perspective · lookAt, invert(MVP)    │
 │ age and prune shocks, compute multiplicity, write Sim (448 B) and Post (128 B) uniform blocks  │
 └───────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                             │ queue.writeBuffer ×2
 GPU (one GPUCommandEncoder → one queue.submit)
 ┌───────────────────────────────────────────┴──────────────────────────────────────────────────┐
 │ 1  compute  simulate       ping  ───────────► pong      ceil(N/64) workgroups × 64 invocations │
 │ 2  compute  copy (parity)  pong  ───────────► ping      same dispatch geometry                  │
 │ 3  render   fade history   history ─────────► history   rgba16f, luma-weighted decay           │
 │ 4  render   particles      ping (read-only) + additive sprites ► history (load) + depth32f     │
 │ 5  render   bright pass    scene ───────────► bloom[0]  half res, 4-tap box + soft knee        │
 │ 6  render   bloom down     bloom[i-1] ──────► bloom[i]  13-tap Karis, ×(levels-1)              │
 │ 7  render   bloom up       bloom[i] ────────► bloom[i-1] 25-tap tent, additive, ×(levels-1)    │
 │ 8  render   god rays       bloom[0] + scene ► bloom[0]  24 radial taps, additive               │
 │ 9  render   dof down       scene + depth ───► dof       half res, 3×3, max CoC in alpha        │
 │10  render   dof gather     scene + dof ─────► scene     13 taps, golden angle                  │
 │11  render   composite      scene + bloom[0] ► scratch   ACES, grain, vignette, dither, sRGB    │
 │12  render   fxaa           scratch ────────► canvas     bgra8unorm swap chain                 │
 └──────────────────────────────────────────────────────────────────────────────────────────────┘
```

| File | Responsibility |
| --- | --- |
| `src/core/config.ts` | Device profiles and quality tiers, the full `Params` list with `PARAM_RANGES`, memory constants |
| `src/core/math4d.ts` | SO(4) plane rotations and their fixed composition order, perspective/look-at with NDC depth in [0,1], matrix inverse, click-ray transform |
| `src/core/uniforms.ts` | `SimWriter` / `PostWriter`: pack `SimState` / `PostState` into the 448-byte and 128-byte blocks |
| `src/core/types.ts` | The `Renderer` contract both backends implement, `HudStats`, telemetry hooks, `estimateParticleMemory` |
| `src/gpu/device.ts` | Adapter/device acquisition with limit fallbacks, capability probing, `compileChecked` WGSL diagnostics |
| `src/gpu/bridge.ts` | The single hop the WebGL2 fallback is allowed to import through |
| `src/gpu/particles.ts` | Ping-pong storage buffers, seed/simulate/copy pipelines, shock ring buffer, memory estimate |
| `src/gpu/post.ts` | Post-chain targets, four bind-group layouts, pipelines and the pass order |
| `src/gpu/renderer.ts` | Frame orchestration, camera and shock ownership, GPU timestamps, adaptive-quality control loop |
| `src/gpu/wgsl/common.wgsl.ts` | Uniform layouts, hash/RNG, palette, SO(4) algebra, analytic 4D curl noise, ACES, sRGB, dither |
| `src/gpu/wgsl/simulate.wgsl.ts` | `simulate`, `seed` and `copy` compute kernels |
| `src/gpu/wgsl/particle.wgsl.ts` | Billboard vertex/fragment shaders with multiplicity scatter and depth grading |
| `src/gpu/wgsl/post.wgsl.ts` | One module per post pass plus the fullscreen vertex shader |
| `src/webgl2/glutil.ts` | GL program/render-target helpers and the float-colour-buffer probe |
| `src/webgl2/shaders.ts` | GLSL ES 3.00 ports of the hash, field, palette, ACES and post vocabulary |
| `src/webgl2/renderer.ts` | Transform-feedback fallback renderer |
| `src/audio/analysis.ts` | Analyser band split, adaptive beat detector, procedural drone synth |
| `src/input/controls.ts` | Pointer, touch, wheel, keyboard and gyroscope → the six renderer verbs |
| `src/ui/panel.ts` + `styles.css` | Sliders, stats readouts, buttons, toasts, idle fade; glass styling, focus rings, reduced motion |
| `src/main.ts` | Bootstrap, backend selection, frame loop, URL-hash persistence, PNG/WebM export |
| `tools/verify.mjs` | Headless Chrome verification runner (§12) |
| `tools/profile-gpu.mjs` | Per-pass GPU breakdown and ablations (`pnpm profile`) |
| `tools/tune-visual.mjs` | Visual calibration loop that measures structure, not taste (`pnpm tune`) |
| `index.html` | The single page: canvas, boot overlay, ARIA prose, hint strip |

### The ping-pong parity trick

Particles live in eight storage buffers per set — `P`, `V`, `C`, `M` as `array<vec4f>` — held twice (`bufP[0..1]`, `bufV[0..1]`, `bufC[0..1]`, `bufM[0..1]`). WGSL forbids a dispatch from reading and writing the same storage binding, so the compute step uses two bind groups over one layout: `sim-ping->pong` binds buffers `[0]` to bindings 1–4 (read) and `[1]` to bindings 5–8 (write); `copy-pong->ping` swaps them. `ParticleSystem.step()` dispatches `simulate` and then `copy` with the same workgroup count, so the **ping** set holds this frame's state and the **pong** set holds an identical copy.

The render bind group is built once against `bufP[0] … bufM[0]` and never rebinds, so the vertex shader always reads the ping set: there is no "which half is live this frame" flag anywhere in the render path. That statelessness is what keeps the renderer correct when the particle count, the quality tier or an adapter-imposed cap changes mid-flight, because a reallocation rebuilds both bind groups symmetrically instead of leaving the renderer one parity flip out of phase. The cost is one extra full-rate dispatch of `ceil(N/64)` workgroups per frame; the source estimates roughly 0.25 ms for two 1M-particle kernels on an RTX-class part.

## The four-dimensional fractal attractor

`simulate` evaluates the force model in a fixed order so every term sees the same rotated frame. The shared preamble is `rp = rot4_apply(pos)`, `rp3 = rp.xyz`, `r3 = length(rp3)`, plus the four audio signals.

1. **4D quaternion Julia attractor.** `julia4(rp · innerScale, uJulia, iters)` iterates `z = quatSquare(z) + c`, with the quaternion square expanded explicitly as `(2(xw − yz), 2(yw + xz), 2(zw − xy), w² − x² − y² − z²)`. Each iterate is accumulated with the weight `exp(−0.85·i)`, so early iterates dominate, and the loop stops as soon as `|z| > 6` (the bail-out radius) or after at most 16 iterations. `c` is the four-component `julia` parameter (default `−0.42, 0.61, −0.27, 0.35`, settable only through the URL hash's `j=` key); `innerScale = mix(0.72, 0.36, dim)` and `iters = clamp(4 + 12·dim + 3·mid, 4, 16)` with `dim = clamp(fractalDim, 0, 1)` and `mid` the mid-band audio signal. The R⁴ displacement is saturated by `softLimitAccel(·, 26)`, then scaled by `0.30 + 0.42·dim`.
2. **4D Mandelbulb triplex deformation.** `mandelbulb4(rp · 0.55, power, warp)` converts the point to `(r, θ, φ, ψ)` with `θ = acos(clamp(p.w/r))`, `φ = atan2(p.y, p.x)`, `ψ = atan2(p.z, hypot(p.x, p.y))`, raises it to `zr = r^power`, wraps every angle by `power`, rebuilds `q`, and returns `(q − p) / (1 + dot(q − p, q − p)) · warp` — a bounded push toward the deformed iso-surface. `power = 2.35 + 4.15·fractalDim` (equivalently `mix(2.35, 6.5, dim)`, the expression `GpuRenderer` uploads) and `warp = mix(0.22, 0.85, dim) · (1 + 0.6·mid)`. Because the exponent follows the slider, `fractalDim` changes the topology of the nebula rather than just its brightness. The result is saturated by `softLimitAccel(·, 14)` and scaled by 0.5.
3. **Thomas attractor.** `thomas = (sin(y) − 0.19x, sin(z) − 0.19y, sin(x) − 0.19z)`, scaled by 0.14. It is a bounded chaotic flow and supplies the large-scale spiral arms; it is deliberately the cheapest term in the model.
4. **Analytic divergence-free 4D curl noise.** Three octaves (`amp /= 1.7`, `freq *= 2.03`) of a vector potential `A = (s₀, s₁, s₂, s₀s₁s₂)` with `s_k = sin(dot(w_k, p) + phase_k)` and three tilted wave vectors per octave (`w₀ = (0.83, 1.31, 0.57, 1.07)·f`, `w₁ = (1.21, 0.49, 1.37, 0.73)·f`, `w₂ = (0.61, 1.13, 0.91, 1.49)·f`). The field is assembled from antisymmetric gradient combinations, for example `curl.x = (g.y − g.z) + 0.5·g.w`, with cyclic permutations for `y`, `z` and `w`. **Why this matters:** the continuous curl satisfies `∇·(∇×A) ≡ 0` identically — a curl field has no sources and no sinks. A flow that follows one is incompressible: it cannot pile particles into knots or evacuate voids, so density is only ever rearranged. That is what produces silk-like filaments instead of noisy clumping; rotation is what makes those filaments look like turbulence. The gain is `curlGain = curl · (1 + 1.35·low + 0.35·beat)`, passed as `curl4(rp, t·0.7, curlGain·0.62, warp + 0.9·mid)`, and only the `.xyz` components join the 3D acceleration.
5. **Gravity and the confinement shell.** With `rn = rp3 / r3`, the inverse-square pull is `−rn · (gravity / (0.55 + r3²))`, and the shell is `−rp3 · confinement · smoothstep(1.9, 5.2, r3)`: zero inside radius 1.9, growing outside 5.2 — a soft wall rather than a hard box.
6. **Supernova shockwaves.** Up to `MAX_SHOCKS = 8` expanding shells. The shock buffer is two `vec4` arrays: the first eight entries are `(origin.xyz, age)`, the second eight carry the strength in `.x`. For each live shell, `shell = 0.35 + 2.6·age`, `band = 0.30 + 0.72·age`, and the envelope is `exp(−x²) · exp(−1.45·age) · strength` with `x = (distance − shell)/band`. The impulse is `(dir · 3.1 + cross(dir, (0.31, 0.62, 0.19)) · 0.85) · env`: a radial push plus a fixed-axis shear, so the shell expands as a ring with a twist rather than as a perfect sphere.
7. **Audio coupling.** Finally `sparkle = high · col.w · 0.9` adds a per-particle jitter along three incommensurate sinusoids (41.0, 47.3 and 53.7 Hz times time, with the particle seed as phase), so only bright particles sparkle and they never repeat.

Integration is semi-implicit Euler with exponential damping: `dtS = dt · clamp(speed, 0, 3)`, `v = (vel + vec4(acc·dtS)) · exp(−damping·dtS)`, `p = pos + v·dtS`, with `dt` clamped to 1/240 – 1/30 s. Three guards keep the cloud stable: a tanh soft acceleration limiter (linear near zero so small forces stay exact, asymptotic at the limit so nothing explodes), a tanh velocity ceiling of 3.2 units/s in R⁴, and a geometric safety net — if `|p|` is not `< 12` (NaN and infinity included) the particle is reborn, and if `|p| > 6.5` it is projected back onto the 6.5 shell with its radial velocity component removed.

Respawn is deterministic and metered. A life ends when `age > lifespan`, with the lifespan drifting by `dt·1e-4` per frame so lifetimes stay mutually incommensurate. Only particles for which `fract(|seed|·137 + t·0.37) < 0.16` are reborn on a given frame; the rest are parked in a long tail (`age = 0.02·lifespan`, `lifespan = 1.6·lifespan + 1`) so they never simply vanish. Rebirth cross-fades 86 % of position and velocity and 90 % of colour, turning a teleport into a drift. `birth()` derives position, velocity, colour and lifecycle from `(seed, epoch)` alone through the integer hash, so no per-particle RNG state exists and the universe is reproducible from `(index, epoch)`.

### Six rotation planes, and what the hyperplane does

A 4D rotation has six independent planes. Aetheria composes all six in a fixed order that is identical in the WGSL shader, in `rotationMatrix4()` (`src/core/math4d.ts`) and in the WebGL2 fallback:

```wgsl
// R = Rzw · Ryw · Rxw · Ryz · Rxz · Rxy   (applied xy, xz, xw, yz, yw, zw)
o = rot4_plane(o, 0u, 1u, U.uRot0.x); // xy     o = rot4_plane(o, 1u, 2u, U.uRot0.w); // yz
o = rot4_plane(o, 0u, 2u, U.uRot0.y); // xz     o = rot4_plane(o, 1u, 3u, U.uRot1.x); // yw
o = rot4_plane(o, 0u, 3u, U.uRot0.z); // xw     o = rot4_plane(o, 2u, 3u, U.uRot1.y); // zw
```

The angles arrive as `uRot0 = (xy, xz, xw, yz)` and `uRot1 = (yw, zw, tanHalfFov, time4D)`. Dragging advances `xw`/`yz` most strongly (the planes the eye reads as tumbling) with a smaller `xy`/`yw` component; Shift raises keyboard orbit steps; `X`/`Y`/`Z` lock one plane. After 2.2 s of inactivity the renderer adds a slow drift on `xy`, `yw` and `zw`, disabled under `prefers-reduced-motion`.

The hyperplane slice is what turns the fourth coordinate into apparent volume. In `vs_main`:

```wgsl
var q = rot4_apply(vec4f(p4.xyz + jitter, p4.w + U.uSlice.x));  // uSlice.x = wSlice
let wPersp = 1.0 / (1.0 + q.w * U.uSlice.y);                    // uSlice.y = 4D fov (1.9)
q = vec4f(q.xyz * wPersp, q.w);
let wDepth = q.w * 0.34 * wPersp;
```

`uSlice.x` shifts the cut through 4-space; `uSlice.y` sets how strongly the surviving `w` acts as a perspective divide. A particle with a large `|w|` is scaled toward the origin in xyz, so its view-space depth changes and, with it, both its projected size (the sprite size divides by `max(clip.w, 0.05)`) and the depth value written to the depth attachment. That attachment is the only depth DOF has: the perceptual circle of confusion is `clamp(|d − focus| / focus · aperture · 4, 0, 1)`, so the 4D perspective divide is exactly what makes the near and far sides of the hyperplane cut receive different blur. `w` also enters the palette phase (`+ q.w · 0.10`) and the focus distance (`focusDist = clamp(zoom − 0.25·wSlice, 1.0, 20)`), so slicing the hyperplane retargets the focal plane as well as the geometry.

## Shader reference

| Module (source) | Entry point | Stage | Bindings, group 0 | What it does |
| --- | --- | --- | --- | --- |
| `simulate.wgsl.ts` → `SIMULATE_WGSL` | `simulate` | compute, `@workgroup_size(64)` | 0 `Sim` uniform; 1–4 ping `P/V/C/M` storage rw; 5–8 pong `P/V/C/M` storage rw; 9 `shocks` storage | One particle step in R⁴: lifecycle, seven force terms, integration, guards |
| `simulate.wgsl.ts` → `SEED_WGSL` | `seed` | compute, 64 | Same layout, ping→pong group | Writes a fresh `Birth` into the write set, ages scattered over the full lifespan |
| `simulate.wgsl.ts` → `SEED_WGSL` | `copy` | compute, 64 | Same layout, pong→ping group | Parity blit: read set → write set, four `vec4` arrays |
| `particle.wgsl.ts` → `PARTICLE_VERTEX_WGSL` | `vs_main` | vertex | 0 `Sim` uniform (VERTEX\|FRAGMENT); 1–4 ping arrays as read-only storage (vertex); 9 `shocks` | Six vertices per instance expand an axis-aligned quad in pixel space; decodes multiplicity sub-copies from `instance_index` |
| `particle.wgsl.ts` → `PARTICLE_FRAGMENT_WGSL` | `fs_main` | fragment | Shares the same bind group; reads only the uniform | Two analytic falloffs (`exp(−9r²)` core, `(1−r)^2.6` halo), diffraction spikes, rim dispersion |
| `post.wgsl.ts` → `FULLSCREEN_VS` | `vs_fullscreen` | vertex | None | Oversized triangle `(0,0) (2,0) (0,2)`; no vertex or index buffer |
| `post.wgsl.ts` → `FADE_WGSL` | `fs_fade` | fragment | A: 0 `texA`, 1 sampler, 4 `Post` uniform | Frame-history decay |
| `post.wgsl.ts` → `BRIGHT_PASS_WGSL` | `fs_bright` | fragment | A | 4-tap box + soft knee |
| `post.wgsl.ts` → `BLOOM_DOWN_WGSL` | `fs_down` | fragment | A | 13-tap Karis downsample |
| `post.wgsl.ts` → `BLOOM_UP_WGSL` | `fs_up` | fragment | AB: 0 `texA`, 1 sampler, 2 `texB`, 4 `Post` | 25-tap tent, additive blend |
| `post.wgsl.ts` → `GODRAY_WGSL` | `fs_godray` | fragment | AB (the pipeline is built against A — see the notes in §7) | 24 radial taps toward the frame centre |
| `post.wgsl.ts` → `DOF_DOWN_WGSL` | `fs_dof_down` | fragment | ABD: 0 `texA`, 1 point sampler, 2 `texture_depth_2d`, 4 `Post` | 3×3 tile average + maximum CoC |
| `post.wgsl.ts` → `DOF_GATHER_WGSL` | `fs_dof` | fragment | AB | 13-tap golden-angle bokeh |
| `post.wgsl.ts` → `COMPOSITE_WGSL` | `fs_composite` | fragment | AB | Radii, chroma, bloom, exposure, ACES, grain, vignette, dither, sRGB |
| `post.wgsl.ts` → `FXAA_WGSL` | `fs_fxaa` | fragment | A | Luma edge detection + directional blur |

Shared preludes: `WGSL_CONSTANTS` (π/τ and the `ARRAY_*` indices), `WGSL_UNIFORMS` (the `Sim` struct and all storage declarations), `WGSL_CORE_MATH` (hash, palette, hue rotation, ACES, sRGB, dither — binding-free, so post passes can include it), `WGSL_SIM_MATH` (SO(4), curl, Julia, Mandelbulb — these read `U`, so only the simulation and the particle pass may include it) and `WGSL_POST_UNIFORM`. The `Sim` block also declares binding 10 (`Post`); the compute kernels never reference it, so it is pruned and the compute layout has no entry for it.

`Sim` — 448 bytes, 16-byte aligned, written by `SimWriter`:

| Offset | Size | Field | Components |
| --- | --- | --- | --- |
| 0 | 16 | `uTime` | `x=t`, `y=dt`, `z=frame`, `w=simCount` |
| 16 | 16 | `uViewport` | `x=w`, `y=h`, `z=1/w`, `w=1/h` |
| 32 | 16 | `uCamera` | `xyz=eye`, `w=zoom` (projection distance) |
| 48 | 16 | `uSlice` | `x=wSlice`, `y=4D fov`, `z=nearBlend`, `w=aspect` |
| 64 | 16 | `uJulia` | `(a, b, c, d)` Julia quaternion |
| 80 | 16 | `uForce` | `x=curl`, `y=damping`, `z=gravity`, `w=confinement` |
| 96 | 16 | `uAttract` | `x=depth`, `y=escape`, `z=power`, `w=warp` |
| 112 | 16 | `uAudio` | `x=low`, `y=mid`, `z=high`, `w=beat` |
| 128 | 16 | `uPalette` | `x=hue`, `y=saturation`, `z=paletteMix`, `w=exposure` |
| 144 | 16 | `uRender` | `x=sizeScale`, `y=energyScale`, `z=multiplicity`, `w=speed` |
| 160 | 64 | `uProj` | `mat4x4f` projection |
| 224 | 64 | `uView` | `mat4x4f` view |
| 288 | 64 | `uRot4` | `mat4x4f` composed SO(4) rotation |
| 352 | 16 | `uRot0` | plane angles `xy, xz, xw, yz` |
| 368 | 16 | `uRot1` | plane angles `yw, zw`, `tanHalfFov`, `time4D` |
| 384 | 16 | `uMisc` | `x=shockGain`, `y=bloom`, `z=dpr`, `w=quality` |
| 400 | 16 | `uFocus` | `x=focusDist`, `y=aperture`, `z=grain`, `w=vignette` |
| 416 | 16 | `uShockMisc` | `x=activeShocks`, `y=shockRadius`, `z=thickness`, `w=life` |
| 432 | 16 | `uMisc2` | `x=trailDecay`, `y=paletteMix`, `z=saturation`, `w=reserved` |

`Post` — 128 bytes:

| Offset | Size | Field | Components |
| --- | --- | --- | --- |
| 0 | 16 | `pResolution` | `x=w`, `y=h`, `z=1/w`, `w=1/h` |
| 16 | 16 | `pTexel` | `xy = blur direction (per mip)`, `zw = mip texel size` |
| 32 | 16 | `pParams0` | `x=exposure`, `y=bloom`, `z=bloomRadius`, `w=threshold` |
| 48 | 16 | `pParams1` | `x=chroma`, `y=grain`, `z=vignette`, `w=dof` |
| 64 | 16 | `pParams2` | `x=volumetric`, `y=time`, `z=aspect`, `w=frame` |
| 80 | 16 | `pFocus` | `x=focusDepth`, `y=focusRange`, `z=maxCoC`, `w=nearPlane` |
| 96 | 16 | `pFar` | `x=farPlane`, `y=sourceLod`, `z=upsampleRadius`, `w=bloomLevels` |
| 112 | 16 | `pColor` | `x=hue`, `y=saturation`, `z=quality`, `w=sceneLuma` |

## Post-processing chain

Every intermediate is linear HDR (`rgba16float`) until the composite encodes sRGB; depth is `depth32float`. Bloom levels default to 4 (mobile), 5 (balanced) or 6 (high, ultra) and are clamped to 2–8; the pyramid stops early if a mip would fall below 4 px, and two levels are forced if that would leave fewer than two.

### Fade history

One tap: the history texture times `k · mix(0.55, 1.0, smoothstep(0, 1.2, luma))`, where `k = params.trails` (0 – 0.97). The decay runs in linear HDR before anything else, so a 12.0-luminance star leaves a longer trail than a 0.3-luminance dust mote — the behaviour of a real long exposure — and the luminance weighting stops high persistence from turning the nebula into a uniform grey wash. It runs before the particle pass so the particles add onto an already-decayed image rather than underneath it.

### Bright pass

Half resolution: a 4-tap box (`±0.5` texel), then `threshold = 1.0` applied with a soft knee of `max(threshold · 0.62, 1e-3)` — `soft = clamp(br − thr + knee, 0, 2·knee)`, `contrib = max(soft²/(4·knee), br − thr) / max(br, 1e-4)`. **Why a soft knee rather than a hard threshold:** thousands of sprites cross any fixed cut every frame as particles are born and die, and a hard cut turns each crossing into a visible twinkle along the bloom boundary; the quadratic knee fades the contribution in over a 0.62-unit band instead. The 4-tap box keeps the smallest mip stable under motion.

### Bloom downsample

13 taps in the Call-of-Duty / Jimenez layout (3×3 core, four mid-edges, four corners), each tap passed through a Karis average `rgb / (1 + luma(rgb))` before the weighted mix `mix(mean13, core·1.6, 0.55)`. **Why Karis rather than a plain average:** the weights are inverse-luma, so one bright firefly — a freshly born hot star — cannot dominate a tile and flicker as it moves. **Why 13 taps rather than a naive bilinear chain:** halving by bilinear sampling pulse-beats on thin, high-contrast features, which is exactly what particle filaments are, while the wider kernel suppresses that pulsing.

### Bloom upsample

A separable 5-wide tent with weights `1, 2/3, 1/3, 2/3, 1` per axis — 25 taps as implemented, referred to as "9-tap" in the source comment — times `1/16`, then blended additively (`one + one`) into the finer level with `radius = mix(0.7, 1.25, bloomRadius)` scaling the upsampled energy against the level's own accumulated value. **Why a tent rather than a box:** a box upsample leaves rectangular mip boundaries that are obvious against a black sky, while the tent's linear falloff produces a smooth halo; accumulating additively up the chain yields a multi-scale glow without a separate combine pass.

### Volumetric light (god rays)

24 radial taps stepping toward the frame centre, `dir = (0.5 − uv) · (0.22 + 0.18·brightBias)` with `brightBias = smoothstep(0.12, 1.4, luma(sceneSample))`, a 0.94 decay per tap, normalised by the accumulated weight, then gated twice: `smoothstep(0.25, 1.1, maxChannel)` keeps only bright structures, and the `(0.6 + 0.4·brightBias)` factor ties the streak to where the light actually is. It is written as a **gather** (each pixel walks toward the centre) rather than a scatter, which makes it deterministic and removes the need for atomics or a ping-pong target, and it returns black immediately when `volumetric <= 0.001`.

### DOF downsample

Half resolution, 3×3 = 9 taps of colour and depth through a non-filtering sampler (depth textures pair with a `non-filtering` sampler under WebGPU's default sample types). Colour is averaged; the tile keeps the **maximum** perceptual CoC `clamp(|d − focus| / focus · aperture · 4, 0, 1)` in its alpha channel. **Why maximum CoC instead of the average:** averaging lets one sharp foreground pixel drag a fully blurred tile back toward focus, which produces the classic light halo around in-focus particles; the maximum guarantees that a tile containing anything strongly defocused is treated as defocused.

### DOF gather

13 taps: the centre plus 12 golden-angle samples at `r = sqrt((i + 0.5)/12)` and `a = i · 2.399963229728653`, with a pixel radius of `upsampleRadius · coc^0.85 · min(height, 1440) · 0.045` and neighbours weighted by their own stored CoC (`mix(0.35, 1.0, neighbourCoC)`) so sharp detail never bleeds outward. **Why the golden angle:** the low-discrepancy spiral fills a disc evenly at any sample count, so twelve taps already read as a round, smooth bokeh disc instead of a rosette, and no jittered frames or history are needed to hide the pattern. **Why `sqrt` on the radius:** it keeps the sample area proportional to the blur amount instead of to its square, so raising `dof` widens the disc at a visually even rate. **Why `pow(coc, 0.85)`:** a mild gamma on the CoC keeps the transition into blur from snapping.

### Composite

The order is deliberate: barrel distortion (`1 + r²·0.012·chroma·0.5`), three-tap radial chromatic aberration (`ca = rel·(r² · chroma · 0.0022 · 3 + chroma · 0.0022)` sampled as R, G and B separately), bloom added from the pyramid's first level, luma-weighted film grain (two interleaved-gradient-noise evaluations scaled by 0.085 and weighted `0.35 + 0.65·smoothstep(0, 0.5, luma)`), exposure, tone mapping as `mix(acesApprox, acesFilm, 0.35)` — 65 % of the ACEScg-matrix RRT+ODT fit for richer shadows, 35 % of Hill's simpler fit — then the master hue rotation and saturation, then a dynamic radial vignette, and only then a dither of `bayer8 + ign − 1` scaled by `1/255`, before the sRGB encode. **Why chromatic aberration before the tone map:** fringing is a lens property, so it belongs on scene-referred values; applied after ACES it would fringe the compressed image and read as a colour cast. **Why dither after tone mapping:** quantisation error exists only in the final 8-bit signal, so noise added earlier would be tone mapped away. The amplitude is one LSB — enough to break up Mach banding in the dark nebula without adding visible grain — and combining an ordered 8×8 Bayer term with interleaved gradient noise avoids the fixed cross-hatch of Bayer alone.

### FXAA

Luma edge detection over the centre pixel and its four diagonal neighbours, threshold `max(0.028, lMax · 0.115)`, a direction estimate from the diagonal luma differences, `dirReduce` of `max(mean·0.06, 1/128)`, the blur vector clamped to ±8 texels, and two candidate averages (2-tap and 4-tap) selected by whether the wider one stays inside the local luma range. It runs last, on the already sRGB-encoded image, so edge detection is perceptual, and it returns after a single texture read on flat pixels — most of the frame. The composite writes its LDR result into an `rgba16float` scratch rather than straight to the swap chain, because FXAA must not read and write the same texture in one pass.

**Source-level notes.** These five points were read out of the code; this documentation pass did not build or run the application, so they are observations about the source rather than observed behaviour.

- The particle pass renders into `history` (`gpu/renderer.ts`, `colorAttachments: [{ view: this.post.historyView … }]`), while the bright, DOF and composite bind groups all sample `scene` (`gpu/post.ts`). Nothing in the WebGPU path writes `scene` except the DOF gather's own target.
- `pTexel` is never assigned on the WebGPU path: `GpuRenderer.frame()` sets the other `PostState` fields but leaves `texelX/texelY/srcTexelX/srcTexelY` at the `makePostState()` defaults of 1, and `fs_bright`, `downsample13`, `fs_dof_down`, `fs_dof` and `fs_fxaa` all derive their offsets from `P.pTexel`. The WebGL2 backend does set them, in `uploadPost`.
- Three passes bind a texture as a sampled resource and as the colour attachment of the same pass: the fade pass (history → history), the god-ray pass (bloom[0] → bloom[0]) and the DOF gather (scene + dof → scene, `post.ts` line 443). A single WebGPU render pass may not do that, which is the same constraint `post.ts` cites in its header as the reason two full-resolution HDR buffers exist.
- The god-ray pipeline is created against `layoutA` while its bind group is built with `layoutAB` (`post.ts` lines 222 and 342–351); the shader also declares `texB` on binding 2 for `brightBias`, and `layoutA` has no entry for it.
- `vs_main` computes `out.depth = depth − wDepth` and documents it as the channel fog and DOF read, but `fs_main` never consumes the `depth` varying; the DOF coupling runs through the rasterised depth attachment instead.

## Audio reactivity

Two sources feed one `AnalyserNode` (`FFT_SIZE = 2048`, `smoothingTimeConstant = 0.68`, range −96 dB to −14 dB), and the visual engine never learns which one is running:

- **Microphone** — `getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } })`, requested only from a user gesture. A denied or unavailable microphone sets `permission = 'denied'`, keeps the error text, and leaves the visuals running with audio off.
- **Procedural drone** — a minor-ninth pad: a lowpass at 420 Hz with Q 6.5 into a compressor (−18 dB threshold, 24 dB knee, ratio 4), five detuned voices at ratios `1, 1.5, 2.25, 3, 4.5` of a 55 Hz (A1) or 61.735 Hz (B1) root, detunes `−7, 5, −3, 9, −11` cents, the first three sawtooth and the last two triangle, gains `0.14/(1 + 0.4i)`, a sub sine at half the root, a shimmer triangle at `9.02·root`, a sub pulse every 2.6 s, detune drift every 4.2 s and filter breathing every 2.6 s; output fades in to 0.32 over 3 s. The graph is connected to the analyser **only** — never to `AudioContext.destination` — so the drone drives the picture without making a sound.

Bands are perceptual thirds of the spectrum, computed as energy (the mean of squared, normalised bin magnitudes) and then compressed by a root curve:

| Signal | Range (Hz) | Curve | Gain after transform |
| --- | --- | --- | --- |
| low | 28 – 180 | 0.5 | ×3.2 |
| mid | 180 – 2000 | 0.5 | ×3.4 (×1.25 tilt) |
| high | 2000 – 12000 | 0.55 | ×4.0 (×1.5 tilt) |
| rms | all bins, 0 – Nyquist | — | HUD level meter only |

Bin edges are `floor(fromHz / binHz) … ceil(toHz / binHz)` with `binHz = (sampleRate/2) / 1024`; the header comment describes the bands as "30–180 Hz, 180–2 kHz, 2–12 kHz", while the code's own edges are 28, 180, 2000 and 12000 Hz. Smoothing is asymmetric so transients survive but the picture does not flicker: attack `1 − exp(−22·dt)`, release `1 − exp(−6·dt)`, selected per sample by whether the target is above or below the current envelope. With audio switched off the envelopes decay by `1 − min(1, 3·dt)` (4·dt for beat) rather than snapping to zero, so turning audio off never jolts the nebula.

Beat detection is adaptive and source-agnostic. The low band's energy (`rawLow²`) goes into a rolling 84-sample history — about 1.4 s at 60 fps — and a beat is declared when `energy > mean·1.35 + std·1.15 + 1e-4`, the history holds more than 12 samples, and at least **0.22 s has elapsed since the last beat** (the refractory period). Strength scales with the overshoot: `min(1.6, 0.55 + (energy − threshold)/max(threshold, 1e-5) · 0.35)`, and the beat envelope decays as `exp(−7.5·dt)`.

| Signal | Simulation (`simulate.wgsl.ts`) | Particle vertex shader | WebGL2 fallback |
| --- | --- | --- | --- |
| low | curl gain ×`(1 + 1.35·low)` — the nebula churns and breathes | sprite size ×`(1 + 0.25·low)` | curl gain ×`(1 + 0.8·low)`, colour ×`(1 + 0.5·low)` |
| mid | Julia iterations `+3·mid` (clamped to 4–16), Mandelbulb warp ×`(1 + 0.6·mid)`, curl warp `+0.9·mid` — the fractal changes shape | palette phase `+0.30·mid` | palette phase `+0.35·mid` |
| high | per-particle sparkle jitter `high · col.w · 0.9`, energy ×`(1 + 0.4·sparkle)` | diffraction spikes `hero · high` | via the size pulse |
| beat | curl gain `+0.35·beat`, energy ×`(1 + 0.75·beat)` | sprite size ×`(1 + 0.22·beat)` | auto-supernova when `beat > 0.6`, at most every 0.28 s |
| rms | — | — | HUD meter |

The renderer additionally scales all three bands by `0.6 + 0.4·audioSensitivity` (clamped to 0–3), and when `beat > 0.72` the app itself ignites a supernova at a slowly drifting off-centre position with strength `0.55 + 0.4·beat`, so strong hits become expanding shells rather than only brightness.

## Performance

**Memory constants.** `PARTICLE_BYTES_PER_SIM = 88` is the budget constant declared in `src/core/config.ts` ("2× (Vec4+Vec4+Vec4+Vec2) = 2×44"), and it is what the documented budget arithmetic uses: `estimateParticleMemory()` multiplies it by 2, giving **176 bytes per simulated particle** in the HUD and in the 380 MiB guard. For reference, `ParticleSystem.allocate()` creates eight buffers of `n · 16` bytes (four arrays × two ping-pong sets), i.e. **128 bytes per particle** actually allocated — the constant is a conservative budget rather than the allocation size. Shock storage adds `8 · 2 · 16 = 256` bytes and the two uniform blocks add 448 + 128 bytes.

| Particle count | 88 B/particle (declared) | 176 B/particle (estimator) | 128 B/particle (allocated) |
| --- | --- | --- | --- |
| 200 000 (mobile) | 16.8 MiB | 33.6 MiB | 24.4 MiB |
| 500 000 (balanced) | 42.0 MiB | 83.9 MiB | 61.0 MiB |
| 1 000 000 (high) | 83.9 MiB | 167.8 MiB | 122.1 MiB |
| 1 500 000 (ultra) | 125.9 MiB | 251.8 MiB | 183.1 MiB |
| 2 263 970 (380 MiB guard) | 190.0 MiB | 380.0 MiB | 276.4 MiB |
| 4 000 000 (`HARD_MAX_SIM_PARTICLES`) | 335.7 MiB | 671.4 MiB | 488.3 MiB |

Render targets, computed from `PostChain.estimateBytes()`: three full-resolution `rgba16float` buffers (history, scene, scratch) at 8 B/px each, depth at 4 B/px, the bloom pyramid at 8 B/px over areas `1/4 + 1/16 + …`, and half-resolution god-ray and DOF buffers at 8 B/px each:

| Draw resolution | Full-res HDR + depth | Bloom pyramid (6 levels) | God rays + DOF | Total |
| --- | --- | --- | --- | --- |
| 1280 × 720 | 24.6 MiB | 2.3 MiB | 3.5 MiB | 30.5 MiB |
| 1920 × 1080 | 55.4 MiB | 5.3 MiB | 7.9 MiB | 68.6 MiB |
| 2560 × 1440 | 98.4 MiB | 9.4 MiB | 14.1 MiB | 121.9 MiB |
| 3840 × 2160 | 221.5 MiB | 21.1 MiB | 31.6 MiB | 274.2 MiB |

Combined, a 1920 × 1080 frame with 1 000 000 simulated particles is roughly 167.8 + 68.6 ≈ 236 MiB by the estimator's arithmetic. The HUD turns its VRAM readout red above 480 MB.

**Targets.** Desktop targets 60 fps — a 16.6 ms budget; mobile targets 30 fps — 33.3 ms (`adapt()` chooses from `profile.isMobile`). The HUD frame meter is always scaled against 16.6 ms.

**Adaptive quality.** `frame()` calls `adapt()` every 30 frames, driven by an EMA of the measured round-trip frame time (`rttEma`, α = 0.08). Degradation walks down three tiers in order and recovers in reverse order with tighter thresholds:

| Tier | Degrade trigger | Action | Floor | Recover trigger | Action |
| --- | --- | --- | --- | --- | --- |
| 1. Resolution | `ms > budget · 1.18` | `targetScale − 0.06` | 0.55 | `ms < budget · 0.85` | `targetScale + 0.03`, capped at `maxPostScale` |
| 2. Post effects | `ms > budget · 1.5` | `postLevel − 1`: 2 → 1 drops DOF and god rays, 1 → 0 leaves bloom only | 0 | `ms < budget · 0.72` | `postLevel + 1`, capped at 2 |
| 3. Particles | `ms > budget · 1.75` | `simCount × 0.82` and reallocate | 150 000 | never | — |

Resolution is shed first because it is the cheapest and least visible; particle count is last because shrinking the store invalidates every particle's history, so it carries strong hysteresis and never recovers automatically. `renderScale` eases toward `targetScale` at 25 % per evaluation instead of snapping. Each downgrade is recorded once in `stats.degraded` and surfaced in the backend badge (`auto-degraded: resolution`, `… depth of field + volumetric light`, `… post-processing`, `… particle count`).

**Startup caps.** The requested count is `min(params.simCount, HARD_MAX_SIM_PARTICLES)`, then capped by `maxParticlesForDevice()` (`min(maxStorageBufferBindingSize, maxBufferSize) / 88`, itself capped at 4 000 000), then by the 380 MiB guard `floor(380 · 1024 · 1024 / 176) = 2 263 970`. Both caps are reported as downgrade strings in the HUD. Multiplicity is `clamp(showCount / simCount, 1, 8)`; because the drawn count is `simCount · multiplicity`, a 10 000 000-particle request over a 1 000 000-particle store is served by eight scatter copies per particle rather than eight times the memory. Instances are drawn in calls of at most `max(65 536, min(2 000 000, maxBufferSize/16 − 1))`, with `firstInstance` carrying the global instance id so the shader can decode both the particle index and the sub-copy index; at most 64 calls are issued per frame.

**GPU timing.** When `timestamp-query` is available, three rotating query sets measure the particle pass (`beginningOfPassWriteIndex: 0`, `endOfPassWriteIndex: 1`) and are read back without blocking via `mapAsync`; the reported GPU time is an EMA with α = 0.15. Pass-level timestamps are used rather than `GPUCommandEncoder.writeTimestamp`, so the code does not depend on a newer entry point.

## Degradation and fallbacks

`main.ts` tries WebGPU first, then the WebGL2 renderer, then reports failure. Both implement the same `Renderer` contract and are fed by the same `Params`, so the UI, hash serialisation and audio layer never branch on the backend.

| Feature | WebGPU | WebGL2 fallback |
| --- | --- | --- |
| Particle simulation | Compute kernel, 64-wide workgroups, ping-pong with an explicit parity blit | Transform feedback: one `POINTS` draw per frame into three separate buffers (position, velocity, meta) |
| Sim force model | Full model: soft acceleration limiters, metered respawn drizzle, radial guards, beat/band coupling, up to 16 Julia iterations | Same terms with fixed gains (`JULIA_GAIN 0.06`, `BULB_GAIN 0.85`, `THOMAS_GAIN 0.15`, `CURL_SCALE 0.09`), simple `age > lifespan` respawn, no soft velocity ceiling, no radial guards |
| Fractal mapping | `power = 2.35 + 4.15·dim`, warp `0.22 + 0.63·dim`, iterations `4 + 12·dim + 3·mid` | `power = 2.4 + 4.1·dim`, warp `0.25 + 0.65·dim`, `escape = 2.5 + 3.5·dim`, iterations from `DeviceProfile.juliaIterations` |
| Per-particle colour | Simulated in the compute kernel, so it can evolve | Static birth colour buffer; only size and palette grading evolve |
| 4D rotation and slice | Composed SO(4) matrix plus a hyperplane perspective divide on `w` | The same six-plane composition; `w` becomes a `z` offset plus a relative `w` term rather than a perspective divide |
| Multiplicity scatter | Up to ×8, all copies inside the same instanced draw | Up to ×4, one draw per scatter copy with the copy index passed as a uniform |
| Super-sampling | `DeviceProfile.supersample` is unused | Scene rendered at `supersample` × the draw size (1.4 high, 1.6 ultra) |
| Bloom | 2–8 levels from the profile, 13-tap Karis downsample, 25-tap tent additive upsample | Up to 6 levels, the same 13-tap downsample, 9-tap tent upsample scaled by `1 + bloomRadius` |
| Depth of field | 3×3 max-CoC tile + 13-tap golden-angle gather | Real depth prepass (`depthFunc LESS`, colour writes masked) → 4-tap CoC pack → 12-tap golden-angle gather weighted by the CoC difference |
| Volumetric god rays | 24 radial taps toward the frame centre, biased by scene luma | 24 taps toward the projected origin into a dedicated target, with a distance mask |
| Trails | Fade pass over the history buffer, then particles load and add | Trail shader writes the previous scene × `trails` into the cleared scene buffer, then `blitFramebuffer` scene → trail |
| Chromatic aberration, grain, vignette, dither | Full chain in the composite, 1-LSB Bayer + gradient dither | Same features; the dither is interleaved gradient noise only |
| Tone mapping | `mix(acesApprox, acesFilm, 0.35)` on HDR values | `acesFilm`, or clamped output when the pipeline degraded to RGBA8 (`uAces = 0`) |
| HDR targets | Always `rgba16float` + `depth32float` | `RGBA16F` when `EXT_color_buffer_float` (or `EXT_color_buffer_half_float`) is present, otherwise `RGBA8` with the `no-float-hdr` downgrade, `c/(1+c)` compression and `ONE_MINUS_SRC_COLOR` blending |
| Shockwaves | 8 shells, band `0.30 + 0.72·age`, decay `exp(−1.45·age)`, 2.4 s lifetime | 8 shells, `SHOCK_LIFE 1.9 s`, band `0.32 + 0.5·age`, decay `exp(−1.6·age)`; also fires its own shocks on beats |
| Adaptive quality loop | Resolution → post effects → particle count | None; quality changes only when the panel's Resolution slider does (`setQualityScale`) |
| GPU timing | `timestamp-query` when available | Not supported; `gpuMs` stays 0 and the HUD shows the frame time instead |
| PNG capture | `canvas.toBlob()` on the WebGPU canvas | `canvas.toBlob()` with `preserveDrawingBuffer: true` |
| Reported `kind` | `'webgpu'` | `'webgl2'` |

Honestly stated, the fallback loses: per-particle colour evolution; the metered respawn model, radial guards and soft velocity ceiling, so motion is a little less controlled; multiplicity depth (×4 instead of ×8) and bloom depth (6 levels instead of 8); the automatic quality control loop; and, on drivers without float render targets, the entire HDR chain — everything then runs in 8-bit with tone-curve compression, which visibly clips highlights. It also has no timestamp equivalent, so its GPU cost is only inferred from the frame time. What it gains in exchange is robustness: it needs only WebGL2 plus transform feedback, and its uniform upload is the same 448/128-byte writer the WebGPU path uses.

With neither backend, `bootstrapRenderer` throws an aggregated message (the WebGPU reason plus the WebGL2 reason), `Aetheria.showFatal` writes it into `#boot-error`, removes the `gone` class from the boot overlay so it becomes visible again, and logs `[aetheria]` to the console. The render loop is never started, so nothing else runs. With JavaScript disabled entirely, the page's `<noscript>` block renders instead. `BackendKind` includes `'none'` for exactly this state.

### Verifying the fallback

The fallback is the one code path a normal session never touches, so it has a switch: **`?renderer=webgl2`** forces it on a machine that has WebGPU, and `?renderer=webgpu` pins the primary path. `tools/compile-glsl.mjs` compiles all sixteen GLSL sources through a real `WebGL2RenderingContext` and prints every info log at once, which is far faster than discovering one failure per boot attempt.

Four defects were found and fixed this way. They are recorded because each is a trap worth knowing about:

- **`active` is a reserved word in GLSL ES 3.00.** Using it as a loop bound made the whole simulation vertex shader fail to compile. Renamed to `nShock`.
- **Two interface blocks may not share an instance name.** `SimBlock` and `PostBlock` were both instanced as `U`, so any shader including both failed with *"redefinition of an interface block instance name"*. `PostBlock` is now `P`.
- **The composite needs `GLSL_COLOR` and `GLSL_HASH`, not just `POST_BLOCK_GLSL`.** The latter supplies uniforms only; `luma`, `acesFilm`, `srgbEncode`, `ign` and `TAU` live in the other two blocks.
- **Unbounded frame-history feedback.** The trail shader multiplied the previous scene by a flat `trails` value. The loop's gain stayed above the emission feeding it, so the history buffer climbed without bound; every later stage then saw a saturated buffer, and *the exposure control stopped having any effect at all*. The fix is the same luma-weighted decay the WebGPU fade pass uses, plus a hard ceiling of `8.0` on the history buffer. This was isolated by disabling crops: trails off dropped the centre luma from 255 to 171 for identical particles.

The WebGL2 path was then verified end to end at 1920 × 1080 with 1.5M particles: 0 console errors, 0% clipped pixels, mean luma 101.5 and peak 187, and a mean colour of 108/98/120 — the violet cast of the palette, from a shader chain that is genuinely independent of the WebGPU one.

## Accessibility

- **Reduced motion.** `prefers-reduced-motion: reduce` is honoured twice. In CSS, the panel, toast, boot and hint transitions are cut to 1 ms and the boot bar's sweep animation is removed. In `main.ts`, the constructor clamps `trails ≤ 0.34`, `dof ≤ 0.25`, `chroma ≤ 0.18` and `volumetric ≤ 0.18`, and `GpuRenderer` disables idle auto-rotation (`idleAutoRotate = !reducedMotion`) so the camera moves only when the user moves it. A `change` listener on the media query reports the switch in a toast.
- **Keyboard operability.** Every verb is reachable without a pointer: rotation (`WASD`/arrows, with `Alt` plane overrides), zoom and slice (`+`/`−`, `Q`/`E`), pause (`Space`), reset (`R`), UI toggle (`H`), PNG (`S`), recording (`V`), plane locks (`X`/`Y`/`Z`) and help (`?`/`F1`). The panel is built from real `<input type="range">` and `<button>` elements with visible `:focus-visible` outlines (2 px `--accent`, 2–3 px offset), and it restores to full opacity on `focus-within`. Keys are ignored while a form field has focus.
- **ARIA.** The canvas is `role="img"` with `tabindex="0"` and a descriptive `aria-label`; the boot overlay is `role="status" aria-live="polite"`; the panel is `role="complementary"` with `aria-label="Aetheria controls"`; the stats list is labelled "Performance"; toasts are `role="status" aria-live="polite"`; toggle buttons carry `aria-pressed` (pause, audio source, recording, gyro, focus, hide-UI); the hide-UI icon button's `aria-label` flips between "Hide interface" and "Show interface"; the hint strip is `aria-hidden="true"` because a screen-reader-only paragraph describes the same interactions in prose. Each slider wires a `<label>`, an `aria-label` and an `<output htmlFor>` to its input id.
- **Contrast.** The interface is dark by design (`#04060d` page, `rgba(14,16,30,0.52)` glass panes) with `#e8ecff` primary text, `#97a3cc` / `#6b77a3` secondary text and a hairline `rgba(150,180,255,0.16)` border. Text sits on a blurred, dimmed pane rather than directly on the nebula, and readouts use a monospace face with tabular numerals so a changing value cannot reflow its own label. There is no light theme, and `color-scheme: dark` is declared.
- **Audio permission flow.** Audio starts off. The `Mic` button is the only path that calls `getUserMedia`, and it runs from the click handler (a user gesture), so the browser prompt appears in context. A denial is reported as `Microphone unavailable: <message>` in a toast, the audio state stays `off`, and the visuals keep running. The `Drone` button needs no permission at all and is the documented alternative. No audio leaves the machine, and the drone is never routed to the speakers.

## Verified performance report

The full machine-generated report lives in [`perf/report.md`](perf/report.md); it is regenerated by the runner below and is the authoritative record. This section summarises what was measured and states plainly which numbers to trust.

```bash
pnpm build
pnpm preview --port 4173 &
node tools/verify.mjs --url=http://127.0.0.1:4173/ --seconds=10   # or: pnpm verify
```

### What the runner measures

The runner launches **headless Chrome** (`--headless=new`, `--enable-unsafe-webgpu`, `--ignore-gpu-blocklist`, `--mute-audio`, a fixed viewport), waits for the boot overlay to clear, and captures:

- **Console, page and network diagnostics** — every `console` message is classified, and `pageerror` plus failed requests count as errors. WGSL diagnostics surface here because `compileChecked` logs them.
- **Startup** — navigation to the boot overlay receiving `gone`, i.e. "the renderer is constructed and the first frame is up".
- **Frame-time distribution** — in-page `requestAnimationFrame` deltas, reported as mean/p50/p95/worst and effective FPS rather than one flattering average.
- **Live HUD telemetry** read out of the DOM (`#stat-fps`, `#stat-gpu`, `#stat-cpu`, `#stat-particles`, `#stat-scale`, `#stat-memory`), the canvas backing-store size, and the control count.
- **Frame content** — read back **off the GPU** with `copyTextureToBuffer` (see the note below), summarised as mean luma, peak luma, lit and clipped ratios, mean RGB, and the luma spread over a 40 × 18 tile grid.

It writes `perf/report.json`, `perf/report.md` and `perf/gpu-frame.png`, and exits non-zero if the frame is essentially black, if any console or page error occurred, or if the boot overlay never cleared. `puppeteer-core` is a declared dev dependency.

### Why the frame is read back on the GPU

A headless Chrome compositor does not present a WebGPU canvas to `page.screenshot()`, and `canvas.getContext('2d').drawImage(webgpuCanvas, …)` returns empty pixels. Both produce a *blank* image that looks exactly like a renderer that never drew anything. The runner therefore arms an in-page capture that copies the swap-chain texture with `copyTextureToBuffer` immediately after the frame's own submission. That is why `perf/gpu-frame.png` is the file to inspect and the four `perf/0*.png` screenshots are only useful for checking layout and chrome.

### Where the frame time actually goes

Attribution has to be measured, so `tools/profile-gpu.mjs` brackets **every pass** in
the frame with GPU timestamps and prints the breakdown. On the reference machine the
result was not what the design assumed:

```
pass                       ms   share
particles               48.93   99.3%
simulate                 0.12    0.2%
composite                0.06    0.1%
fxaa                     0.05    0.1%
fade-scene               0.04    0.1%
bloom-* (9 passes)       ~0.05    0.1%
dof-* / godray           ~0.15    0.3%
```

The whole hand-written post chain — thirteen bloom levels, volumetric streaks,
depth-of-field, chromatic aberration, ACES, grain, vignette, FXAA — costs **under
1 ms combined**. The simulation of a million 4D particles costs **0.12 ms**. The
particle rasterisation pass was **99% of the frame**, and that is what the
performance work targeted.

Two things came out of that, both measured by ablation rather than reasoned about:

- **An integer modulo per vertex.** The shader mapped instances to particles with
  `inst % count`. Unsigned integer modulo is not a hardware instruction on current
  GPUs; it expands into a long reciprocal-multiply sequence, and it ran once per
  vertex — millions of times per frame. Replacing it with an exact float division
  (`inst - count * floor(inst / count)`, exact because the count is under 2^24)
  is the single largest change in this codebase's history.
- **Sprite quads sized to a fixed multiple of the sprite.** The bounding quad was
  several times larger than the visible glow, so most fragments were rasterised and
  then discarded. Sizing the quad to the radius where the gaussian has actually
  decayed (`sqrt(-ln(T)/12)`) removed the wasted fill.

After those, a fill-rate ablation (sprite radius forced to about one pixel) still
cost 29 ms at 1.57M drawn particles — that residue is per-*instance* vertex work,
which does not care about sprite size at all. Both terms scale with the drawn count,
and fill additionally scales with radius squared. That is why the particle budget is
now **derived from the sprite radius** rather than fixed: larger sprites merge into
gas at a lower count, smaller ones form dense star fields at a higher count, and the
frame cost stays bounded either way. When the budget declines to draw everything the
user asked for, the HUD says so rather than silently looking like a broken slider.

The measured cost is roughly **55 ns per drawn particle** through a headless
compositor, which is the figure the budget's constant is fitted to. Treat it as a
floor: a foreground window with a real swap chain should sustain more, and the SHOW
PARTICLES slider takes it higher by choice.

### Measured results

Measured on the reference machine this repository was developed on — NVIDIA RTX A5000, 16 logical cores, Chrome 154, Windows, 1920 × 1080 viewport:

| Metric | Value |
| --- | --- |
| Backend selected | WebGPU (ultra tier) |
| Time to first rendered frame | **~1.05 s** |
| Shader compilation errors | **0** (across all 14 WGSL modules) |
| Console / page errors | **0** |
| Captured frame, mean luma | 11.2 / 255 |
| Captured frame, peak luma | 217 / 255 |
| Pixels above 6/255 | 63.9% |
| Pixels fully clipped (above 242/255) | **0.0%** |
| GPU time, whole frame (in-app timestamps) | **19–20 ms** at the shipped defaults |
| GPU time, particle pass | 19 ms of that |
| GPU time, simulation | 0.7 ms |
| GPU time, entire post chain | 0.8 ms |
| Estimated VRAM | 218–240 MB |

The two numbers worth reading closely are **zero clipped pixels with a peak of 217**:
the nebula reaches near-white in its filament cores while nothing in the frame blows
out, which is the whole point of grading in HDR and tone mapping last. And the
**particle pass share** is where any future optimisation should go — everything else
in the frame is already under a millisecond.

### About the FPS figures

`perf/report.md` also reports an rAF-derived frame time, and in headless that figure reads around 23–60 ms. **Treat it as a lower bound, not a score** — and, for the same reason, treat the GPU figures above as a floor too. Headless Chrome composites every frame to a virtual display and applies backpressure to `requestAnimationFrame` that has nothing to do with GPU cost; the in-app GPU timers report the stages directly, and the discrepancy is measurable — the in-app counter reads 56–57 FPS while the runner's rAF sampler reads 33 FPS *for the same frames*.

This is also why the adaptive controller keys off GPU timestamps rather than rAF deltas: with rAF as the signal, a throttled compositor made the renderer shed quality for reasons that had nothing to do with it, and it degraded to 8% of the particle budget while the GPU stage was using a quarter of the frame. See `GpuRenderer.adapt()` for the signal selection and the reasoning.

**Honest statement of scope.** This was verified on one machine, in headless Chrome, and the per-particle cost quoted above was measured in that environment. The renderer's instrumentation now reports every pass separately, so the breakdown can be reproduced anywhere with `pnpm profile`. It has **not** been verified on a mid-range laptop GPU, on Apple silicon, or on a physical phone, and the mobile tier's 30 FPS target is a design intent, not a measurement.

## Project layout

```text
Aetheria/
├── index.html                 single page: canvas, boot overlay, ARIA prose, hint strip
├── package.json               scripts + dev-only dependencies (no runtime deps)
├── tsconfig.json              strict, ES2022, @webgpu/types, noEmit
├── vite.config.ts             single self-contained bundle, base './', dev 5173 / preview 4173
├── src/
│   ├── main.ts                bootstrap, backend choice, frame loop, URL hash, PNG/WebM export
│   ├── core/
│   │   ├── config.ts          device profiles, Params + ranges, memory constants
│   │   ├── math4d.ts          SO(4) rotations, MVP, inverse, [0,1] depth conventions
│   │   ├── types.ts           Renderer/HudStats contracts, memory estimator
│   │   └── uniforms.ts        SimWriter/PostWriter, byte offsets, default states
│   ├── gpu/
│   │   ├── device.ts          adapter/device bring-up, limit fallbacks, WGSL diagnostics
│   │   ├── bridge.ts          re-exports the WebGL2 side is allowed to import
│   │   ├── particles.ts       ping-pong buffers, seed/simulate/copy, shocks
│   │   ├── post.ts            post targets, layouts, pipelines, pass order
│   │   ├── renderer.ts        frame orchestration, camera, timestamps, adaptive quality
│   │   └── wgsl/
│   │       ├── common.wgsl.ts    uniform layouts, hash, palette, SO(4), curl, ACES
│   │       ├── simulate.wgsl.ts  simulate / seed / copy kernels
│   │       ├── particle.wgsl.ts  billboard vertex + fragment
│   │       └── post.wgsl.ts      one module per post pass
│   ├── webgl2/
│   │   ├── glutil.ts          GL helpers, float-colour-buffer probe, device label
│   │   ├── shaders.ts         GLSL ES 3.00 ports of the shared vocabulary
│   │   └── renderer.ts        transform-feedback fallback renderer
│   ├── audio/analysis.ts      analyser bands, adaptive beat detector, procedural drone
│   ├── input/controls.ts      pointer/touch/wheel/keyboard/gyro → six verbs
│   └── ui/
│       ├── panel.ts           sliders, stats, buttons, toasts, idle fade
│       └── styles.css         glass UI, focus rings, sr-only, reduced motion
└── tools/
    ├── verify.mjs                 headless verification runner (§12)
    ├── profile-gpu.mjs            per-pass GPU breakdown and ablations
    ├── tune-visual.mjs            visual calibration loop
    ├── compile-glsl.mjs           compiles every GLSL source and prints all logs
    └── check-docs.mjs             cross-checks README paths, scripts and deps
```

## Troubleshooting

**"WebGPU is not available in this browser", or the app silently runs the fallback.** `navigator.gpu` is undefined. Use Chrome or Edge 113+ on desktop, or Safari 18+ where available. When WebGL2 takes over, the toast reads `WebGPU unavailable: running the WebGL2 fallback` and the console logs `[aetheria] falling back to WebGL2: …` followed by the `BackendUnavailable` detail string.

**"No WebGPU adapter available".** The browser exposes WebGPU but no adapter could be created. The detail string names the usual causes: a blocklisted GPU, software rendering disabled, or a driver reset. `device.ts` retries `requestAdapter` three times (high-performance, then low-power, then `forceFallbackAdapter: true`) before giving up, so this message means even the software adapter was refused. Check `chrome://gpu` for blocklist status, and for a headless run pass the flags the verification runner uses (`--enable-unsafe-webgpu --ignore-gpu-blocklist --enable-gpu`). A related failure is `requestDevice() failed for every limit configuration`, which means the adapter refused the requested limits three times over (with limits, with default limits, then minimal).

**Shader compilation errors on startup.** `compileChecked` runs `getCompilationInfo()` on every module and prints `label line:column: message` for each diagnostic; those strings reach `showFatal`/`onError` and land in `#boot-error` on the boot overlay. A module that fails to compile therefore leaves a black canvas with a populated overlay rather than a silent failure.

**Microphone denied.** The Mic button reports `Microphone unavailable: <error message>` in a toast and leaves the audio source off; nothing else changes. Permission is requested only from that click, so a browser that remembers a denial keeps failing until the site permission is reset. Use the `Drone` button — it needs no permission and drives the same four signals.

**The screenshot is black.** The PNG button calls `canvas.toBlob()`: the WebGL2 path relies on `preserveDrawingBuffer: true`, the WebGPU path captures whatever the swap chain last presented, and `capture()` returns `null` with the toast `PNG export is not available on this backend` if neither is possible. An export taken before the first frame, while the boot overlay is still up, or from a background tab — where the loop returns early when `document.hidden` and nothing is recording — comes out empty. External screenshot tools need WebGPU enabled in the automation profile, otherwise they capture the WebGL2 path, or a black canvas if float colour buffers are missing there too. `tools/verify.mjs` detects a blank frame with the mean-luma and lit-pixel guards described in §12.

**Low frame rate.** Open the panel and read the backend badge: any `auto-degraded: …` entry means the control loop has already shed resolution, then DOF and god rays, then particles. In order of effect, reduce `Displayed` and `Particles`, lower `Trails` (history feedback keeps bright pixels alive for many frames) and `Volumetric` (24 texture taps per pixel), and watch `VRAM` — above 480 MB the readout turns red. Remember that the drawing buffer is `cssSize × min(devicePixelRatio, mobile ? 2.5 : 2) × maxPostScale × resolution`, so a 2× DPR display plus a `resolution` slider above 1 multiplies quickly. On machines without `timestamp-query`, the GPU readout falls back to the frame time, so judge by FPS and Scale instead.
