# Technical Design Document — Voice-Controlled Quadrotor Attitude System

Status: Stage A parameters locked 2026-08-24 (see §9 log). Remaining
placeholders are Stage B/D/E decisions that genuinely can't be fixed until
those stages' data/tuning exists.

## 1. Conventions (must be fixed before any code is written)

- **Units:** SI throughout — meters, kilograms, seconds, radians, Newtons,
  Newton-meters, rad/s.
- **Coordinate frame — revised 2026-08-24:** single consistent convention,
  **ENU world / FLU body** (Forward-Left-Up), not NED/FRD. World frame is
  MuJoCo's native right-handed frame with +Z up and gravity along -Z (no
  conversion needed — this is MuJoCo's default). Body frame is also
  Z-up-aligned: body +X forward, +Y left, +Z up (matches ROS REP-103/105
  convention, chosen specifically because it avoids the alternative: an
  NED-world/FRD-body or ENU-world/FRD-body pairing would force the drone's
  hover attitude to be a 180° rotation away from the identity quaternion —
  a permanent, easy-to-get-backwards flip baked into every frame
  conversion. With ENU/FLU, **identity quaternion = level hover**, thrust
  acts along +Z body = +Z world when level, and no NED↔ENU or FRD↔FLU
  conversion utility is needed anywhere in this project. This single-frame
  choice is deliberate risk reduction, not a default left unexamined.
- **Quaternion convention:** Hamilton, scalar-first: `q = [w, x, y, z]`,
  body-to-world rotation. State this convention in every function docstring
  that touches a quaternion.
- **Sample rate: 500 Hz** simulation/control/log rate (`dt = 0.002 s`).
  Chosen over the original 200 Hz starting assumption because Stage B's
  Savitzky-Golay differentiation benefits from headroom well above the
  excitation band (§3: content up to ~15 Hz) — 500 Hz gives a Nyquist limit
  of 250 Hz, >15x the highest excited frequency, keeping differentiation
  noise low without oversampling to the point of wasting storage.

## 2. MuJoCo model

Built fresh for this project (per 2026-08-24 decision — the prior attempt's
`quad.xml`/`world.xml` in `DRONEPROJECT_AFTERREVIEW 1/lwstack/models/` are
**not** reused, to avoid silently inheriting that attempt's assumptions).

- Rigid body with 4 rotors in an **X configuration** (chosen over + because
  X gives every rotor authority over both roll and pitch simultaneously,
  which is what "left/right/forward/back" tilt commands actually need, and
  it's the more common real-world layout).
- State: full 6-DOF — position `p ∈ R³`, velocity `v ∈ R³`, attitude
  quaternion `q ∈ R⁴` (`‖q‖=1`), body angular velocity `ω ∈ R³`. 13-element
  state vector `x = [p, v, q, ω]`.
- Control input: 4 individual rotor thrusts — `u ∈ R⁴`, each `u_i ∈ [0,
  u_max]`. This is the actuation level at which excitation is injected and
  at which the identified model and Stage D controller operate; total
  thrust/torque `[T, τx, τy, τz]` is a linear remap of `u` via the mixer
  matrix (below) and is used only for human-readable plots, not for the
  model fit itself.

**Physical parameters (self-consistent, not copied from a specific real
drone's published spec sheet — chosen to be a plausible small-quad class
and derived from a simple lumped point-mass model so the inertia is
consistent with the mass and geometry, not an arbitrary guess):**

| Parameter | Value | Derivation |
|---|---|---|
| Total mass `m` | 1.0 kg | Central body 0.7 kg (treated as a point at the CG) + 4 × 0.075 kg motor/rotor point masses |
| Arm length `l` (center to rotor) | 0.15 m | Chosen for a compact ~1 kg class frame |
| Rotor position (body FLU, X config) | `(±l/√2, ±l/√2, 0)` ≈ `(±0.106, ±0.106, 0)` m | Standard X-frame geometry |
| `Ixx = Iyy` | 3.37 × 10⁻³ kg·m² | `Σ m_rotor · y_i²` over the 4 lumped rotor masses |
| `Izz` | 6.74 × 10⁻³ kg·m² | `Σ m_rotor · (x_i² + y_i²)` — comes out ≈ 2×`Ixx`, consistent with typical flat-quad geometry |
| Hover thrust (total) | 9.81 N (`m·g`) | ≈ 2.45 N per rotor at hover |
| Max thrust per rotor `u_max` | 6.0 N | Thrust-to-weight ratio ≈ 2.45, enough control authority for aggressive excitation without being unrealistic |
| Rotor torque coefficient `k_m` | 0.02 m | Reaction yaw torque per rotor: `τ_i = ±k_m · u_i` (sign per spin direction) |
| Rotational drag `c_rot` | 0.008 N·m·s/rad | Applied torque `-c_rot·ω` (body frame) |
| Translational drag `c_trans` | 0.1 N·s/m | Applied force `-c_trans·v` (world frame) |

**Aerodynamic drag — added 2026-08-24, not optional.** The rigid body
alone (rotor thrust + gravity, no drag) is a pure undamped double
integrator in rotation: an early smoke test showed a 40%-hover-amplitude
chirp reaching 840°/s angular velocity and still climbing after 2s of a
planned 20s trial, because any sustained low-frequency torque component
has nothing to bound it. Two problems follow from that: (1) it violates
the Stage A gate's own language — "oscillation/**decay**" implies bounded
behavior, not monotonic runaway — and (2) it gives SINDy nothing to
discover in Stage B, since a plain double integrator has no time-constant
behavior. Linear viscous drag (`-c_rot·ω`, `-c_trans·v`) is applied as an
explicit generalized force each simulation step (implemented in
`01_simulation/sim_driver.py`, not in the MJCF, since a
freejoint's single `damping` attribute can't give translation and rotation
different time constants). `c_rot` is chosen for an ~0.4–0.8s rotational
time constant — a few oscillation cycles visible within a 20s trial.
**Consequence for Stage B:** the SINDy function library (§4 below) must
include a linear-in-`ω` damping term, or the fit will systematically
underpredict decay and one-step-ahead error will be spuriously inflated
by a missing-term bias rather than genuine model mismatch.

**Rotor numbering and spin directions (X config, viewed from above, body
+X = forward, +Y = left):**

| Rotor | Position | Spin | Diagonal pair |
|---|---|---|---|
| 1 | front-left `(+x,+y)` | CCW | pairs with 2 |
| 2 | back-right `(-x,-y)` | CCW | pairs with 1 |
| 3 | front-right `(+x,-y)` | CW | pairs with 4 |
| 4 | back-left `(-x,+y)` | CW | pairs with 3 |

Diagonal pairs share spin direction so equal thrust on all 4 rotors
produces zero net yaw torque, matching standard quadrotor convention. This
numbering is defined for this project only — it is not claimed to match
any specific real flight controller's motor-numbering convention.

**Mixer matrix** (`u ∈ R⁴` rotor thrusts → `[T, τx, τy, τz]`, body FLU):

```
T   = u1 + u2 + u3 + u4
τx  = (l/√2) · ( u1 - u2 - u3 + u4)     (roll,  about body +X)
τy  = (l/√2) · (-u1 + u2 - u3 + u4)     (pitch, about body +Y)
τz  = k_m     · ( u1 + u2 - u3 - u4)     (yaw,   reaction torque, sign per spin table above)
```

Derived from `τ = r × F` per rotor (`F` along body +Z, `r` = rotor position
above): `τx_i = y_i·u_i`, `τy_i = -x_i·u_i`, summed over the 4 rotor
positions in the table above. Re-verified numerically against the actual
MJCF rotor site positions in `tests/test_mixer.py` before any excitation
trial is trusted — the derivation here is the design intent, the test is
what makes it load-bearing.

## 3. Excitation signal design

- Per-axis open-loop excitation: PRBS and chirp signals injected into
  differential rotor commands to excite roll, pitch, and yaw independently,
  plus a combined/mixed-axis run to capture cross-coupling.
- No closed-loop controller active during any excitation trial — this is
  what keeps Stage A data usable for identification (a closed loop would
  correlate control input with state error and bias the fit).
- Multiple independent trials per axis at different amplitude/frequency
  settings, so later trials can be **held out entirely** for Stage B/C
  (see `DATA_MODEL.md` §4 for the exact split).
- **Frequency range: 0.1–15 Hz.** Rationale: the vehicle is open-loop
  (free joint, no controller holding it), so there is no closed-loop
  bandwidth to target — instead the range is chosen to span well below and
  above the timescales a Stage D setpoint-tracking controller will
  eventually command (attitude step responses on a ~1 kg/~3×10⁻³–7×10⁻³
  kg·m² class vehicle typically settle over hundreds of ms to a couple of
  seconds, i.e. ~0.5–3 Hz) — 0.1 Hz captures slow/near-static behavior,
  15 Hz is a decade above that range with headroom before the 500 Hz
  sample rate's Nyquist limit (250 Hz) becomes a factor.
  - **Chirp:** linear frequency sweep 0.1 → 15 Hz over a 12 s trial (revised
    from an initial 20 s plan — see below).
  - **PRBS:** bit period 0.05 s (≈20 Hz fastest switching content) up to a
    max hold of 5 s (≈0.1 Hz slowest content), same 12 s trial length.
  - **Trial duration — revised to 12 s, not 20 s (2026-08-24):** with drag
    added (§2) but no closed-loop stabilization, sustained rotation at even
    a few rad/s accumulates many full revolutions over 20 s, triggering
    frequent local-coordinate resets (§4) and unbounded position drift.
    Attitude identification is unaffected either way (Stage B fits `(θ,ω)`
    only — position is never a fit target — and resets are logged/handled
    explicitly per §4), but 12 s keeps reset frequency and position drift
    more manageable while still covering more than one full cycle of the
    0.1 Hz chirp start frequency.
  - **Per-axis amplitude calibration (2026-08-24):** amplitude is *not* a
    flat fraction of hover thrust across axes — roll/pitch torque comes
    from the full moment arm (`l/√2 ≈ 0.106 m`) while yaw only comes from
    the much smaller reaction-torque coefficient `k_m = 0.02`. A flat
    amplitude either saturates roll/pitch into repeated full inversions
    (an early attempt at ±40% hover amplitude produced peak angular
    accelerations of ~123 rad/s² — physically absurd — and altitude
    free-fall once the vehicle inverted and thrust reversed) or leaves yaw
    producing a barely-visible response. Each axis's base amplitude is
    calibrated so **peak angular acceleration ≈ 12 rad/s²** on every axis:
    roll/pitch base amplitude 0.10 N, yaw base amplitude 1.00 N (still
    well under `u_max=6.0 N` combined with hover's 2.45 N/rotor). See
    `01_simulation/excitation.py` `AXIS_BASE_AMPLITUDE_N`.

## 4. SINDy — quaternion constraint handling (decision, required before Stage B)

**Decision: option (b) — identify dynamics in a constraint-free local
attitude coordinate, not raw quaternion components.**

Specifically: at each excitation trial, track attitude as a **body-frame
rotation vector (axis-angle, `θ ∈ R³`)** relative to a per-trial reference
attitude, reset to a fresh reference (re-linearization point) whenever
`‖θ‖` approaches a chosen singularity margin (e.g. 150°, well short of the
180° rotation-vector singularity). SINDy is fit on `(θ, ω)` and their
derivatives, which live in an unconstrained `R⁶` — no post-hoc
renormalization hack, no constrained optimization needed, and no silent
"low one-step-ahead error while drifting off the unit sphere" failure mode.

- **Why not (a) renormalize after every step:** renormalizing hides
  constraint violation inside the rollout instead of preventing it in the
  fit; a model can still report deceptively low one-step-ahead error while
  its raw (pre-renormalization) prediction is inconsistent — exactly the
  degenerate-model failure mode this project is designed to catch.
- **Why not (c) constrain the fit directly:** constrained SINDy variants
  exist but add solver complexity and tuning surface without a clear
  benefit over working in an already-unconstrained coordinate; revisit only
  if (b) proves to have unacceptable reset-transient artifacts in practice.
- Reference-attitude resets are logged (trial id, reset timestamp, reason)
  so Stage C validation can account for them explicitly rather than let
  them appear as unexplained discontinuities in rollout plots.
- Function library: physically-motivated terms, not blind high-order
  polynomials — rigid-body terms (`ω × Iω` type cross/quadratic terms in
  `ω`), a **linear damping term in `ω`** (required — the ground-truth
  simulator applies `-c_rot·ω` drag per §2; omitting this term from the
  library guarantees a missing-term bias no amount of sparsity tuning can
  fix), linear/low-order terms in `θ`, and terms linear in control input
  `u` (and, if needed, low-order `θ·u` / `ω·u` cross terms for
  gyroscopic/control coupling). Blind polynomial libraries up to degree 3+
  are avoided as a default — they overfit small excitation datasets and
  produce numerically fragile rollouts.
- Sparsity: STLSQ (or equivalent thresholded regression) with a sparsity
  sweep — see `ENGINEERING_PLAN.md` Stage C task for the ablation that
  reports one-step-ahead error vs. active-term count, so the final sparsity
  level is a documented choice, not a default.

## 5. DMDc — regime (decision, required before Stage B)

**Decision: (a) near-hover-trim regime — plain linear DMDc restricted to
small perturbations around near-hover attitude, not the full nonlinear
excitation envelope.**

- DMDc fits a linear operator `x_{k+1} = A x_k + B u_k`. Full attitude
  dynamics are nonlinear (quaternion kinematics, `ω × Iω` term), so a plain
  linear fit is only physically meaningful in a small-perturbation regime.
- Excitation trials used for the DMDc fit/validation are restricted to
  amplitude ranges that keep `‖θ‖` (the same rotation-vector coordinate
  used for SINDy) within a small-angle bound — **[TO CONFIRM AT STAGE B
  START once Stage A data exists — pick a bound, e.g. ~15–20°, and verify
  it against actual linearization error]**.
- No EDMD/Koopman lift is implemented in this project (explicit choice —
  keeping DMDc simple lets the SINDy-vs-DMDc Stage C comparison isolate
  "sparse nonlinear identification" vs. "linear near-hover identification"
  as the actual variable, rather than confounding it with lift-basis
  design). Consequence, stated on every plot/report that uses DMDc: **DMDc's
  claimed validity is restricted to the near-hover trim regime** and it is
  not expected to match SINDy's accuracy at larger excitation amplitudes —
  this is an expected, reported result, not a failure.
- If Stage C shows DMDc's near-hover accuracy insufficient even for Stage D
  (which itself tracks small-perturbation setpoint steps from near-hover),
  SINDy becomes the Stage C-selected model by default; this is anticipated
  and acceptable per the model-selection step in Stage C, not a project
  blocker.

## 6. Validation protocol (Stage C — exact sequence, no reordering)

1. **One-step-ahead prediction error** on held-out trials (see
   `DATA_MODEL.md` §4 for the held-out split — separate excitation trials,
   never a time-slice of one continuous trajectory). Must pass the Stage B
   threshold before step 2 begins.
2. **Short-horizon open-loop rollout** (5–10 s), same held-out trials,
   reporting error growth over the horizon.
3. **Long-horizon open-loop rollout**, reporting the timestamp/step at
   which prediction error crosses a stated divergence bound.
4. **Eigenvalue/pole-spectrum analysis** of the identified linear part (DMDc
   `A` matrix directly; SINDy's local linearization at near-hover for a
   fair side-by-side).
5. **Sparsity-vs-error ablation** (SINDy): one-step-ahead and short-horizon
   error as a function of STLSQ threshold / active term count.
6. **Noise-level-vs-error ablation**: inject known synthetic noise levels
   into a copy of the excitation data, refit, report error degradation for
   both models.
7. **SINDy-vs-DMDc head-to-head**: side-by-side error tables and plots
   across steps 1–3, with DMDc plots explicitly labeled with its near-hover
   validity restriction (§5).
8. **Model selection**: written justification for which model (SINDy or
   DMDc) is carried forward as "the identified model" into Stage D.

No plot from this section may be generated for a model that has not already
passed the Stage B one-step-ahead gate — see `../CLAUDE.md` rules.

## 7. Stage D — inner loop: attitude controller against the identified model

- Controller family: singularity-free quaternion attitude control, Fresk-
  style P² law (proportional on quaternion error vector part, proportional
  on angular velocity error — same family as the CD5 course baseline).
- Designed/tuned using the Stage C-selected identified model's dynamics
  (its `A, B` for DMDc, or its fitted `ẋ = Θ(x)ξ` for SINDy) as the plant
  model for gain selection — not the ground-truth MuJoCo equations.
- Setpoint input: no longer a single fixed attitude target — this loop now
  receives a **continuously-updating attitude + thrust-magnitude setpoint**
  from the outer position loop (§7b) at each control step. The identified
  model is still exactly what the gains are designed and validated against;
  only the *source* of the setpoint changed (was: fixed per-command target,
  now: time-varying trajectory output).
- Verification order (unchanged from the original attitude-only design):
  (1) stabilizes and tracks a battery of representative attitude+thrust
  setpoint trajectories (not just static targets — must include a
  changing setpoint stream) when the controller's internal plant model *is*
  the identified model driving simulated response, (2) sim-to-sim gap
  check — same controller, same gains, now driving actual ground-truth
  MuJoCo physics — compare settling time / overshoot / steady-state error
  between the two, still using representative step and ramp setpoint
  trajectories, not the full outer loop yet (isolates the inner loop's own
  gap from the cascade's).

## 7b. Stage D — outer loop: position control & trajectory planning (standard physics, not identified)

This loop is deliberately **not** a target of SINDy/DMDc — see the
architecture decision in `PRD.md` §1. It uses known Newtonian mechanics:
horizontal/vertical acceleration is produced by the thrust vector's
direction and magnitude, so a small-angle approximation maps a desired
horizontal acceleration to a desired tilt (roll/pitch) — exactly the
standard cascaded-quadrotor-control structure, kept intentionally simple
so all identification rigor stays on the attitude loop.

- **Trajectory planner**: point-to-point, straight line between current
  position and the commanded waypoint, with a smooth (trapezoidal or
  minimum-jerk) velocity profile so the outer loop is never asked to track
  a position step discontinuity — position step-changes would demand
  attitude step-changes the inner loop wasn't validated against in Stage D
  §7. Planner parameters (max velocity, max acceleration):
  **[TO CONFIRM AT STAGE D START]**.
- **Position/velocity controller**: PD (or PID with a small integral term
  for steady hover-offset rejection) on position and velocity error,
  producing a desired horizontal acceleration and a desired vertical
  thrust. Desired horizontal acceleration is converted to a desired
  roll/pitch via the standard small-angle thrust-vector relation; desired
  yaw is held constant (yaw is not commanded by any of the 8 voice
  commands — see 2026-09-07 addition below). Output: `(q_desired,
  thrust_desired)` — exactly the setpoint format the Stage D §7 inner loop
  already consumes, unchanged.
- **Waypoint semantics for the 8 voice commands** (`up`/`down` added
  2026-09-07 for the Mission Control controller, `CLAUDE.md` "Stage E —
  Mission Control"; same world-frame-offset semantics as the original 6,
  just along +/-Z instead of horizontal, and a smaller 0.5m offset rather
  than the lateral 1.5m since hover altitude is only ~1.0m and the visual
  floor is non-collidable): `left`/`right`/`forward`/`back`/`up`/`down` =
  move a fixed offset distance (**[TO CONFIRM AT STAGE D START]**, e.g.
  1–2 m) from the vehicle's *current* position in that body-relative
  direction; `hover` = hold current position; `stop` = abort any
  in-progress trajectory and hold position at wherever the vehicle
  currently is (not a return to origin).
- **Cascade stability check**: because the outer loop's bandwidth must be
  slower than the inner (identified-model-driven) attitude loop's
  bandwidth for the cascade assumption to hold, this is checked explicitly
  once both loops are tuned — not assumed. If the identified model's
  attitude loop turns out too slow to support a responsive-feeling outer
  loop, the trajectory planner's velocity/acceleration limits are reduced
  first (cheap fix) before touching Stage B/C (never touch identification
  to fix a controller-tuning problem).

## 8. Stage E — voice pipeline

Blocks, in order:

1. **VAD** (voice activity detection) — trims silence, detects utterance
   boundaries before ASR runs.
2. **Offline ASR** — `vosk` (small model) or `faster-whisper` (tiny/base) —
   final choice: **[TO CONFIRM AT STAGE E START]**, both are Windows-wheel
   installable with no compiler; pick based on measured latency/accuracy
   once both are trialed on the 6-word command vocabulary.
3. **Command classifier** — rule-based keyword match against the 6-word
   vocabulary (left, right, forward, back, hover, stop) is sufficient given
   the fixed small vocabulary; a tiny classifier is only justified if
   keyword matching proves unreliable in practice.
4. **Fixed waypoint map** — command string → target waypoint (relative
   offset from current position, or hold), per §7b.
5. **Trajectory planner + outer loop (§7b)** — generates the time-varying
   attitude+thrust setpoint stream for the current waypoint.
6. **Stage D inner loop (§7)** — receives that setpoint stream, drives the
   identified model in closed loop, output attitude/position trajectory
   feeds the renderer.
7. **Live 3D render** — MuJoCo's built-in viewer/renderer draws the vehicle
   flying to the commanded waypoint each frame; a telemetry panel alongside
   shows quaternion state, position, attitude error, position error, and
   the per-stage latency breakdown (VAD, ASR, classification, trajectory
   generation, control-loop dispatch).

Microphone capture: `sounddevice` (PortAudio, Windows wheels available) —
not `pyaudio`.

## 9. Open decisions log

Track every `[TO CONFIRM ...]` placeholder above here as it gets resolved,
with the date and the value chosen, so the decision history is auditable:

| Decision | Resolved | Value |
|---|---|---|
| Coordinate/quaternion convention | 2026-08-24 | ENU world / FLU body (revised from initial NED/FRD sketch — see §1) |
| Rotor configuration (X vs +) | 2026-08-24 | X, numbering/spin per §2 table |
| Mass/inertia/arm length/thrust coefficients | 2026-08-24 | m=1.0 kg, l=0.15 m, Ixx=Iyy=3.37e-3, Izz=6.74e-3 kg·m², u_max=6.0 N/rotor, k_m=0.02 m — see §2 |
| Sample rate | 2026-08-24 | 500 Hz (`dt=0.002s`) |
| Excitation frequency range | 2026-08-24 | 0.1–15 Hz, chirp + PRBS, 20s trials — see §3 |
| DMDc near-hover angle bound | 2026-08-24 | 20° peak rotation, enforced by `assert_near_hover_bound()`; dedicated `near_hover` trial family added (worst actual 17.8°) |
| Stage B one-step-ahead error threshold | 2026-08-24 | step NRMSE < 0.10 (normalised by true per-step state *change*, trivial baseline = 1.0) **and** derivative R² > 0.99 |
| SINDy sparsity threshold | 2026-08-24 | 0.2 → 44 active terms of 156. Selected by parsimony rule (sparsest within 0.002 val-R² of best) on a validation split of the **fit** trials only |
| Stage D inner-loop setpoint tolerance | 2026-08-24 | 1° settle band; gains derived from SINDy hover-linearization (Kq=[29.0,29.0,29.0], Komega≈[3.0,3.0,4.2]) targeting 1.5s/ζ=0.7 — achieved faster (~0.56–0.86s) due to rotor saturation on large steps, still stable/monotonic in both SINDy- and MuJoCo-driven rollouts |
| Trajectory planner max velocity/acceleration | 2026-08-24 | 1.0 m/s, 1.0 m/s² |
| Voice-command waypoint offset distance | 2026-08-24 | 1.5 m, world-frame (forward=+X, back=-X, left=+Y, right=-Y) — not body-relative, since yaw is never commanded |
| Outer-loop position tolerance | 2026-08-24 | 0.15 m settle band; gains target 3.0s/ζ=1.0 assuming a pure double-integrator plant — actual cascade shows mild ~0.07m ringing (inner-loop lag + drag not in the design model), decaying by ~4s, well inside tolerance |
| ASR engine (vosk vs faster-whisper) | 2026-08-24 | **vosk** (small en-us model) chosen — already in `requirements.txt` and installed; offline, no GPU, ships Windows wheels. `faster-whisper` not installed. Vosk path coded in `05_voice_interface/voice_input.py`; headless trials use the `TextSource` fallback (no mic). Measured classify latency ~0.1 ms; end-to-end pipeline success 24/24 (100%) on N=24 scripted trials (see `data/processed/stage_e_voice_sessions/`). |
