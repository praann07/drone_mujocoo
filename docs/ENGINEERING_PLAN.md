# Engineering Plan — ordered task breakdown per stage

Each task is tagged `[Stage-X]`. Do not begin a stage's tasks until the
prior stage's gate is confirmed passed by the user (see `../CLAUDE.md`).

## Stage A — Simulation & excitation

**Definition of done:** Full nonlinear 6-DOF MuJoCo quadrotor model exists
and runs; open-loop PRBS/chirp excitation trials logged per axis (roll,
pitch, yaw, combined); response plots reviewed and show rich
oscillation/decay across varied frequency content on every excited axis —
not a flat near-zero signal. User has explicitly confirmed the gate passed.

1. `[Stage-A]` Pin down MuJoCo model parameters (rotor config, mass,
   inertia, arm length, thrust/torque coefficients) — record in `TDD.md` §2.
2. `[Stage-A]` Build the MuJoCo XML (`01_simulation/models/`) — 4
   rotors, full 6-DOF free joint, realistic mass/inertia.
3. `[Stage-A]` Implement the sim driver: step the model, apply per-rotor
   thrust commands, log full state `x = [p, v, q, ω]` and control `u` at
   the fixed sample rate.
4. `[Stage-A]` Implement PRBS and chirp excitation generators, per-axis and
   combined, at multiple amplitude/frequency settings — enough distinct
   trials to support the Stage B/C held-out split (see `DATA_MODEL.md`).
5. `[Stage-A]` Run all excitation trials, log to `data/raw/` per the schema
   in `DATA_MODEL.md`.
6. `[Stage-A]` Plot every trial's response (attitude, angular velocity) and
   visually confirm rich oscillation/decay, not flat/degenerate signals.
7. `[Stage-A]` **Gate check:** present plots to user, get explicit
   confirmation the gate is passed. Update `../CLAUDE.md` stage status.

## Stage B — System identification

**Definition of done:** SINDy and DMDc both fit on the same excitation
dataset, using the quaternion constraint-handling strategy and DMDc regime
fixed in `TDD.md`. One-step-ahead prediction error on genuinely held-out
trials is below a stated threshold for both models. No rollout attempted
yet. User has explicitly confirmed the gate passed.

1. `[Stage-B]` Implement Savitzky-Golay smoothing + differentiation
   pipeline (`02_identification/preprocessing.py`), applied before any
   derivative estimate feeds either fit.
2. `[Stage-B]` Define and lock the fit/held-out trial split per
   `DATA_MODEL.md` §4 (trial-level, never a time-slice).
3. `[Stage-B]` Implement the rotation-vector local-coordinate transform
   (TDD.md §4) with reset-on-margin logic and reset logging.
4. `[Stage-B]` Fit SINDy with the physically-motivated library from
   TDD.md §4, STLSQ sparsity regression.
5. `[Stage-B]` Fit DMDc restricted to the near-hover-trim trial subset per
   TDD.md §5.
6. `[Stage-B]` Compute one-step-ahead prediction error on held-out trials
   for both models; set/confirm the numeric threshold in `TDD.md` §9 based
   on observed noise floor.
7. `[Stage-B]` **Gate check:** report one-step-ahead error table to user,
   get explicit confirmation the gate is passed. Update `../CLAUDE.md`.

## Stage C — Validation & analysis

**Definition of done:** Full validation protocol from `TDD.md` §6 executed
in order (one-step-ahead already passed at Stage B → short-horizon →
long-horizon → eigenvalue analysis → sparsity ablation → noise ablation →
head-to-head comparison → model selection). Every plot traceable to a
Stage-B-passing model. User has explicitly confirmed the gate passed and
agrees with the model selection.

1. `[Stage-C]` Short-horizon (5–10s) open-loop rollout on held-out trials,
   both models, plot predicted-vs-true.
2. `[Stage-C]` Long-horizon rollout, report the timestep where error
   crosses a stated divergence bound, both models.
3. `[Stage-C]` Eigenvalue/pole-spectrum analysis: DMDc `A` matrix directly;
   SINDy linearized at near-hover for a fair comparison.
4. `[Stage-C]` Sparsity-vs-error ablation for SINDy (sweep STLSQ threshold,
   plot one-step-ahead/short-horizon error vs. active-term count).
5. `[Stage-C]` Noise-level-vs-error ablation: synthetic noise injection,
   refit, report degradation for both models.
6. `[Stage-C]` Head-to-head SINDy-vs-DMDc report with DMDc plots labeled
   with its near-hover validity restriction.
7. `[Stage-C]` Write model-selection justification; update `TDD.md` §9 and
   `../CLAUDE.md` with the selected model.
8. `[Stage-C]` **Gate check:** present full validation report to user, get
   explicit confirmation. Update `../CLAUDE.md`.

## Stage D — Inner attitude loop (data-driven) + outer position/trajectory loop (standard)

**Definition of done:** Fresk-style quaternion P² inner-loop controller
designed against the Stage-C-selected identified model, verified against a
battery of representative setpoint trajectories (not just static targets).
A standard PD/PID outer position loop plus a point-to-point trajectory
planner (§TDD.md §7b) produces the setpoint stream for that inner loop and
tracks all 6 voice-command waypoints. Sim-to-sim gap check passed for the
inner loop; cascade stability confirmed once both loops are tuned together.
User has explicitly confirmed the gate passed.

1. `[Stage-D]` Implement the quaternion P² control law
   (`04_control/controller.py`) parameterized by the identified
   model's dynamics for gain selection.
2. `[Stage-D]` Tune inner-loop gains against the identified model; verify
   stabilization and tracking against representative step/ramp attitude
   setpoint trajectories (TDD.md §7).
3. `[Stage-D]` Run the same inner-loop controller/gains against ground-truth
   MuJoCo physics (sim-to-sim gap check, inner loop only); compute settling
   time, overshoot, steady-state error for both; set/confirm tolerance in
   `TDD.md` §9.
4. `[Stage-D]` Implement the point-to-point trajectory planner
   (`04_control/trajectory.py`) with a smooth velocity profile; set
   max velocity/acceleration in `TDD.md` §9.
5. `[Stage-D]` Implement the outer position/velocity PD(I) controller
   (`04_control/position_controller.py`), producing
   `(q_desired, thrust_desired)` for the inner loop, using the standard
   small-angle thrust-vector relation.
6. `[Stage-D]` Define the 6 voice-command waypoint offsets and position
   tolerance; record in `TDD.md` §7b/§9.
7. `[Stage-D]` Cascade the two loops on the identified model; verify
   point-to-point tracking for all 6 waypoints; explicitly check outer-loop
   bandwidth stays slower than inner-loop bandwidth (TDD.md §7b cascade
   stability check) — if not, first reduce trajectory planner limits, never
   touch Stage B/C to fix a controller-tuning problem.
8. `[Stage-D]` Cascade the two loops driving ground-truth MuJoCo physics;
   compute waypoint position-tracking metrics; compare against the
   identified-model-driven cascade.
9. `[Stage-D]` **Gate check:** present inner-loop and outer-loop comparison
   metrics to user, get explicit confirmation. Update `../CLAUDE.md`.

## Stage E — Voice interface & live 3D navigation demo

**Definition of done:** Full pipeline (VAD → ASR → classifier → waypoint
map → trajectory planner → outer loop → Stage D inner loop → live 3D
MuJoCo render + telemetry panel) working end-to-end, showing the vehicle
visibly fly point-to-point to each commanded waypoint. Success rate
measured over N ≥ 20 trials with latency breakdown, reported to user.

1. `[Stage-E]` Implement VAD + ASR capture (`sounddevice` mic input),
   choose vosk vs faster-whisper per measured latency/accuracy; record
   choice in `TDD.md` §9.
2. `[Stage-E]` Implement rule-based command classifier for the 6-word
   vocabulary.
3. `[Stage-E]` Wire classifier output → fixed waypoint map → trajectory
   planner → outer loop → Stage D inner loop (never directly to
   ground-truth sim, per `../CLAUDE.md` rules).
4. `[Stage-E]` Build live 3D render (MuJoCo viewer) + telemetry panel
   (quaternion state, position, attitude error, position error, latency
   breakdown).
5. `[Stage-E]` Run N ≥ 20 end-to-end trials across all 6 commands, log
   transcript, confidence, latency breakdown, dispatched command, target
   waypoint, achieved position/attitude per `DATA_MODEL.md` §5.
6. `[Stage-E]` Compute and report success rate + latency breakdown.
7. `[Stage-E]` **Gate check:** present results to user. Update
   `../CLAUDE.md`.
