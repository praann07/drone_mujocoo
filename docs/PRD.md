# Product Requirements Document — Voice-Controlled Quadrotor Navigation System (Data-Driven)

Course: Data Driven Control of Drones
Status: see [`../CLAUDE.md`](../CLAUDE.md) "Stage status" — the single
source of truth for what stage this project is on. Deliberately not
restated here: a status line duplicated across README.md/PRD.md/TDD.md
is a status line that goes stale in two of the three the moment progress
is made, which is exactly what happened before this line was rewritten.

## 1. Problem statement

Course projects for "Data Driven Control of Drones" typically demonstrate one
of two things: a voice/UI layer bolted onto a textbook-physics controller, or
a system-identification exercise with no live demo. Neither alone is a
genuine data-driven-control project. This project requires both, wired
correctly: an offline voice command drives a waypoint, an outer position/
trajectory loop (standard, known physics) converts that into a time-varying
attitude setpoint, and an inner quaternion controller **designed against a
dynamics model obtained purely from data** (SINDy and/or DMDc fit to
simulated excitation trials) tracks that setpoint, with the model's validity
established through a staged, gated protocol before it is ever allowed to
drive a controller or a demo.

**Architecture decision (2026-08-24):** position/trajectory control is a
**cascade**, not a joint identification target. The outer loop (position →
desired attitude + thrust) uses standard Newtonian physics — translational
dynamics under a thrust vector are a simple, well-understood double
integrator, so there is nothing meaningful for SINDy/DMDc to discover there.
All identification effort stays concentrated on the **attitude (rotational)
dynamics**, which is where the real nonlinearity and quaternion-constraint
complexity live, and which is the actual graded content for this course.
The inner attitude loop is unchanged from the original attitude-only design
— it just now receives a continuously-updating setpoint from the outer loop
instead of a single fixed target per command.

The central risk this project exists to manage: an identified model that is
subtly wrong (unconstrained quaternion drift, DMDc used outside its linear
validity regime, held-out data that isn't really held out) can still produce
a controller that *looks* fine in a live demo. The staged gates in
`../CLAUDE.md` exist specifically to prevent that from happening silently.

## 2. Goals

- Collect open-loop excitation data from a full nonlinear 6-DOF MuJoCo
  quadrotor model, rich enough to identify attitude dynamics across the
  operating envelope actually used by the controller.
- Identify attitude dynamics from that data using SINDy and DMDc, each with
  an explicit, justified strategy for handling the unit-quaternion
  constraint (SINDy) and the nonlinear/bilinear nature of attitude dynamics
  (DMDc regime).
- Validate both identified models with a fixed protocol — one-step-ahead →
  short-horizon rollout → long-horizon rollout — on genuinely held-out
  excitation trials, and select the better-validated model.
- Design a singularity-free quaternion attitude controller against the
  selected identified model, and confirm it still performs acceptably when
  applied to ground-truth MuJoCo physics (sim-to-sim gap check).
- Design a standard (non-identified) outer-loop position/trajectory
  controller that converts a target waypoint into a time-varying attitude +
  thrust setpoint trajectory for the Stage D inner loop, with a point-to-
  point trajectory planner (straight-line, smooth velocity profile) between
  the vehicle's current position and the commanded waypoint.
- Build an offline voice interface (8 fixed commands: left, right, forward,
  back, up, down, hover, stop — up/down added 2026-09-07 for the Mission
  Control controller, see `CLAUDE.md` "Stage E — Mission Control") that
  maps speech to a **waypoint** (a position offset from the vehicle's
  current position, or "hold current position" for hover/stop), drives the
  position → attitude cascade, and renders the response live in 3D with a
  telemetry panel.
- Produce paper-grade evidence at every stage: plots, error metrics,
  eigenvalue spectra, ablations — not just a working demo.

## 3. Non-goals

- No identification of translational/position dynamics — those use known
  Newtonian physics in the outer loop; only attitude dynamics are identified.
- No obstacle sensing or avoidance.
- No ROS/Gazebo/PX4 integration — MuJoCo only, matching course scope and the
  prior baseline projects (CD5, Anirudh et al.) this project follows.
- No cloud speech services — voice pipeline is fully offline/local.
- No continuous/freeform voice commands or arbitrary named waypoints — 6
  fixed discrete commands only, each mapping to a relative position offset
  or hold.
- No multi-vehicle, swarm, or formation control.
- No global path planning around obstacles — point-to-point straight-line
  trajectories only.

## 4. Target deliverables

1. A fully reproducible Stage A excitation dataset (raw + processed, logged
   per the schema in `DATA_MODEL.md`).
2. Fitted SINDy and DMDc models with documented library/regime choices, held
   in `02_identification/`.
3. A validation report (Stage C) with one-step-ahead error tables,
   short/long-horizon rollout plots, eigenvalue spectra, sparsity and
   noise ablations, and a written SINDy-vs-DMDc comparison and model
   selection decision.
4. A tuned quaternion attitude controller (Stage D, inner loop) with
   step-response metrics on both the identified model and ground-truth
   MuJoCo, plus a standard outer-loop position/trajectory controller and
   point-to-point trajectory planner, with a stated sim-to-sim gap for the
   inner loop and tracking metrics (position error, settle time to
   waypoint) for the outer loop.
5. A live voice-to-3D-navigation demo (Stage E) — spoken command → waypoint
   → visible point-to-point flight to that waypoint — with a telemetry
   panel and a logged multi-trial success-rate report.

## 5. Success criteria per stage

Measurable acceptance criteria — restated from the stage-gate table in
`../CLAUDE.md` (that table is the source of truth; keep this section in
sync with it):

- **Stage A gate:** response plots show rich oscillation/decay across varied
  excitation frequency content — not a flat near-zero signal — for every
  excited axis.
- **Stage B gate:** one-step-ahead prediction error on held-out excitation
  trials (never a time-slice of a training trajectory) is below a stated
  numeric threshold, for both SINDy and DMDc, before any rollout is
  attempted. Threshold value to be fixed in `TDD.md` once Stage A data
  exists and noise characteristics are known.
- **Stage C gate:** every plot traces back to a model that already passed
  Stage B. Divergence point (where open-loop rollout error crosses a stated
  bound) is explicitly reported for both models at both short and long
  horizon. A model is selected as "the identified model" with a written
  justification.
- **Stage D gate:** (inner loop) step-response settling time, overshoot, and
  steady-state error for the identified-model-driven controller are within
  a stated tolerance (to be fixed in `TDD.md`) of the same metrics measured
  with the controller driving ground-truth MuJoCo physics. (outer loop)
  point-to-point waypoint tracking achieves steady-state position error
  below a stated tolerance, with no instability introduced by cascading
  onto the identified-model-driven inner loop.
- **Stage E gate:** end-to-end spoken-command-to-correct-final-position
  success rate is measured over N ≥ 20 trials (6 commands, repeated) and
  reported alongside a latency breakdown (VAD → ASR → classification →
  trajectory generation → control → visible response).

## 6. Users / audience

The primary audience is the course instructor/grader — the deliverable must
stand on its own as evidence of correct data-driven-control methodology, not
just a working demo. Secondary audience: the student team itself, who need
the staged structure to avoid repeating the prior attempt's failure mode.
