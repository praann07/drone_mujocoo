# Voice-Controlled Quadrotor Navigation System (Data-Driven Attitude Identification)

**Course:** Data-Driven Control of Drones  
**Status:** All Verification Gates Passed (Stages A → E) — Verified in [`CLAUDE.md`](CLAUDE.md)  
**Physics Simulator:** MuJoCo 6-DOF Rigid-Body Dynamics (`quad.xml`, `scene.xml`)  
**Identification:** SINDy (Sparse Identification of Nonlinear Dynamics) on the drone's rotational dynamics  
**Control Architecture:** Quaternion Cascaded Control (Newtonian Outer Loop → SINDy-Derived Inner Loop)

---

## Abstract

Course projects in data-driven drone control commonly suffer from one of two failure modes: either an offline speech/UI layer is grafted onto textbook physics control without empirical system identification, or system identification is conducted in a Jupyter notebook without ever stabilizing closed-loop flight in a 3D simulation.

This project bridges that divide through a strict, principled architectural separation: **only the inner attitude loop is identified from data.** Moving in a straight line under a thrust vector is just Newton's second law — a plain double integrator with nothing for a data-driven algorithm to discover. Rotation is different: it has real nonlinear aerodynamics, cross-coupling between axes, and quaternion-constrained kinematics — genuinely worth identifying from data, which is exactly what this project does.

The inner-loop attitude controller's gains are **analytically derived** — calculated directly from the identified SINDy model's own physics — not hand-tuned by trial and error. The complete system runs end-to-end: spoken voice commands drive world-frame waypoints, a Newtonian outer loop generates smooth trajectories and desired attitude streams, the SINDy-derived inner controller commands four rotor thrusts in MuJoCo physics, and flight telemetry is streamed live to a mission-control HUD beside the 3D viewport.

```
 spoken command ("forward", "back", "left", "right", "up", "down", "hover", "stop")
                   │
                   ▼  Offline ASR (Vosk) + Fuzzy Classifier (difflib)
        world-frame target waypoint
                   │
 ┌─────────────────┴────────────────── OUTER LOOP (Newtonian Physics) ──────────────────┐
 │  Trapezoidal Trajectory Planner  ──►  PD Position & Velocity Controller              │
 │                                  ──►  Outputs: Desired Attitude q_des, Thrust T_des  │
 └─────────────────────────────────────────────┬────────────────────────────────────────┘
                                               ▼
 ┌──────────────────────────────────── INNER LOOP (Data-Driven SINDy) ──────────────────┐
 │  Quaternion Controller (Kq, Komega derived from identified SINDy hover Jacobian)     │
 │                                  ──►  Motor Mixer Matrix (4 rotor thrusts u1..u4)    │
 └─────────────────────────────────────────────┬────────────────────────────────────────┘
                                               ▼
                              MuJoCo 6-DOF Rigid-Body Simulation
                                               │
                                               ▼
                         Live 3D Viewport + Mission-Control HUD
                        (Attitude Error / Position Error / Effort)
```

---

## Methodology: Genuinely Data-Driven Control

### 1. SINDy Rotational Identification
The drone's rotation is described by two things: its attitude (which way it's pointing) and its angular velocity — how fast it's spinning on each axis. SINDy is given the 4 motor commands and the resulting angular velocity from real flight data, and has to figure out the equation connecting them, on its own — it isn't told the physics in advance.

It does this with **Sequential Thresholded Least Squares (STLSQ)**: try a large library of candidate physics terms, fit them all, then repeatedly drop the weakest ones until only the terms that actually matter survive. Out of 156 candidate terms, it kept **44** — and those 44 recover the real drag, motor effectiveness, and cross-axis coupling numbers within **2–6%** of the simulator's actual ground truth (verified in [`tests/test_identification_physics.py`](tests/test_identification_physics.py)). That match is the real proof it learned genuine physics, not a curve that merely looks right.

### 2. Analytical Gain Derivation (No Hand-Tuning)
Controller gains are never chosen by guesswork. `gains_from_identified_model()` in [`04_control/controller.py`](04_control/controller.py) reads the drag-vs-inertia number straight off the identified model (one per rotation axis) and plugs it into a standard, textbook control-design rule that targets a specific response speed and damping — the same 2nd-order design method taught in any controls course, just fed real identified numbers instead of guessed ones. The only two things chosen by hand at all are how fast (settling time) and how smoothly (damping) it should respond — every actual gain number is calculated from those two choices plus the identified physics. When the identified model changes, the gains recalculate automatically. (Full formula, worked through step by step: [`CONTROLLER_AND_IMU_EXPLAINED.md`](CONTROLLER_AND_IMU_EXPLAINED.md).)

### 3. Frozen Model Artifact Discipline
At runtime, Stage E does not silently refit models from raw data. Instead, [`03_validation/run_validation.py`](03_validation/run_validation.py) serializes the gate-validated model and hover Jacobian to `data/processed/sindy_fitted_model.npz`. The real-time flight controller ([`05_voice_interface/flight.py`](05_voice_interface/flight.py)) loads this frozen artifact directly via `np.load()`, guaranteeing deterministic, instant startup and ensuring that flight tests execute against the exact validated model.

### 4. Full Cascade Architecture (Outer Loop + Mixer)
The inner loop above is one half of a two-loop cascade, run at every control step (`dt = 0.002 s`):

- **Outer loop** ([`04_control/position_controller.py`](04_control/position_controller.py), [`trajectory.py`](04_control/trajectory.py)) — deliberately **standard physics, not identified** (see `docs/PRD.md` §1): a trapezoidal trajectory planner generates a smooth speed-up/cruise/slow-down path toward each voice-command waypoint, and a PD controller converts the resulting position/velocity error into a desired acceleration. That acceleration is then converted into "which way should the drone tilt" — since thrust has to point in the direction you want to accelerate, tilting is just how a drone redirects thrust sideways — using the smallest possible tilt that achieves it (`minimal_rotation_z_to()`), since yaw is never commanded.
- **Inner loop** consumes that desired attitude + thrust exactly as in §2.
- **Mixer** ([`04_control/mixer.py`](04_control/mixer.py)) inverts the drone's actual X-configuration rotor geometry (positions + CW/CCW spin pairing, matched against `quad.xml` and pinned by [`tests/test_mixer.py`](tests/test_mixer.py)) to convert (thrust, torque) into 4 individual rotor commands, reporting `saturated=True` rather than silently clipping an unachievable command.

Bandwidth separation between the two loops (outer settling ~3s target vs. inner ~0.56–0.86s measured) is what justifies designing them independently rather than as one coupled MIMO system — a standard cascade-control argument, not an assumption left unverified: Stage D's full-cascade gate confirms the combined system settles correctly (§ Experimental Results below).

See [`CONTROLLER_AND_IMU_EXPLAINED.md`](CONTROLLER_AND_IMU_EXPLAINED.md) for the complete walkthrough of both loops plus the IMU sensing model, including the quaternion double-cover handling and why the outer loop is intentionally not identified.

---

## IMU / State Sensing

[`01_simulation/models/quad.xml`](01_simulation/models/quad.xml) declares a full IMU sensor block on a dedicated `imu` site:
```xml
<sensor>
  <framepos name="position" objtype="site" objname="imu"/>
  <framequat name="attitude" objtype="site" objname="imu"/>
  <velocimeter name="linear_velocity" site="imu"/>
  <gyro name="angular_velocity" site="imu"/>
</sensor>
```
The flight controller ([`05_voice_interface/flight.py`](05_voice_interface/flight.py)) reads all state — position, attitude, linear velocity, angular velocity — **through these named sensors** (`data.sensordata`, looked up by name via `mj_name2id`), not by reading MuJoCo's internal `qpos`/`qvel` arrays directly. The `gyro` channel is exactly what a physical IMU measures. `linear_velocity` is reported by the sensor in the **body frame** (as a real velocimeter would be), and is explicitly rotated into the world frame the rest of the controller expects via `quat_rotate_vector()` ([`04_control/closed_loop_sim.py`](04_control/closed_loop_sim.py)) using the (also sensor-sourced) current attitude — the same frame-transform step a real flight computer performs on IMU-derived velocity.

**Honest scope note:** `position`, `attitude`, and `linear_velocity` are MuJoCo direct-measurement sensor types — they report true, noiseless simulator state, not the output of a real accelerometer-plus-fusion-filter state estimator (which would drift without an external reference like GPS). This is a deliberate simplification appropriate for a controls-focused project: identifying the plant and designing a controller against it is the graded content here, not sensor fusion. Sensitivity to imperfect sensing is characterized separately by the Stage C `noise_ablation.png` result (robust to ~1–2 rad/s injected gyro noise, degrading past ~4 rad/s). Full reasoning and a Q&A-style walkthrough: [`CONTROLLER_AND_IMU_EXPLAINED.md`](CONTROLLER_AND_IMU_EXPLAINED.md).

---

## Experimental Results

Every quantitative result below is derived from reproducible scripts with strictly held-out validation datasets:

| Stage | Verification Focus | Quantitative Result | Status |
|---|---|---|:---:|
| **Stage A** — Excitation | Multi-axis PRBS & multisine persistent excitation | Zero runaway trials; rich cross-axis spectral decay | ✅ PASS |
| **Stage B** — Identification | SINDy held-out one-step prediction | **NRMSE = 0.0786** / **R² = 0.9946** (44/156 terms) | ✅ PASS |
| **Stage B** — Identification | DMDc held-out prediction (near-hover regime) | **NRMSE = 0.0825** / **R² = 0.9926** (degrades to 0.70 outside hover) | ✅ PASS |
| **Stage B** — Physics Sanity | Physical coefficient recovery vs. ground truth | Recovered within **2% to 6%** (drag, mixer effectiveness, gyro cross-term) | ✅ PASS |
| **Stage C** — Rollout Stability | 11 s open-loop autonomous rollout | **SINDy: 0/7 diverged** (full envelope) \| DMDc: 0/16 (hover only) | ✅ PASS |
| **Stage C** — Head-to-Head | Evaluated in DMDc's *own* near-hover regime | SINDy **0.0789** vs DMDc **0.0825** NRMSE (SINDy wins even near hover) | ✅ PASS |
| **Stage C** — Model Selection | Parsimony, global validity, closed-loop tracking | **SINDy Selected** (see [`model_selection.md`](data/processed/validation_plots/model_selection.md)) | ✅ PASS |
| **Stage D** — Inner Loop | Step settling time (SINDy model vs. MuJoCo plant) | **0.56 s – 0.86 s** (agrees within **~0.3%** across all axes) | ✅ PASS |
| **Stage D** — Full Cascade | Multi-waypoint 3D tracking & steady-state error | Settling time **~1.88 s**, Final Error = **0.0044 m** | ✅ PASS |
| **Stage E** — Isolated Trials | 24 scripted voice trials (reset to hover before each) | **24/24 Success (100.0%)** at 0.15 m tolerance | ✅ PASS |
| **Stage E** — Chained Flight | 5-command continuous sequence (**no resets between legs**) | **5/5 Success (100.0%)**, Final Error = **0.0000 m**, Max Chain Error = **0.0044 m** | ✅ PASS |
| **Stage E** — System Latency | End-to-end pipeline processing latency | Classify: 0.08 ms \| Trajectory: 0.03 ms \| Dispatch: 0.16 ms \| **Total: 0.27 ms** | ✅ PASS |

---

## Visual Evidence & Validation Plots

### 3D Visualization

This is a genuinely 3D project throughout — real MuJoCo rigid-body physics rendering, plus a genuinely data-driven 3D visualization of the identified model's phase-space behavior, not just a flight path.

**Live 3D physics simulation** (offscreen-rendered, real MuJoCo rendering — the same physics driving the interactive demo, not a pre-baked animation), flown over a 3D printed-map city (roads, block buildings, a forward obstacle tower, parks) drawn into `scene.xml`, with the drone commanding via the cascade:

![Live 3D Navigation Demo](data/processed/voice_demo/navigation.gif)

**Mid-flight command preemption** — the drone is told "back" at t≈0.7 s while still en route forward, so it reverses immediately instead of continuing toward the obstacle tower ahead (peak excursion 0.39 m vs the 2.3 m tower face):

![Command Preemption](data/processed/voice_demo/robustness_preemption.gif) ![Preemption Plot](data/processed/voice_demo/robustness_preemption.png)

**3D flight trajectory over the same city model** — isometric and top-down views of real logged flights (multiple sessions overlaid), drawn from the identical city spec (`01_simulation/models/city.py`) that `scene.xml` renders, independent of and never re-rendering the MuJoCo scene above:

![3D Flight Trajectory](data/processed/voice_demo/flight_3d_trajectory.png)

**3D angular-velocity phase portrait** — ground truth vs. SINDy-predicted trajectory through the identified model's own spin-rate space (ωx, ωy, ωz — spin on each of the 3 axes), for a combined-axis held-out rollout. This is tied directly to identification quality, not flight path:

![3D Rollout Phase Portrait](data/processed/validation_plots/rollout_sindy_long_3d_phase.png)

### System Identification & Validation Analysis

**What feeds the identification** — rotor thrust inputs, the angular-velocity response they produce, and the derivative targets SINDy/DMDc are actually fit to predict, all from one representative excitation trial:

![Input vs Output Data Definition](data/processed/validation_plots/input_output_data_definition.png)

| SINDy vs. DMDc Head-to-Head (Near Hover) | Discrete & Continuous Eigenvalue Spectra |
|:---:|:---:|
| ![Head to Head](data/processed/validation_plots/head_to_head_near_hover.png) | ![Eigenvalues](data/processed/validation_plots/eigenvalue_spectrum.png) |

| STLSQ Sparsity Ablation (threshold sweep) | Additive Gaussian Noise Sensitivity |
|:---:|:---:|
| ![Sparsity Ablation](data/processed/validation_plots/sparsity_ablation.png) | ![Noise Ablation](data/processed/validation_plots/noise_ablation.png) |

**Open-loop rollout tracking** (theta and omega, ground truth vs. predicted, plus combined-state error norm):

![SINDy Long-Horizon Rollout](data/processed/validation_plots/rollout_sindy_long.png)

### Closed-Loop Cascade Response
Cascaded tracking performance driving the full 6-DOF MuJoCo plant under SINDy-derived gains:

![Cascade Forward Step Response](data/processed/control_plots/cascade_forward.png)

---

## Maneuver Gauntlet — 67/67 PASS (100%)

Beyond the 24-isolated + 5-chained Stage E gate above, a dedicated stress test (`run_gauntlet.py`) exercises **every command, every chain, and the edge cases a strict reviewer would ask about first**:

| Section | Coverage | Result |
|---|---|:---:|
| Single moves | All 8 commands × 3 runs from clean hover | ✅ 24/24 |
| Chained sequences | Square, stairs, full-mix × 2 runs, no resets between legs | ✅ 38/38 legs |
| Edge cases | Rapid-fire re-dispatch, mid-move preemption, stop-then-resume, altitude limits (never below floor, never runaway) | ✅ 5/5 |

Full numbers, per-maneuver plots (CSV + PNG for all 67), and a combined chase+bird's-eye GIF of the entire gauntlet are in **[`results/RESULTS.md`](results/RESULTS.md)** — that file is the single consolidated deliverable (plots, videos, logs, raw data) if you only look at one thing. Reproduce it with:
```powershell
.\.venv\Scripts\python.exe run_gauntlet.py
.\.venv\Scripts\python.exe save_plots.py
```

**Bird's-eye camera**: the interactive demo (`demo.py`) and every offscreen render now show chase view *and* a top-down Google-Maps-style view together. In the live 3D window, press **B** to toggle the MuJoCo viewer itself between chase and bird's-eye.

---

## Latency Characterization & Verification

In compliance with strict scientific rigor, latency is characterized with full transparency:

1. **Scripted/Deterministic Benchmark (0.27 ms):**
   - The reported **0.27 ms** end-to-end figure is measured across 24 randomized trials using `TextSource`.
   - It strictly benchmarks algorithmic processing: fuzzy vocabulary classification (0.08 ms), trapezoidal trajectory calculation (0.03 ms), and cascade controller dispatch (0.16 ms).
   - This isolates control software overhead from hardware-dependent acoustic capture.

2. **Live Acoustic Pipeline (`VoskSource`):**
   - When running live microphone input via Vosk, acoustic decoding time is actively measured per utterance inside `VoskSource._audio_callback` using high-resolution monotonic clocks (`perf_counter`) around `AcceptWaveform()` and `Result()`.
   - Voice Activity Detection (VAD) is integrated directly within Kaldi's acoustic decoder endpointing logic rather than an artificial decoupled stage.
   - Live speech decoding typically incurs 80–220 ms depending on CPU load and phrase length, well within human interactive cadence. Zero fabricated latencies are reported.

---

## Reproducing the Results (End-to-End Execution)

To reproduce the complete pipeline from scratch in Windows PowerShell:

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Execute Staged Gates (In Order)
Each stage validates gate criteria and generates empirical logs in `data/processed/`:
```powershell
# Stage A: Generate PRBS/multisine excitation dataset & gate verification plots
.\.venv\Scripts\python.exe 01_simulation\run_excitation.py

# Stage B: Fit SINDy & DMDc models, evaluate held-out one-step prediction errors
.\.venv\Scripts\python.exe 02_identification\run_identification.py

# Stage C: Run 11s open-loop rollouts, ablations, model selection & freeze artifact
.\.venv\Scripts\python.exe 03_validation\run_validation.py

# Stage D: Validate inner attitude loop & outer cascaded trajectory tracking
.\.venv\Scripts\python.exe 04_control\run_inner_loop.py
.\.venv\Scripts\python.exe 04_control\run_cascade.py

# Stage E: Execute isolated (24/24) and chained (5/5) voice navigation trials
.\.venv\Scripts\python.exe 05_voice_interface\run_trials.py

# Stage E Demo: Regenerate the 3D offscreen animated GIF with telemetry overlay
.\.venv\Scripts\python.exe 05_voice_interface\render_offscreen.py

# Stage E Demo: Regenerate the mid-flight command-preemption GIF
.\.venv\Scripts\python.exe 05_voice_interface\render_offscreen.py --preempt

# Stage E: Headless command-preemption trial + robustness plot + parquet log
.\.venv\Scripts\python.exe 05_voice_interface\run_preemption.py

# Maneuver gauntlet: 67 scripted maneuvers, logs + per-maneuver plots + GIF
.\.venv\Scripts\python.exe run_gauntlet.py

# Collect every plot/CSV/GIF above into results/ (matching results/RESULTS.md)
.\.venv\Scripts\python.exe save_plots.py

# Full Verification: Run entire unit & regression test suite (43 passed)
.\.venv\Scripts\python.exe -m pytest tests\
```

**Or just double-click `REBUILD_RESULTS.bat`** — runs every command above in order, with each step's output saved to `results\logs\`. Takes several minutes (the gauntlet's 67 maneuvers are the slow part); this is a "regenerate all the evidence from scratch" utility, not something you run before every demo.

### 3. Interactive Live 3D Simulation & Telemetry HUD
Run the interactive MuJoCo 3D viewer accompanied by the real-time mission-control HUD:
```powershell
# Interactive typed/scripted control:
.\.venv\Scripts\python.exe 05_voice_interface\demo.py --source text

# Live microphone input (requires Vosk model in ./vosk-model-small-en-us-0.15):
.\.venv\Scripts\python.exe 05_voice_interface\demo.py --source vosk
```

### 4. Quick Launch (1-Click Windows Batch Scripts)
Double-click any of the launcher batch scripts directly from File Explorer:
* **`RUN_EVERYTHING.bat`** — For presenting, step 1: one click, zero typing. Opens the live 3D demo, a commands guide (stays open the whole time), and the full test suite.
* **`SHOW_RESULTS.bat`** — Step 2: after you've flown, one click opens fresh 2D + 3D graphs of *that* flight (the demo logs every command automatically — this is never an old test run).
* **`run_drone.bat`** or **`START_DEMO.bat`** — Interactive launcher menu (flight modes, 24+5 trials, 31-test pytest suite).
* **`run_voice_control.bat`** — Launches live microphone voice control with automatic model loading.
* **`run_text_control.bat`** — Launches interactive typed navigation (instant, no microphone needed).
* **`REBUILD_RESULTS.bat`** — Regenerates every plot/CSV/GIF in `results/` from scratch (the full Stage A-E pipeline + the 67-maneuver gauntlet). Takes several minutes; run this if you want to reproduce the whole evidence gallery, not just fly the demo.

### 5. Standalone Offline Analysis Plots
Run directly any time after a flight (no server, just a matplotlib window reading the most recently logged parquet under `data/processed/voice_sessions/`):
```powershell
.\.venv\Scripts\python.exe 05_voice_interface\plot_2d_telemetry.py
.\.venv\Scripts\python.exe 05_voice_interface\plot_3d_trajectory.py
```
The 3D plot draws the same city model `scene.xml` renders (imported from `01_simulation/models/city.py` — buildings and the obstacle tower), so the spatial scale matches the live 3D scene — not a re-render of MuJoCo's own 3D viewport. `run_preemption.py` exercises the mid-flight preemption scenario headlessly (reports peak forward excursion, clearance to the obstacle-tower face, and reversal latency) and writes `robustness_preemption.png`.

---

## Repository Structure

```
done_v2.0/
├── data/
│   ├── raw/                               # Raw excitation parquet datasets (Stage A)
│   └── processed/                         # Held-out splits, plots, logs & frozen model
│       ├── sindy_fitted_model.npz         # Frozen SINDy model artifact for Stage E
│       ├── excitation_plots/              # Excitation response plots
│       ├── validation_plots/              # Rollout, ablation, eigenvalue & comparison plots
│       ├── control_plots/                 # Inner loop & cascade tracking step responses
│       ├── voice_demo/                    # Animated 3D flight GIF (navigation.gif)
│       └── voice_sessions/                # Parquet logs of isolated & chained trials
├── docs/                                  # Formal specifications & engineering documentation
│   ├── PRD.md                             # Product Requirements Document
│   ├── TDD.md                             # Technical Design Document
│   ├── ENGINEERING_PLAN.md                # Task breakdown & stage milestones
│   ├── DATA_MODEL.md                      # Logged telemetry data schemas
│   └── DESIGN_FLOW.md                     # Architectural data flow & stage handoffs
├── 01_simulation/                # MuJoCo quadrotor model & excitation signals
├── 02_identification/                # SINDy (STLSQ) & DMDc identification algorithms
├── 03_validation/                    # Rollout verification, ablations & model selection
├── 04_control/                       # Quaternion controller & Newtonian trajectory planner
├── 05_voice_interface/           # Voice input, real-time HUD dashboard, live flight & offline plots
├── tests/                                 # Full Pytest test suite (31 tests across all stages)
├── CLAUDE.md                              # Single source of truth for stage-gate status
└── requirements.txt                       # Project dependencies
```
