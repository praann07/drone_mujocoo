# Understand My Project — Written For Me, Not For Grading

This file is just for you. `README.md` is the polished version for your
professor/GitHub. This one explains everything in plain English, in the
order it actually happens, file by file, so you can talk about your own
project without getting lost.

**If your professor is specifically focused on the controller logic or
the IMU, also read [`CONTROLLER_AND_IMU_EXPLAINED.md`](CONTROLLER_AND_IMU_EXPLAINED.md)
— it goes much deeper on just those two topics.**

---

## 1. The one-paragraph version

You built a drone that flies in a physics simulator (MuJoCo). You speak a
word like "left." Your laptop turns your voice into text, figures out
which of 8 commands you meant, and sends the drone flying there. **The
important part, the part that's actually graded:** the part of the drone
that keeps it stable while flying (the "attitude controller") was NOT
hand-programmed by writing physics equations. Instead, you flew the drone
around randomly first, recorded what happened, and had a computer program
(**SINDy**) *discover* the physics equations by itself, just by looking at
the data. Then you built the controller on top of those *discovered*
equations, not equations you looked up in a textbook. That's what "data
driven control" means. Everything else — the voice commands, the 3D city,
the graphs — is the demo that shows this actually works.

---

## 2. THE most important thing to understand: two separate timelines

This is the part that's easy to miss, so it gets its own section before
anything else.

**Stages A → B → C → D happen ONCE** (or whenever you deliberately rerun
them). They are a one-time (or occasional) *build* process. Each one
writes its output to a file, and the NEXT stage reads that file — they
never talk to each other directly while running.

**Stage E is what runs EVERY time you fly.** It does NOT redo any of the
identification or validation work. It just loads the *already-finished*
model and controller that A→D produced, and uses them live.

```
 ONE-TIME BUILD PIPELINE (rerun only if you change something upstream)
 ======================================================================

 [Stage A]                 [Stage B]                [Stage C]
 01_simulation/       ->   02_identification/  ->   03_validation/
 shake the drone,          fit SINDy + DMDc          test the fitted
 record raw motion         to that recording         models don't fall
      |                         |                     apart; pick winner
      v                         v                         |
 data/raw/*.parquet    (fitted model, in memory)          v
                                                  data/processed/
                                                  sindy_fitted_model.npz
                                                  <- THE FROZEN, PROVEN MODEL
                                                         |
                                                         v
                                                    [Stage D]
                                                    04_control/
                                                    build a controller
                                                    ON TOP OF that frozen
                                                    model (gains derived
                                                    from it, not guessed)


 EVERY-TIME LIVE FLIGHT (what actually runs when you use the demo)
 ======================================================================

 [Stage E]  05_voice_interface/
   loads the frozen model (Stage C's file) + the controller design
   (Stage D's math, re-derived instantly from that file) ONCE at startup
   -> then just flies, live, using them, every time you speak a command
```

If you ever get asked "does saying a command re-run the SINDy fitting or
the validation," the answer is a clean **no** — all of that already
happened, was checked, and got saved to one file
(`data/processed/sindy_fitted_model.npz`). Flying just *uses* it.

---

## 3. Every stage, every file in it

### Stage A — `01_simulation/` (shake the drone, record what happens)

| File | What it does |
|---|---|
| `excitation.py` | Generates the actual "shake" signals — PRBS (fast random jitter) and chirps (slow-to-fast sweeps), as small offsets around hover thrust. |
| `sim_driver.py` | Actually steps the MuJoCo model through those signals. **No controller runs during this** — that's deliberate (docs/TDD.md §3): if a controller were correcting the drone WHILE you're trying to learn its physics, the correction would hide the very physics you're trying to discover. |
| `run_excitation.py` | The entry point — runs every trial, saves the raw recordings to `data/raw/`, builds the train/held-out split, and makes the gate-check plots. |
| `models/quad.xml` | The drone itself: 4 rotors, mass, the IMU sensor block, the mixer geometry. |
| `models/scene.xml` | Wraps `quad.xml` with a floor, sky, lights, and (via `city.xml`) the 3D city — visual only, zero physics effect. |
| `models/city.py`, `make_city.py` | Single source of truth for the city layout (buildings, tower, roads) — generates both the 3D geoms and the printed map texture from ONE list, so they can never drift apart. |

### Stage B — `02_identification/` (guess the physics from that recording)

| File | What it does |
|---|---|
| `preprocessing.py` | Cleans the raw recording: smooths it (Savitzky-Golay filter, removes sensor jitter without lagging the signal), and converts the quaternion attitude into a safe local math representation (a rotation vector) that can't blow up near certain orientations the way raw quaternions can during fitting. |
| `sindy_fit.py` | The actual SINDy algorithm — builds a big library of candidate physics terms, fits them, then throws away every term that isn't pulling its weight (sparsity). |
| `dmdc_fit.py` | The DMDc algorithm — a simpler, linear-only alternative, fit only on near-hover data. |
| `run_identification.py` | Entry point: preprocesses everything, fits both models, checks the gate (accurate on data it never saw), saves the numeric results. |

### Stage C — `03_validation/` (fact-check the guess)

| File | What it does |
|---|---|
| `rollout.py` | The actual "predict forward continuously for several seconds and see if it diverges" engine — used by every test in this stage. |
| `run_validation.py` | Entry point: runs short + long rollouts, the eigenvalue check, the sparsity ablation, the noise ablation, the SINDy-vs-DMDc head-to-head — picks the winner (SINDy) — and **freezes it** to `data/processed/sindy_fitted_model.npz`. This is the one file everything downstream depends on. |

### Stage D — `04_control/` (build the autopilot on the frozen model)

| File | What it does |
|---|---|
| `mixer.py` | Converts (total thrust, twisting force) into 4 individual motor numbers, using the drone's real "X" rotor geometry. Also the reverse direction (used by the controller). |
| `controller.py` | The **inner loop** — keeps the drone pointed the right way. Its gains are calculated directly from the frozen SINDy model, not guessed. |
| `position_controller.py` | The **outer loop** — decides where to go and how fast, using plain textbook physics (deliberately NOT identified — see `CONTROLLER_AND_IMU_EXPLAINED.md` for why). |
| `trajectory.py` | Plans a smooth speed-up/cruise/slow-down path to each target, so the outer loop never demands an impossible instant jump in speed. |
| `metrics.py` | Computes settling time / overshoot / steady-state error from a test flight's logged error — used to grade the controller's own step responses. |
| `closed_loop_sim.py` | A test harness: runs the controller against EITHER the identified SINDy model OR real MuJoCo physics, so the two can be compared side by side (this is where `quat_rotate_vector()` lives — the same helper `flight.py` reuses for the IMU velocity fix). |
| `run_inner_loop.py`, `run_cascade.py` | Entry points: test the inner loop alone, then the full inner+outer cascade, on real setpoint scenarios, and save the plots/CSVs. |

### Stage E — `05_voice_interface/` (fly it live, with your voice)

| File | What it does |
|---|---|
| `voice_input.py` | Microphone capture (`sounddevice`) + Vosk speech recognition. Also a `TextSource` fallback for typing instead of speaking. |
| `commands.py` | Turns a transcript into one of the 8 fixed commands (or "not recognized"), plus the world-frame waypoint offset for each. |
| `flight.py` | Wraps MuJoCo + the whole Stage D cascade into one object you can call `.step()` on, over and over, in real time. Loads the FROZEN model from Stage C — never refits it. Reads all drone state through the IMU sensors (see `CONTROLLER_AND_IMU_EXPLAINED.md`). |
| `demo.py` | The actual live window: the 3D MuJoCo viewer + the telemetry dashboard (strip charts, HUD, heard-log, buttons) you interact with. |
| `run_trials.py`, `run_preemption.py` | Headless (no window needed) scripted test flights — used to generate the 24/5/mid-flight-override evidence numbers, not for interactive use. |
| `render_offscreen.py` | Flies a scripted sequence with no display and saves it as a GIF (chase + bird's-eye side by side). |
| `plot_2d_telemetry.py`, `plot_3d_trajectory.py` | Standalone "after the flight" graphs, read from whatever was most recently logged. |
| `print_commands_guide.py` | Prints the on-screen command reference used by `RUN_EVERYTHING.bat`. |

---

## 4. The full journey: you say "left", then "stop"

This is the part that actually runs live when you use the demo — entirely
inside Stage E, using what Stages A–D already built.

**Step 1 — Your voice becomes a sound wave.**
Your microphone (via `sounddevice`, in `voice_input.py`) is constantly
recording tiny chunks of audio, 16,000 samples per second.

**Step 2 — The sound becomes text.**
Those audio chunks are fed into **Vosk** (an offline speech-recognition
engine — no internet needed, it never leaves your laptop). Vosk doesn't
wait for you to finish talking to update its guess — while you're still
speaking, it shows a "partial" guess that updates live (that's the
`LISTENING: ...` line you see on screen). Once you pause, it finalizes a
transcript, e.g. `"go left"`.

**Step 3 — The text becomes a command.**
`commands.py::classify()` takes that transcript and decides which of the
8 fixed commands you meant: `forward, back, left, right, up, down, hover,
stop`. It's comparing the words you said against the 8 known words (and a
list of natural synonyms like "straight" → forward, "stay" → hover) using
letter-by-letter similarity scoring. If nothing scores high enough, it
honestly returns "not recognized" instead of guessing — and that miss
shows up on screen too, in the HEARD LOG panel, instead of vanishing
silently.

**Step 4 — The command becomes a target position.**
Each command has a fixed real-world offset: `left` = +1.5 meters sideways
from wherever the drone currently is, `up`/`down` = ±0.5 meters, `hover`/
`stop` = stay exactly where you are (`commands.py`).

**Step 5 — Safety check.**
Before that target is accepted, `city.py::clamp_target_to_safe_zone()`
checks whether flying straight there would put the drone inside one of
the city's buildings (buildings are visual-only — nothing in the physics
would stop the drone from flying through one). If it would, the target
gets pulled back to stop just short of the building instead.

**Step 6 — The outer loop plans a smooth path.**
`trajectory.py` builds a **trapezoidal** speed profile from where the
drone is now to the (safety-checked) target: speed up, cruise, slow
down. `position_controller.py` continuously compares where the drone
actually is against where that plan says it should be right now, and
outputs "tilt this way, with this much total thrust" — the outer,
plain-physics loop from Stage D.

**Step 7 — The inner loop makes it happen.**
`controller.py` takes that "tilt this way" instruction and computes exact
motor power for all 4 rotors, using gains derived from the SINDy-discovered
physics (Stage D). This runs 500 times per second (every 2 milliseconds)
— much faster than the outer loop needs to think, because keeping the
drone from tumbling is more urgent than steering it somewhere.

**Step 8 — The mixer converts to real motor numbers.**
`mixer.py` turns (total thrust, twisting force) into 4 individual rotor
commands, respecting the drone's actual X-shaped rotor layout.

**Step 9 — MuJoCo actually simulates it.**
Those 4 motor power numbers get handed to MuJoCo, which simulates real
forces, gravity, and rotation for one 2-millisecond tick, and moves the
drone's simulated body accordingly. The controller then reads the NEW
state back through the IMU sensors (`flight.py`) for the next iteration.

**Step 10 — You see it.**
The MuJoCo 3D window redraws the drone (chase camera or bird's-eye — the
TOGGLE CAMERA button). The dashboard window updates its 3 live
strip-charts (attitude error, position error, control effort) and the HUD
text (position, quaternion, latency numbers).

**Step 11 — It gets written down.**
Every time you finish one command and start the next, `demo.py`'s
`LiveFlightLogger` closes the books on the command that just finished —
where it started, where it was told to go, where it actually ended up,
how long it took, whether it was within tolerance — and saves that to a
file immediately (not just when you quit). More in section 6.

**Then you say "stop."** Steps 1–11 repeat exactly the same way — "stop"
classifies to a target offset of (0,0,0), meaning "don't add any offset,
hold right here" — so the drone just cancels its motion and holds its
current position.

---

## 5. What actually happens when you double-click a launcher

### `RUN_EVERYTHING.bat` (fly, step 1 of 2)
1. Checks your Python virtual environment exists.
2. Checks whether the Vosk voice-recognition model folder is present — if
   yes, launches with your microphone; if not, falls back to typing
   commands instead (never just crashes).
3. Opens 3 separate windows: the **Commands Guide** (`print_commands_guide.py`,
   stays open), the **3D Demo** (`demo.py` — this is where you
   speak/type/click), and the **Test Suite** (runs all 43 automated tests
   in the background, proving the code still works).

### `SHOW_RESULTS.bat` (see results, step 2 of 2)
Run this AFTER you've flown at least once. Opens 2 windows
(`plot_2d_telemetry.py`, `plot_3d_trajectory.py`) reading whatever you
just flew — not old data — because of the live-logging in section 6.

### `REBUILD_RESULTS.bat` (rebuild everything from scratch)
This is the "prove it all still works, from zero" button. It re-runs
Stage A through E in order, runs the 67-maneuver gauntlet, runs
`save_plots.py` (gathers every plot/CSV/video into `results/`), then runs
the full test suite. Every step's raw console output is saved into
`results/logs/`.

---

## 6. How results actually get saved (the data trail)

There are TWO separate places things get saved, and they mean different
things:

**`data/processed/`** — the WORKING data. Every stage script writes its
own output here (`excitation_plots/`, `validation_plots/`,
`control_plots/`, `voice_demo/`, `voice_sessions/`). This grows every time
you run anything, including live flights — that's normal, expected
clutter, not something to clean up by hand.

**`results/`** — the CURATED, presentation-ready copy. This is what
`README.md` and `results/RESULTS.md` actually point to and show. It's
built by copying/regenerating specific files FROM `data/processed/` — it
never gets written to directly by the pipeline scripts.

**The live-flight logging specifically**: `demo.py`'s `LiveFlightLogger`
picks one filename the moment you start flying, then rewrites that same
file every single time you finish a command — not just when you close the
window. So if you check `SHOW_RESULTS.bat` mid-flight, you see everything
you've done SO FAR, live-updating, not stale data from days ago.

---

## 7. Every result image, explained

All paths below are inside `results/`.

### 3D visuals (the "wow, it's really flying" evidence)

**`videos/navigation.gif`** — the actual MuJoCo physics simulation,
recorded frame-by-frame, flying the full command sequence over the 3D
city (roads, buildings, the obstacle tower, parks). Two camera views side
by side: chase camera (behind the drone) on the left, bird's-eye
(straight down) on the right. This is real physics, not a hand-animated
video.

**`videos/robustness_preemption.gif`** — same idea, demonstrating one
specific thing: the drone is flying forward, then told "back" partway
through, and it reverses immediately instead of finishing the forward
move first — proving commands interrupt each other instantly rather than
queuing up.

**`videos/gauntlet_full.gif`** — a longer recording covering the entire
67-maneuver test run: every single command, plus chained sequences,
played back to back.

**`plots_3d/flight_3d_trajectory.png`** — NOT a re-render of the MuJoCo
scene; a separate matplotlib plot of the ACTUAL recorded flight path (x,
y, z position over time) drawn as a 3D line, with the city buildings
drawn as simple boxes for scale. Two views side by side: isometric
(angled) and top-down. Multiple recent flights are overlaid in different
colors so you can compare sessions.

**`plots_3d/rollout_sindy_long_3d_phase.png`** — about the IDENTIFICATION,
not the flight demo. Plots the drone's actual spin speed on 3 axes (ωx,
ωy, ωz — "omega," angular velocity) as a 3D curve, ground truth vs. what
SINDy predicted, for an 11-second held-out test. If the two curves trace
the same loop through this 3D space, the discovered physics is genuinely
tracking real rotational behavior, not just coincidentally matching
numbers.

### Stage A — did the recorded data actually teach anything?

**`response_roll.png`, `response_pitch.png`, `response_yaw.png`,
`response_combined.png`** — raw recordings of the drone being shaken on
each axis (and all axes together). Each has 2 columns (a "chirp" trial
and a "PRBS" trial — the two excitation signal types) and 3 rows
(attitude as a quaternion, total rotation angle, angular velocity). The
point is just "yes, this data has real, rich, oscillating motion in it,"
not a flat boring line.

### Stage B/C — did the computer's physics guess hold up?

**`input_output_data_definition.png`** — 3 stacked rows: (1) the 4 motor
power signals sent in, (2) the resulting spin (ω) that came out, (3) the
rate-of-change targets SINDy/DMDc were actually trained to predict. "Here's
exactly what the learning algorithm was shown," for one recording.

**`eigenvalue_spectrum.png`** — a stability check, drawn as dots on a
graph. For SINDy (right panel): dots to the LEFT of the red line = stable
(a disturbance dies out); dots ON the line = "marginal," which is
CORRECT and expected here, not a bug — a drone with no controller running
has no reason to naturally return to a given heading. For DMDc (left
panel): dots INSIDE the dashed circle = stable in the same sense.

**`sparsity_ablation.png`** — x-axis: how many physics terms were kept.
y-axis: how wrong the prediction was. The curve drops steeply then goes
flat — past a certain point (the dashed red line, 44 terms), adding MORE
terms barely helps, which is exactly why 44 was chosen.

**`noise_ablation.png`** — x-axis: how much fake sensor noise was added
before fitting. y-axis: resulting error. Flat and low up to about 1–2
rad/s of noise, then curves upward sharply — a clear breaking point
rather than silently degrading.

**`head_to_head_near_hover.png`** — SINDy vs. DMDc, tested on the exact
same near-hover data DMDc was specifically built for (a fair fight).
SINDy matches or beats DMDc even here.

**`rollout_sindy_short.png`, `rollout_sindy_long.png`,
`rollout_dmdc_short.png`, `rollout_dmdc_long.png`** — each shows 3 rows:
attitude angle over time, spin (ω) over time, combined error over time,
ground truth (solid) vs. predicted (dashed), for a short (~2s) and long
(~11s) simulated flight with NO controller involved — testing the RAW
discovered physics equations by themselves.

### Stage D — does the autopilot actually work?

**`inner_loop_roll_step_15deg.png`, `inner_loop_ramp_roll_to_30deg.png`,
`inner_loop_large_step_60deg.png`** — attitude error (degrees) dropping to
near-zero over time, for a small sudden step, a smooth ramp, and a large
sudden step. Blue solid = controller on the SINDy-discovered model,
orange dashed = same controller on the real MuJoCo simulator — they sit
almost exactly on top of each other.

**`cascade_forward.png`, `cascade_left.png`** — the FULL cascade (both
loops together) flying an actual waypoint. Left panel: horizontal flight
path. Right panel: distance-to-target over time, dropping below the
dotted 0.15m tolerance line and staying there — proof the whole stack
(voice → waypoint → outer loop → inner loop → physics) converges.

### Stage E — the live demo's own telemetry

**`telemetry_2d.png`** — 3 stacked rows from a real recorded flight
session: position error over time, attitude error over time, and the 3
latency numbers (classify / trajectory-plan / control-dispatch) in
milliseconds — literally what the live dashboard shows you, saved as a
static image after the fact.

**`robustness_preemption.png`** — the numeric companion to the preemption
GIF: forward-position-over-time, with a shaded band showing where the
"back" command was given and where the velocity actually reversed sign,
plus a dotted line showing where the drone WOULD have ended up if "back"
had been ignored, versus where it actually stopped.

---

## 8. Quick file map (every file, one line each)

```
01_simulation/
  excitation.py         generates the shake signals (PRBS + chirp)
  sim_driver.py          steps MuJoCo through them, no controller, logs raw data
  run_excitation.py      entry point - runs it all, makes gate plots
  models/quad.xml         the drone: rotors, mass, IMU sensors
  models/scene.xml        drone model + floor/sky/lights + the city
  models/city.py           single source of truth for the city layout
  models/make_city.py      generates the 3D city geoms + printed map texture

02_identification/
  preprocessing.py        cleans raw data, smooths, computes derivatives
  sindy_fit.py             the SINDy algorithm
  dmdc_fit.py               the DMDc algorithm
  run_identification.py     entry point - fits both, checks the gate

03_validation/
  rollout.py               the "predict forward, check for divergence" engine
  run_validation.py         entry point - runs every check, freezes the winner

04_control/
  mixer.py                 (thrust,torque) <-> 4 motor numbers
  controller.py             inner loop (data-driven gains)
  position_controller.py    outer loop (plain physics)
  trajectory.py              smooth path planner
  metrics.py                 settling time / overshoot calculators
  closed_loop_sim.py          test harness: identified model vs real MuJoCo
  run_inner_loop.py            entry point - tests the inner loop alone
  run_cascade.py                entry point - tests the full cascade

05_voice_interface/
  voice_input.py            microphone + Vosk speech recognition
  commands.py                 transcript -> command classifier
  flight.py                    wraps MuJoCo + Stage D cascade, reads via IMU
  demo.py                       the live 3D window + dashboard
  run_trials.py                  headless scripted test flights
  run_preemption.py               headless mid-flight-override test
  render_offscreen.py              makes the demo GIFs
  plot_2d_telemetry.py              "after" 2D graph
  plot_3d_trajectory.py              "after" 3D graph
  print_commands_guide.py             on-screen command reference

data/raw/               Stage A's raw recordings
data/processed/          every stage's working output (grows constantly)
results/                the curated, presentation-ready copy of everything
  RESULTS.md              <- the actual human-readable numbers summary
  data/README.md            explains every raw CSV/parquet column
  logs/README.md            explains every raw console log

RUN_EVERYTHING.bat        fly (step 1)
SHOW_RESULTS.bat           see your flight's results (step 2)
REBUILD_RESULTS.bat         rebuild the ENTIRE results/ folder from scratch
CONTROLLER_AND_IMU_EXPLAINED.md   deep dive on the two topics being graded
```

If you ever forget what a file does, this document + `results/RESULTS.md`
+ `CONTROLLER_AND_IMU_EXPLAINED.md` should answer it without needing to
open the code.
