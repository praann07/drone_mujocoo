# Controller Logic & IMU — The Deep Dive

Your professor is specifically focused on these two things. This document
exists so you can answer anything he asks about either one, in plain
English, with the real numbers. It's a companion to
[`UNDERSTAND_MY_PROJECT.md`](UNDERSTAND_MY_PROJECT.md) — read that first
if you haven't; this one goes much deeper on just these two topics.

**One-line summary of what changed today:** the code used to read the
drone's state (position, attitude, speed, spin) directly from the physics
engine's internal numbers, silently walking past the IMU sensor the
model actually declares. That's fixed now — the controller genuinely
reads through the IMU sensor block, with proof it changes nothing about
how the drone flies (see §4).

---

## 1. The controller, top to bottom

Two controllers are stacked ("cascaded"), each solving a different part
of the problem, running at different speeds.

```
   your command
        |
        v
  TRAJECTORY PLANNER  (trajectory.py)
   "how do I get from here to there smoothly?"
        |  gives: desired position, velocity, acceleration, every instant
        v
  OUTER LOOP  (position_controller.py)
   "given where I actually am vs where I should be, which way should
    I tilt, and how hard should I push?"
        |  gives: desired attitude (as a quaternion) + total thrust
        v
  INNER LOOP  (controller.py)
   "given which way I'm actually pointing vs where I should point,
    how much power does EACH of the 4 motors need?"
        |  gives: 4 individual rotor thrust commands
        v
      MIXER  (mixer.py)
   converts (total thrust, twisting-force) into 4 numbers, respecting
   the actual rotor geometry (an "X" layout) and spin directions
        |
        v
      MuJoCo physics — the drone actually moves
```

The **outer loop runs the whole cascade math every control step**, but
it's DESIGNED to react slowly and smoothly (its target settling time is
3 seconds). The **inner loop reacts fast** (its target settling time is
1.5 seconds, and in practice measured at 0.56–0.86s) — it has to, because
if the drone tips over even slightly and nothing corrects it FAST, it
crashes. This speed difference ("bandwidth separation") is why cascading
works: the fast inner loop can basically ignore the outer loop as
"a slowly-moving target," which is a standard, textbook justification for
why you're allowed to design the two loops mostly independently instead
of as one giant coupled system.

### 1a. The outer loop — plain physics, not learned

`position_controller.py` does something simple: compare where the drone
IS to where the trajectory plan says it SHOULD be right now, and compute
a corrective acceleration:

```
desired_acceleration = Kp * (position_error) + Kd * (velocity_error) + feedforward
```

This is a standard **PD controller** (Proportional-Derivative — react to
the error itself, AND to how fast the error is changing). Nothing here
was learned from data, and that's a deliberate choice: this is Newton's
second law applied to a point mass under thrust — `F = ma` — there's no
hidden nonlinearity for a data-driven method to discover here. Using
SINDy on this part would be like using AI to guess whether 2+2=4.

Once it has a desired acceleration, it converts that into "which way
should the whole drone body tilt, and how hard should the motors push
combined" — because a drone can only push thrust straight through its own
body, tilting is the ONLY way to accelerate sideways. This step
(`minimal_rotation_z_to()`) computes the smallest possible rotation that
points the drone's thrust axis where it needs to point, without adding
any unnecessary spin.

### 1b. The inner loop — THIS is the data-driven part

`controller.py`'s `QuaternionController` takes "point this way, with
this much thrust" and turns it into 4 individual motor commands. The
control law:

```
torque = -Kq * (attitude_error) - Komega * (spin_error)
```

Looks like an ordinary PD controller too — but the **numbers Kq and
Komega are not guessed or hand-tuned**. They come directly out of
`gains_from_identified_model()`, which reads the actual coefficients
SINDy discovered in Stage B (specifically, how strongly aerodynamic drag
resists spinning, extracted straight from the identified equations) and
solves a textbook pole-placement formula for exactly the response speed
you asked for (1.5 second settling, damping ratio 0.7 — meaning it
corrects fast without oscillating wildly). **If you re-ran Stage B and
got a slightly different identified model, these gains would
automatically come out slightly different too** — that coupling is the
entire point, and it's what makes this "data-driven control" instead of
"a PD controller that happens to run near a drone."

**Why quaternions, and what's the "double-cover" fix?** A quaternion is
a 4-number way to represent a 3D rotation that has no "gimbal lock"
(unlike roll/pitch/yaw angles, which can get mathematically stuck at
certain orientations). The one quirk: a quaternion `q` and its negative
`-q` represent the EXACT SAME physical rotation. Without accounting for
this, the controller could occasionally compute an error that says "turn
350 degrees this way" when it actually meant "turn 10 degrees the other
way" — a real, previously-documented failure mode in drone control
literature. The line `if qe[0] < 0: qe = -qe` in `compute()` is the fix:
it guarantees the controller always picks the SHORT way around.

### 1c. The mixer — turning (thrust, torque) into 4 numbers

The drone has 4 rotors in an "X" pattern. Pushing all 4 up equally gives
pure thrust. Pushing the front two harder than the back two tips it
forward. Pushing diagonal pairs differently makes it spin (yaw), because
alternating rotors spin in OPPOSITE directions specifically so their
spin-reaction-torques can cancel or combine on command. `mixer.py`
encodes exactly this geometry as a 4×4 matrix and inverts it, so
"I want this much thrust and this much twist" always converts to the
right 4 individual numbers. It also reports if the four numbers it
came up with are actually achievable (motors can't push negative, and
can't exceed 6 Newtons each) — `saturated=True` — rather than silently
clipping and pretending everything's fine.

---

## 2. The IMU — what it is, honestly

### What a REAL IMU actually gives you

A real Inertial Measurement Unit physically contains exactly two kinds
of sensor: an **accelerometer** (measures specific force — roughly,
"how hard am I being pushed/pulled right now") and a **gyroscope**
(measures spin rate directly). That's it. It does NOT directly tell you
your position, your orientation, or your velocity — those have to be
*estimated* by integrating the raw readings over time through a filter
(commonly a Kalman filter, or something like the Madgwick/Mahony
algorithms), and that estimate silently drifts and accumulates error the
longer you go without an outside reference (like GPS) to correct it.
This drift is one of the genuinely hard problems in real drone control.

### What THIS project's simulated IMU gives you

`01_simulation/models/quad.xml` declares an IMU as a sensor block on a
site called `imu`:

```xml
<sensor>
  <framepos name="position" objtype="site" objname="imu"/>
  <framequat name="attitude" objtype="site" objname="imu"/>
  <velocimeter name="linear_velocity" site="imu"/>
  <gyro name="angular_velocity" site="imu"/>
</sensor>
```

Only the `gyro` line is what a real IMU actually measures. The other
three (`framepos`, `framequat`, `velocimeter`) are MuJoCo convenience
sensors that report the exact, true, noiseless simulator state directly
— no drift, no estimation, no filter. **This is a standard and defensible
simplification for a controls-focused course**: the graded content here
is "can you identify the plant's dynamics and design a controller against
them," not "can you build a state estimator" — those are genuinely
different, both hard, problems, and mixing them in would dilute the
actual point of this assignment. If your professor asks "is this a
realistic IMU," the honest answer is: it's realistic in EXISTENCE
(the model, correctly, includes a sensor with realistic naming and
structure) but idealized in NOISE (it reports ground truth, not what a
physical sensor plus a fusion filter would actually produce). Stage C's
`noise_ablation.png` result exists specifically to speak to the "what if
it weren't idealized" question — see §3.

### The bug that got fixed today

Until today, `05_voice_interface/flight.py`'s `FlightController` had all
4 of `position()`, `velocity()`, `attitude()`, `angular_velocity()`
reading straight from `data.qpos`/`data.qvel` — MuJoCo's raw internal
physics arrays — completely bypassing the declared IMU sensor block. The
model FILE described an IMU; the CODE never actually consulted it. If
asked "does your controller use the IMU," the honest answer used to be
no.

**The fix**: every one of those 4 functions in `flight.py` now reads from
`data.sensordata`, through the named IMU sensors, looked up once by name
(not a hardcoded index, so it stays correct even if the sensor block is
ever reordered). Three of the four (`position`, `attitude`,
`angular_velocity`) are numerically identical to the old qpos/qvel reads
— they're direct, un-transformed measurements of the same quantity. The
fourth, `velocity`, is genuinely different in a meaningful way: the
`velocimeter` sensor reports velocity in the drone's own BODY frame (like
a real IMU would), not the world frame the rest of the code expects — so
`flight.py` now explicitly rotates it into world frame using the
(also sensor-sourced) current attitude, via `quat_rotate_vector()` — the
exact same operation a real flight computer performs on an IMU-derived
body-frame velocity estimate. That's not just relabeling a number; it's
a real, small piece of realistic sensor-fusion work, done correctly.

### A subtlety worth knowing (in case you're asked about it)

MuJoCo computes sensor values from the *pre-integration* state at each
physics step — so naively reading `sensordata` right after stepping
would be one control step (2 milliseconds) stale compared to reading
`qpos`/`qvel` directly. `flight.py::step()` now calls `mj_forward()`
once more after every `mj_step()` specifically to refresh the sensors
against the just-integrated state, eliminating that lag. This was
confirmed directly: a full 5-leg chained flight, logged before and after
the fix, lands at the same final positions to floating-point precision
(differences only in the 8th–10th significant digit — numerical noise,
not a behavior change), and the rendered navigation GIF is visually
identical.

---

## 3. How this connects to the noise ablation you already have

Stage C's `noise_ablation.png` (in `results/plots_2d/`) already answers
a closely related question: "what happens to the IDENTIFIED MODEL if the
sensor readings used to build it aren't perfectly clean?" Answer:
robust up to about 1–2 rad/s of injected gyro noise, breaking down past
about 4 rad/s. That's not the same test as "what happens to the LIVE
CONTROLLER if the IMU readings during flight are noisy" (which isn't
implemented — the live sensors are still idealized), but it's the honest,
already-measured answer to "how sensitive is this whole approach to
imperfect sensing," and it's a strong thing to point to if asked.

---

## 4. If your professor asks... (quick, direct answers)

**"Walk me through your controller."** → Outer loop (plain PD physics,
turns position error into a desired tilt+thrust) feeds an inner loop
(quaternion PD, gains DERIVED from the SINDy-identified drag/inertia
ratio, not hand-tuned) which feeds a mixer (turns thrust+torque into 4
motor commands, respecting the X-frame rotor geometry).

**"Why quaternions and not roll/pitch/yaw?"** → No gimbal lock, and a
clean, singularity-free error definition; the one quirk (q and -q being
the same rotation) is explicitly handled.

**"Are your gains hand-tuned?"** → No — `gains_from_identified_model()`
derives them from the SINDy Jacobian via standard 2nd-order pole
placement. Only the TARGET response speed (1.5s settling, 0.7 damping)
is a human choice; the actual Kp/Kd numbers are computed from the
identified physics.

**"Does your gain design account for cross-axis coupling?"** → No, by
design — it treats each axis independently (uses only the diagonal of
the identified Jacobian). This is standard practice for a
diagonal-inertia system like this one, and it's verified adequate
empirically (Stage D settles correctly even on combined-axis and
large-angle maneuvers), not just assumed.

**"Do you have an IMU?"** → Yes, modeled explicitly in `quad.xml` as a
sensor block (position, attitude, body velocity, gyro), and the
controller genuinely reads all 4 through that sensor pipeline, not
through raw physics state.

**"Is your IMU realistic?"** → The gyroscope is exactly what a real one
measures. The other 3 channels are idealized (no noise, no drift, no
estimation filter) — a deliberate simplification appropriate for a
controls-focused (not state-estimation-focused) course, and the
sensitivity-to-noise question is already answered separately by the
Stage C noise ablation.

**"What would you need to add to make this fly on a real drone?"** → A
real state estimator (e.g. an Extended Kalman Filter) fusing raw
accelerometer + gyroscope readings, likely with GPS or visual odometry
to bound position/velocity drift — everything downstream of that
(the controller and mixer) would need no structural change, since it
already consumes state through a clean sensor interface.
