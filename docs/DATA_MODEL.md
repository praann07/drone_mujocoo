# Data Model — logged data schemas

All logged data is Parquet (`pandas`/`pyarrow`), one file per trial/run.
File naming: `<stage>_<trial_type>_<axis-or-command>_<trial_index>_<utc_timestamp>.parquet`.

## 1. Raw simulator output (Stage A)

One row per simulation timestep, per excitation trial.

| Column | Type | Units | Notes |
|---|---|---|---|
| `t` | float64 | s | Simulation time, starts at 0.0 per trial |
| `step` | int64 | — | Integer step index |
| `trial_id` | string | — | Unique id, e.g. `A_roll_prbs_003` |
| `axis` | string | — | `roll`/`pitch`/`yaw`/`combined` |
| `signal_type` | string | — | `prbs`/`chirp` |
| `px, py, pz` | float64 | m | World-frame position |
| `vx, vy, vz` | float64 | m/s | World-frame velocity |
| `qw, qx, qy, qz` | float64 | — | Attitude quaternion, Hamilton, scalar-first, body→world |
| `wx, wy, wz` | float64 | rad/s | Body-frame angular velocity |
| `u1, u2, u3, u4` | float64 | N | Per-rotor thrust command (excitation input) |

Sample rate: fixed per `TDD.md` §1 (starting assumption 200 Hz) — constant
within a trial, recorded once per file as Parquet metadata (`sample_rate_hz`).

## 2. Smoothed/differentiated dataset fed into SINDy/DMDc (Stage B)

Derived from §1, one row per timestep, per trial. Adds:

| Column | Type | Units | Notes |
|---|---|---|---|
| `theta_x, theta_y, theta_z` | float64 | rad | Body-frame rotation vector relative to trial's current reference attitude (TDD.md §4) |
| `theta_reset` | bool | — | True on the row where a reference-attitude reset occurred |
| `theta_dot_x, theta_dot_y, theta_dot_z` | float64 | rad/s | Savitzky-Golay derivative of `theta_*` |
| `omega_dot_x, omega_dot_y, omega_dot_z` | float64 | rad/s² | Savitzky-Golay derivative of `wx, wy, wz` |
| `sg_window, sg_polyorder` | int64 | — | Savitzky-Golay parameters used, recorded per file for reproducibility |

Rows within a reset transient window (a few samples after `theta_reset`)
are flagged, not silently dropped — Stage C must account for them
explicitly (TDD.md §4).

## 3. Fit / held-out split scheme

**Split is defined at the trial level, never within a single continuous
trajectory.** Each `trial_id` from Stage A is assigned exactly one of:

- `fit` — used to fit SINDy/DMDc.
- `held_out` — never seen during fitting; used only for Stage B/C
  evaluation.

The assignment is stored in a single file,
`data/processed/trial_split.parquet`, with columns `trial_id, axis,
signal_type, split`. This file is written once after Stage A data
collection and before any Stage B fitting begins, and is treated as
read-only afterward — changing it after fitting has started would
invalidate the gate.

**Confirmation this is trial-level, not time-slice-based:** each trial_id
is a complete, independently-generated excitation run (its own PRBS/chirp
realization, its own noise realization in simulation if any is injected).
A `held_out` trial shares no timesteps, no random seed, and no excitation
realization with any `fit` trial. This is checked programmatically — a
`tests/test_data_split.py` assertion that no `held_out` trial_id appears in
the fit set and vice versa — before Stage B fitting is allowed to run.

Recommended minimum split: at least 2 independent trials per axis per
signal type held out, remainder used for fitting — exact counts recorded
in `trial_split.parquet` once Stage A trial counts are known.

## 4. Validation-run outputs (Stage C)

One row per timestep, per held-out trial, per model evaluated.

| Column | Type | Units | Notes |
|---|---|---|---|
| `t` | float64 | s | |
| `trial_id` | string | — | Held-out trial being evaluated |
| `model` | string | — | `sindy` or `dmdc` |
| `eval_type` | string | — | `one_step_ahead`/`short_horizon`/`long_horizon` |
| `theta_x_true, theta_y_true, theta_z_true` | float64 | rad | Ground truth |
| `theta_x_pred, theta_y_pred, theta_z_pred` | float64 | rad | Model prediction |
| `omega_*_true, omega_*_pred` | float64 | rad/s | Same pattern for angular velocity |
| `error_norm` | float64 | — | Combined state error norm at this timestep |

## 5. Voice-session logs (Stage E)

One row per spoken-command trial.

| Column | Type | Units | Notes |
|---|---|---|---|
| `trial_id` | string | — | |
| `timestamp_utc` | string | ISO8601 | |
| `transcript` | string | — | Raw ASR output |
| `asr_confidence` | float64 | 0–1 | |
| `dispatched_command` | string | — | One of the 6 fixed commands, or `unrecognized` |
| `start_px, start_py, start_pz` | float64 | m | Vehicle position when the command was dispatched (trajectory origin) |
| `target_px, target_py, target_pz` | float64 | m | Target waypoint (start position + command offset, or start position itself for hover/stop) |
| `target_qw, qx, qy, qz` | float64 | — | Attitude setpoint stream target at trial end (yaw held constant, roll/pitch from outer loop) — logged as the final value; the full stream is in the Stage D telemetry log, not duplicated here |
| `achieved_px, achieved_py, achieved_pz` | float64 | m | Position actually reached at trial end |
| `achieved_qw, qx, qy, qz` | float64 | — | Attitude actually reached at trial end |
| `position_error_m` | float64 | m | Distance between target and achieved position |
| `attitude_error_deg` | float64 | deg | Angular error between target and achieved attitude at trial end |
| `success` | bool | — | Whether `position_error_m` is within the Stage D outer-loop tolerance |
| `latency_vad_ms` | float64 | ms | |
| `latency_asr_ms` | float64 | ms | |
| `latency_classify_ms` | float64 | ms | |
| `latency_trajectory_gen_ms` | float64 | ms | Time to generate the point-to-point trajectory plan |
| `latency_control_dispatch_ms` | float64 | ms | |
| `latency_total_ms` | float64 | ms | |
