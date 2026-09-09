# What's in this folder

**Don't read these directly — they're raw backing data, not built for humans.**
For the actual readable results (tables, key numbers, pass/fail), see
**[`../RESULTS.md`](../RESULTS.md)** instead. This folder exists so every
number in that file is traceable back to a real logged run, not just a claim.

If you do open one, here's what each column means:

## `identification_results.csv` (Stage B — did the model learn real physics?)
One row per model/regime combination.
| Column | Meaning |
|---|---|
| `model` | `sindy` or `dmdc` |
| `regime` | `full_envelope` (all 6 commands) or `near_hover` (small angles only) |
| `deriv_r2` | How well it predicts the rate of change (1.0 = perfect) |
| `step_nrmse` | One-step prediction error, normalized (lower = better) |
| `n_active_terms` | How many physics terms SINDy kept (fewer = simpler model) |

## `rollout_results.csv` (Stage C — does it stay accurate over a full flight, not just one step?)
One row per held-out test segment.
| Column | Meaning |
|---|---|
| `model`, `regime` | Same as above |
| `horizon` | `short` (~2s) or `long` (~11s) simulated flight |
| `trial_id` | Which excitation recording this segment came from |
| `divergence_t` | Time the prediction went unstable (blank = never did) |
| `final_error`, `max_error` | How far off the prediction was at the end / at its worst |

## `inner_loop_results.csv` (Stage D — does the attitude controller actually stabilize the drone?)
One row per test maneuver (step turns, big steps, ramps).
`settling_s` = time to stabilize. `sse_deg` = steady-state pointing error in
degrees. `sindy_*` vs `mujoco_*` = controller running on the identified
model vs. the real physics simulator — these should closely agree.

## `cascade_results.csv` (Stage D — does "fly to this point" actually work?)
One row per voice-command waypoint (forward/back/left/right/hover/stop).
Same `sindy_*` vs `mujoco_*` comparison, but for the full position-tracking
loop, not just attitude.

## `*_latest.parquet` (Stage E — what actually happened during a voice/typed flight)
Raw per-command flight logs (not spreadsheet-friendly — open with
`pandas.read_parquet(...)` in Python, not Excel). `trials_latest` = 24
isolated single-command tests. `chained_latest` = a continuous multi-command
flight. `preemption_latest` = the "changed my mind mid-flight" test.

## `sindy_fitted_model.npz`
The actual frozen, gate-passed SINDy model (numeric coefficients). Not
readable directly — this is what the live demo loads to fly.
