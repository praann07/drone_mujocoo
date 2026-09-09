# What's in this folder

**Don't read these directly — they're raw console output, not built for
humans.** For the actual readable results, see **[`../RESULTS.md`](../RESULTS.md)**.

Each file is just what printed to the screen when that pipeline stage ran —
kept as proof it genuinely ran and passed, not as something to read start to
finish. If your professor asks "prove this actually happened," this is the
receipt.

| File | What it's proof of |
|---|---|
| `excitation.log` | Stage A — the 3D sim was excited/tested (the raw flight data step) |
| `identification.log` | Stage B — SINDy/DMDc were fit to that data |
| `validation.log` | Stage C — the fitted model was validated, not just fit |
| `inner_loop.log` | Stage D — the attitude controller was tested |
| `cascade.log` | Stage D — the full position-tracking controller was tested |
| `trials.log` | Stage E — 24 isolated + 5 chained voice-command flights |
| `preemption.log` | Stage E — the "change command mid-flight" test |
| `render_nav.log`, `render_preempt.log` | The two demo GIFs were rendered |
| `gauntlet.log` | All 67 maneuvers in the full test gauntlet |
| `save_plots.log` | This `results/` folder was assembled from the run above |
| `pytest.log` | The full automated test suite (43 tests) |

If a step ever fails, this is where to look — the exact error will be in
the matching file above, not in `RESULTS.md`.
