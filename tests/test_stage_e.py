"""Stage E tests (docs/ENGINEERING_PLAN.md Stage E).

Validates the command classifier, the world-frame waypoint map, and the
end-to-end flight pipeline (classifier -> waypoint -> Stage D cascade ->
MuJoCo plant) headlessly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in ("05_voice_interface", "04_control"):
    sys.path.insert(0, str(ROOT / _p))

from commands import classify, target_offset, COMMANDS, WAYPOINT_OFFSET_M, VERTICAL_OFFSET_M  # noqa: E402
from trajectory import POSITION_TOLERANCE_M  # noqa: E402
from run_trials import run_all, run_chained  # noqa: E402


def test_classify_known():
    assert classify("go forward please") == "forward"
    assert classify("move back") == "back"
    assert classify("go left now") == "left"
    assert classify("turn right and go") == "right"
    assert classify("hover in place") == "hover"
    assert classify("stop moving") == "stop"


def test_classify_fuzzy():
    assert classify("foward") == "forward"      # typo, difflib
    assert classify("lef") == "left"


def test_classify_none():
    assert classify("") is None
    assert classify("the weather is nice") is None


def test_classify_global_best_not_first_above_cutoff():
    """Regression test for a real misroute: the filler word "to"
    fuzzy-matches "stop" (ratio 0.667) above the 0.6 cutoff, and a
    first-token-wins classifier returned "stop" before ever reaching the
    later, exact "left" (ratio 1.0). classify() must score every token
    and keep the globally best match, not the first one above cutoff."""
    assert classify("five meters to left") == "left"
    assert classify("finally does to left") == "left"
    assert classify("move right") == "right"
    assert classify("go forward") == "forward"
    assert classify("stop") == "stop"
    assert classify("go up") == "up"
    assert classify("go down") == "down"


def test_waypoint_offsets():
    assert np.allclose(target_offset("forward"), [WAYPOINT_OFFSET_M, 0, 0])
    assert np.allclose(target_offset("back"), [-WAYPOINT_OFFSET_M, 0, 0])
    assert np.allclose(target_offset("left"), [0, WAYPOINT_OFFSET_M, 0])
    assert np.allclose(target_offset("right"), [0, -WAYPOINT_OFFSET_M, 0])
    assert np.allclose(target_offset("up"), [0, 0, VERTICAL_OFFSET_M])
    assert np.allclose(target_offset("down"), [0, 0, -VERTICAL_OFFSET_M])
    assert np.allclose(target_offset("hover"), [0, 0, 0])
    assert np.allclose(target_offset("stop"), [0, 0, 0])


def test_headless_pipeline_success():
    df, path = run_all(n_trials=6, seed=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 6
    # success: every trial reaches within the outer-loop tolerance
    assert df["success"].all()
    # required DATA_MODEL sec5 columns present
    for col in ["transcript", "dispatched_command", "position_error_m",
                "attitude_error_deg", "latency_classify_ms",
                "latency_trajectory_gen_ms", "latency_total_ms", "success"]:
        assert col in df.columns
    # all dispatched commands are in the fixed vocabulary
    assert df["dispatched_command"].isin(list(COMMANDS)).all()
    assert df["position_error_m"].max() <= POSITION_TOLERANCE_M + 1e-9


def test_chained_flight_no_reset_between_legs():
    """run_chained must never reset state between legs - each leg's start
    position must equal the previous leg's achieved position exactly,
    proving no hidden fc.reset() sneaks in between commands."""
    df, path = run_chained(sequence=("forward", "left", "back"), hold_s=3.0)
    assert len(df) == 3
    for i in range(1, len(df)):
        prev_end = df.iloc[i - 1]["achieved_position"]
        this_start = df.iloc[i]["start_position"]
        assert np.allclose(prev_end, this_start), (
            "chained trial reset state between legs - defeats the point "
            "of testing compounding error like the GIF demo does")
    # final leg's error is the number that matters for a chained sequence
    assert df["position_error_m"].iloc[-1] <= POSITION_TOLERANCE_M + 0.05


def test_telemetry_dashboard_hud():
    """Verify live telemetry dashboard creates 3 strip charts, accepts
    streaming data in TelemetryHistory, and updates line objects & HUD readout."""
    import matplotlib.pyplot as plt
    from demo import TelemetryHistory, build_dashboard, update_dashboard
    from flight import FlightController

    fc = FlightController()
    hist = TelemetryHistory(maxlen=50)
    for i in range(10):
        fc.step()
        hist.push(i * 0.002, 0.05 * i, 0.01 * i, 9.81 + 0.1 * i)

    fig, axes, lines, status_text = build_dashboard()
    try:
        assert len(axes) == 3
        assert len(lines) == 3
        meta = {
            "transcript": "forward",
            "latencies_ms": {"classify": 0.08, "trajectory": 0.03, "total": 0.25}
        }
        update_dashboard(fig, axes, lines, status_text, hist, fc, "forward", meta)

        line_att, line_pos, line_ctrl = lines
        assert len(line_att.get_xdata()) == 10
        assert len(line_pos.get_ydata()) == 10
        assert len(line_ctrl.get_ydata()) == 10
        assert "TELEMETRY HUD" in status_text.get_text()
        assert "FORWARD" in status_text.get_text()
    finally:
        plt.close(fig)
