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
for _p in ("05_voice_interface", "04_control", "01_simulation", "01_simulation/models"):
    sys.path.insert(0, str(ROOT / _p))

from commands import classify, target_offset, command_vocabulary, COMMANDS, WAYPOINT_OFFSET_M, VERTICAL_OFFSET_M  # noqa: E402
from trajectory import POSITION_TOLERANCE_M  # noqa: E402
from run_trials import run_all, run_chained  # noqa: E402
from run_preemption import run_preemption  # noqa: E402
from city import OBSTACLE_FACE_X, CITY_OBJECTS, clamp_target_to_safe_zone  # noqa: E402
from flight import DT, FlightController  # noqa: E402


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


def test_classify_natural_synonyms():
    """Regression test for a real live-mic misroute: saying "go straight"
    (a completely natural way to say "forward") got classified as "right"
    - "straight" fuzzy-matches "right" at a high ratio purely by sharing
    the letters r-i-g-h-t as a near-contiguous run, and nothing else in
    the sentence scored higher, so "pick the best-scoring token" alone
    could not fix this (the coincidental match WAS the best one). Needs
    an explicit synonym table, checked before fuzzy matching."""
    assert classify("go straight") == "forward"
    assert classify("straight") == "forward"
    assert classify("reverse") == "back"
    assert classify("ascend") == "up"
    assert classify("descend") == "down"
    # 2026-09-09 additions: a few more natural phrasings, same pattern.
    assert classify("pause") == "stop"
    assert classify("lift off") == "up"
    assert classify("go higher") == "up"
    assert classify("drop down") == "down"
    assert classify("stay there") == "hover"
    assert classify("hold position") == "hover"


def test_classify_rejects_ambient_speech():
    """Regression test for false-positive dispatches from background/
    ambient speech picked up by an open mic (real transcripts from a live
    session, not synthetic): unrelated English words must not coincidentally
    fuzzy-match a command. "laptop"->"stop" and "their"->"hover" both
    scored exactly 0.6 (the original cutoff) purely from partial letter
    overlap ("lap-TOP", "t-HEI-R"/"hov-ER"), while genuine matches (typos
    of the actual command word) score 0.85+ - the cutoff was raised to
    0.7 to sit in the gap between those two populations."""
    assert classify("you going to to do some the problem with me") is None
    assert classify("i doubt in a laptop about him returning to funny") is None
    assert classify("a little deployed and are they look their relatives") is None


def test_city_buildings_clear_flight_corridor():
    """Safety check: no building may overlap the drone's full flight
    corridor (spawn at the origin, +-WAYPOINT_OFFSET_M laterally in X/Y,
    hover altitude +-VERTICAL_OFFSET_M plus headroom). If a future city
    edit places a building inside this box, the drone would clip through
    it on an ordinary forward/back/left/right/up/down command - this must
    fail loudly, not be discovered by watching the demo."""
    lateral = WAYPOINT_OFFSET_M
    z_lo, z_hi = 1.0 - VERTICAL_OFFSET_M - 0.3, 1.0 + VERTICAL_OFFSET_M + 0.3
    for obj in CITY_OBJECTS:
        if obj.kind not in ("building", "tower"):
            continue  # parks/trees/pad are flat map marks, not obstacles
        x_overlap = (obj.x - obj.half_x) < lateral and (obj.x + obj.half_x) > -lateral
        y_overlap = (obj.y - obj.half_y) < lateral and (obj.y + obj.half_y) > -lateral
        z_overlap = obj.height > z_lo  # objects sit on the ground (z=0 base)
        assert not (x_overlap and y_overlap and z_overlap), (
            f"{obj.name} at ({obj.x},{obj.y}) with half-size "
            f"({obj.half_x},{obj.half_y}) intrudes on the +-{lateral}m "
            f"flight corridor - move it in city.py")


def test_command_vocabulary_matches_synonyms():
    """Regression test for print_commands_guide.py (the presentation-day
    commands guide, RUN_EVERYTHING.bat): command_vocabulary() must return
    every fixed command, and its synonyms must match what classify()
    actually accepts - a guide that drifts from the real vocabulary would
    be worse than no guide at all."""
    vocab = command_vocabulary()
    assert set(vocab.keys()) == set(COMMANDS)
    assert "straight" in vocab["forward"]
    assert "reverse" in vocab["back"]
    assert "pause" in vocab["stop"]
    assert vocab["left"] == []   # no synonyms defined for this command
    for cmd, synonyms in vocab.items():
        for word in synonyms:
            assert classify(word) == cmd  # guide never lists a stale synonym


def test_latest_parquet_uses_mtime_not_filename_sort(tmp_path, monkeypatch):
    """Regression test for a real bug found while wiring live-demo logging:
    stage_e_voice_sessions/ holds three filename prefixes (stage_e_trials_,
    stage_e_chained_, stage_e_preemption_). Alphabetically 't' > 'c' > 'p',
    so `sorted(glob(...))[-1]` always returns a trials-prefixed file if one
    exists, REGARDLESS of which file was actually written most recently -
    silently showing an old headless test run instead of a live demo
    session's fresh log. Must pick by real file modification time."""
    import os
    import plot_2d_telemetry as p2d

    monkeypatch.setattr(p2d, "SESSIONS_DIR", tmp_path)
    older_alphabetically_last = tmp_path / "stage_e_trials_20260101T000000Z.parquet"
    newer_alphabetically_first = tmp_path / "stage_e_chained_20260101T000000Z.parquet"
    older_alphabetically_last.write_bytes(b"x")
    newer_alphabetically_first.write_bytes(b"x")
    now = __import__("time").time()
    os.utime(older_alphabetically_last, (now - 100, now - 100))
    os.utime(newer_alphabetically_first, (now, now))

    assert p2d._latest_parquet() == newer_alphabetically_first


def test_live_flight_logger_records_and_saves(tmp_path):
    """Unit test for demo.py's LiveFlightLogger - the fix for demo.py
    logging NOTHING during an interactive session, which meant the
    "after" plots could never reflect what a user actually just flew
    live. A leg is closed out using the position/attitude given at the
    NEXT dispatch (or at on_exit for the last one). Also pins the
    2026-09-09 follow-up fix: the file is written after EVERY completed
    leg, not only at the end - a user checking results while the demo
    was still running previously saw nothing from the current session."""
    from demo import LiveFlightLogger

    logger = LiveFlightLogger(tmp_path)
    logger.on_dispatch("forward", "go forward", np.array([0.0, 0.0, 1.0]),
                        np.array([1.5, 0.0, 1.0]), classify_ms=0.1, trajectory_ms=0.2,
                        att_err_deg_before=0.0)
    logger.on_step(0.30)
    logger.on_step(0.20)
    # Dispatching "left" closes out the "forward" leg using THIS position -
    # and must already be on disk now, before on_exit is ever called.
    logger.on_dispatch("left", "go left", np.array([1.5, 0.0, 1.0]),
                        np.array([1.5, 1.5, 1.0]), classify_ms=0.1, trajectory_ms=0.2,
                        att_err_deg_before=0.01)
    assert logger.path.exists()
    mid_session_df = pd.read_parquet(logger.path)
    assert list(mid_session_df["command"]) == ["forward"]  # only the closed leg so far

    logger.on_exit(np.array([1.5, 1.5, 1.0]), 0.02)

    path = logger.save()
    assert path == logger.path and path.exists()
    df = pd.read_parquet(path)
    assert list(df["command"]) == ["forward", "left"]
    assert df["achieved_px"].iloc[0] == pytest.approx(1.5)  # closed at the 2nd dispatch's position
    assert bool(df["success"].iloc[0]) is True               # reached its target exactly
    assert df["latency_control_dispatch_ms"].iloc[0] == pytest.approx(0.25)  # mean of the 2 steps
    assert df["achieved_px"].iloc[1] == pytest.approx(1.5)   # closed at on_exit's position
    assert df["achieved_py"].iloc[1] == pytest.approx(1.5)


def test_live_flight_logger_save_returns_none_if_nothing_flown(tmp_path):
    from demo import LiveFlightLogger
    assert LiveFlightLogger(tmp_path).save() is None


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


def test_command_preemption_mid_flight():
    """A NEW command during transit must immediately override the old one,
    not queue behind it. In the city scene the forward path now runs toward
    a visual obstacle tower (face at OBSTACLE_FACE_X), and the scenario
    that makes the override visible is: forward at t=0, back re-dispatched
    at t=0.7s while still in transit. The drone must reverse before ever
    reaching the forward waypoint (let alone the tower face) and hold the
    NEW back waypoint."""
    # Scenario sanity: the tower stands beyond the forward waypoint, so a
    # drone that obeys only its original forward plan would fly at the
    # building; preemption is what keeps it clear.
    assert OBSTACLE_FACE_X > WAYPOINT_OFFSET_M

    s = run_preemption()
    assert s["success"]
    assert s["peak_x_m"] < OBSTACLE_FACE_X
    assert s["peak_x_m"] < WAYPOINT_OFFSET_M          # never reached the waypoint
    assert s["clearance_to_tower_m"] > 0.0
    assert s["reversal_latency_s"] is not None
    assert s["reversal_latency_s"] <= 0.5             # immediate override
    assert s["final_position_error_m"] <= POSITION_TOLERANCE_M
    assert np.allclose(s["final_position_m"], s["back_target"], atol=POSITION_TOLERANCE_M)
    df = pd.read_parquet(s["log_path"])
    idx = int(round(s["preempt_at_s"] / 0.002))  # round(), not int(): 0.7/0.002 = 349.99...
    assert df["command"].iloc[:idx].eq("forward").all()
    assert df["command"].iloc[idx:].eq("back").all()


def test_format_heard_entry():
    """Regression test for the 2026-09-09 fix: demo.py's main loop used to
    silently drop every heard-but-unrecognized transcript (cmd is None), so
    a user with a perfectly working mic and classifier saw nothing at all
    and reasonably concluded the whole system was broken. format_heard_entry
    is the pure formatting piece of that fix - every heard utterance, hit
    or miss, must produce a visible line."""
    from demo import format_heard_entry
    assert format_heard_entry("go right", "right") == "'go right' -> RIGHT"
    assert format_heard_entry("banana", None) == "'banana' -> NOT RECOGNIZED"


def test_vosk_source_exposes_partial_transcript():
    """Regression test: VoskSource captured the in-progress ASR partial
    result (self._partial) but never exposed it, so the HUD could only ever
    show text after a full utterance finished - no "as you speak" feedback.
    get_partial() must parse the same JSON shape Vosk's PartialResult()
    returns and be safe to call before any audio has arrived."""
    import json
    from voice_input import VoskSource

    src = VoskSource.__new__(VoskSource)  # skip __init__ (needs a real mic/model)
    src._lock = __import__("threading").Lock()
    src._partial = ""
    assert src.get_partial() == ""  # silence before any audio -> not an error

    src._partial = json.dumps({"partial": "go rig"})
    assert VoskSource.get_partial(src) == "go rig"


def test_clamp_target_to_safe_zone_blocks_building_penetration():
    """Unit test for the clamp itself, against the real obstacle tower
    (face at OBSTACLE_FACE_X). A raw target of x=3.0m (what two chained
    "forward"s actually produced live) sits dead center of the tower -
    the clamp must stop short of the face, not accept it."""
    p0 = np.array([1.5, 0.0, 1.0])   # where the drone is after ONE forward
    p1_raw = np.array([3.0, 0.0, 1.0])  # where a SECOND forward would send it
    clamped = clamp_target_to_safe_zone(p0, p1_raw)
    assert clamped[0] < OBSTACLE_FACE_X          # never reaches the tower face
    assert clamped[0] > p0[0]                     # still makes real progress
    assert np.allclose(clamped[1:], [0.0, 1.0])   # y/z untouched

    # A target that never comes near any building must pass through as-is.
    untouched = clamp_target_to_safe_zone(np.array([0.0, 0.0, 1.0]),
                                           np.array([0.0, 1.5, 1.0]))
    assert np.allclose(untouched, [0.0, 1.5, 1.0])

    # hover/stop (zero-length segment) must not divide by zero or move.
    same = clamp_target_to_safe_zone(np.array([1.5, 0.0, 1.0]),
                                      np.array([1.5, 0.0, 1.0]))
    assert np.allclose(same, [1.5, 0.0, 1.0])


def test_dispatch_clamps_chained_forward_into_tower():
    """Regression test for a REAL live-session incident (screenshot +
    console log): a user said "go straight" twice in a row, both correctly
    classified as "forward". dispatch() applies each offset from the
    drone's CURRENT position by design (see run_trials.run_chained), so
    two forwards compounded to a target of x=3.0m - dead center of the
    obstacle tower - and since buildings are non-collidable, nothing
    would have physically stopped it. dispatch() must now clamp the
    second target short of the tower instead of accepting it."""
    fc = FlightController()
    fc.dispatch("forward")
    for _ in range(int(round(3.0 / DT))):  # let it actually fly most of the way there
        fc.step()
    assert fc.position()[0] > 1.0  # sanity: it really moved forward first

    lat = fc.dispatch("forward")  # second forward -> raw target would be ~3.0
    target = lat["target"]
    assert target[0] < OBSTACLE_FACE_X       # clamped short of the tower
    assert target[0] > fc.position()[0]      # still real forward progress commanded


def test_camera_toggle_button_does_not_depend_on_window_focus():
    """Regression test for a real user complaint: pressing 'B' only toggles
    the camera if the MuJoCo VIEWER window (a separate native window from
    the dashboard) has OS keyboard focus - an easy, unannounced way for the
    hotkey to silently do nothing, the same class of "invisible window" bug
    this project already hit once with the old Tk button row. The fix is a
    clickable "TOGGLE CAMERA" button on the dashboard, which - like the 8
    existing command buttons - works via a mouse click on ITS OWN window
    and has no dependency on which window last had focus."""
    import matplotlib
    matplotlib.use("Agg")
    from demo import build_dashboard, update_dashboard, TelemetryHistory
    from flight import FlightController

    fc = FlightController()
    hist = TelemetryHistory(maxlen=10)
    fig, axes, lines, status_text = build_dashboard()
    try:
        assert hasattr(fig, "_pending_view_toggle")
        assert fig._pending_view_toggle["toggle"] is False

        # Fire the button's registered click callback - exactly what
        # matplotlib's own event loop does once its (unchanged, already
        # relied-upon) hit-testing confirms a click landed in this axes.
        btn = fig._buttons[-1]
        btn._observers.process("clicked", None)
        assert fig._pending_view_toggle["toggle"] is True

        update_dashboard(fig, axes, lines, status_text, hist, fc, "hover", {}, cam="birdseye")
        assert "CAM: BIRDSEYE" in status_text.get_text()
        update_dashboard(fig, axes, lines, status_text, hist, fc, "hover", {}, cam="chase")
        assert "CAM: CHASE" in status_text.get_text()
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)


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
