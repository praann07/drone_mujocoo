"""Stage E live 3D navigation demo (docs/ENGINEERING_PLAN.md Stage E task 4).

Runs the real-time FlightController inside a MuJoCo passive viewer (the
genuine 3D rigid-body view), accepts commands from THREE input sources -
a VoskSource (microphone), a TextSource (typed/scripted), and a row of
click-to-fly BUTTONS in the dashboard window itself - and drives a live
telemetry DASHBOARD beside it: three rolling strip charts (attitude
error, position error, control effort) plus a compact text readout,
updated every control step - not a text box refreshed a few times a
second. This is the direct fix for the "stats and metrics must be
visualized beside the sim" gate: the numbers move in real time while the
drone flies, not as a static plot generated after the fact.

The button row exists because typed control needs a live terminal window
(this script can't read your keyboard if launched by something other than
your own interactive shell) and spoken control needs a working
microphone - neither is guaranteed, but the buttons work every time,
in the same window, regardless of --source.

Requires a display + (for Vosk) a microphone and a downloaded Vosk model
(see voice_input.VoskSource). In this headless environment use
run_trials.py for the validated >=20-trial success-rate gate, or
render_offscreen.py for a headless recording of the same dashboard idea
baked into a GIF overlay.

    .venv\\Scripts\\python.exe 05_voice_interface\\demo.py --source text
    .venv\\Scripts\\python.exe 05_voice_interface\\demo.py --source vosk
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
for _p in ("05_voice_interface", "01_simulation/models", "04_control"):
    sys.path.insert(0, str(ROOT / _p))
from flight import DT, FlightController, IDENTITY_Q  # noqa: E402
from voice_input import TextSource, VoskSource  # noqa: E402
from city import OBSTACLE_FACE_X  # noqa: E402
from trajectory import POSITION_TOLERANCE_M  # noqa: E402

SESSIONS_DIR = ROOT / "data" / "processed" / "stage_e_voice_sessions"

# Rolling window length for the strip charts, in seconds of flight time.
HISTORY_S = 8.0
HISTORY_LEN = int(round(HISTORY_S / DT))
# Dashboard redraw cadence - the underlying control loop still runs every
# DT (0.002s); redrawing matplotlib at that rate would be the bottleneck,
# not the control law, so the chart is repainted at ~20 Hz while every
# single control step still gets pushed into the rolling buffers first.
DASHBOARD_REDRAW_S = 0.05


class TelemetryHistory:
    """Rolling buffers of the per-step signals the dashboard plots.
    Fixed-length deques so the strip charts show a moving window rather
    than growing without bound over a long demo session."""

    def __init__(self, maxlen: int = HISTORY_LEN):
        self.t = deque(maxlen=maxlen)
        self.att_err_deg = deque(maxlen=maxlen)
        self.pos_err_m = deque(maxlen=maxlen)
        self.control_effort_n = deque(maxlen=maxlen)

    def push(self, t: float, att_err_deg: float, pos_err_m: float, control_effort_n: float):
        self.t.append(t)
        self.att_err_deg.append(att_err_deg)
        self.pos_err_m.append(pos_err_m)
        self.control_effort_n.append(control_effort_n)


def format_heard_entry(transcript: str, cmd: str | None) -> str:
    """Pure formatting for one heard utterance, recognized or not - factored
    out of the main loop so this is unit-testable without a mic or display.
    This is the direct fix for "I said something and nothing happened": every
    heard utterance gets one of these lines, never silently dropped."""
    result = cmd.upper() if cmd else "NOT RECOGNIZED"
    return f"{transcript!r} -> {result}"


class LiveFlightLogger:
    """Logs every dispatched command in an INTERACTIVE demo.py session to
    the exact same `stage_e_chained_*.parquet` schema `run_trials.py`'s
    `run_chained()` writes for headless runs - previously demo.py logged
    NOTHING, so `plot_2d_telemetry.py`/`plot_3d_trajectory.py` (which just
    read the newest matching parquet) could only ever show an old headless
    test run, never what a user actually just flew live. Matching the
    filename prefix (`stage_e_chained_`) and column set means both plot
    scripts pick this up automatically with zero changes to them.

    One row per dispatched command ("leg"), closed out at the moment the
    NEXT command is dispatched (or the demo exits) using whatever position/
    attitude was actually reached by then - this is the natural interactive
    analogue of run_chained()'s fixed hold-then-measure, since a live
    session has no fixed per-command duration.

    Written to disk after EVERY completed leg, not only once at exit -
    the earlier version only saved on exit, so SHOW_RESULTS.bat run while
    the demo window was still open showed nothing from the current
    session at all (only old, unrelated files) even though the user had
    already flown several commands. One filename is picked ONCE, at
    construction, and grows in place - so the file's own timestamp
    reflects when the session STARTED and repeated writes never leave
    behind a trail of partial files."""

    def __init__(self, out_dir: Path):
        self._out_dir = out_dir
        self._rows: list[dict] = []
        self._pending: dict | None = None
        self._step_latencies_ms: list[float] = []
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = out_dir / f"stage_e_chained_{stamp}.parquet"

    def on_dispatch(self, command: str, transcript: str, p_before: np.ndarray,
                     target: np.ndarray, classify_ms: float, trajectory_ms: float,
                     att_err_deg_before: float) -> None:
        self._finalize(p_before, att_err_deg_before)
        self._pending = {
            "leg_index": len(self._rows),
            "command": command,
            "transcript": transcript,
            "target_position": np.asarray(target, dtype=float),
            "classify_ms": classify_ms,
            "trajectory_ms": trajectory_ms,
        }
        self._step_latencies_ms = []

    def on_step(self, latency_ms: float) -> None:
        if self._pending is not None:
            self._step_latencies_ms.append(latency_ms)

    def _finalize(self, achieved_p: np.ndarray, att_err_deg: float) -> None:
        if self._pending is None:
            return
        p = self._pending
        achieved_p = np.asarray(achieved_p, dtype=float)
        pos_err = float(np.linalg.norm(achieved_p - p["target_position"]))
        self._rows.append({
            "leg_index": p["leg_index"],
            "command": p["command"],
            "transcript": p["transcript"],
            "target_px": float(p["target_position"][0]),
            "target_py": float(p["target_position"][1]),
            "target_pz": float(p["target_position"][2]),
            "achieved_px": float(achieved_p[0]),
            "achieved_py": float(achieved_p[1]),
            "achieved_pz": float(achieved_p[2]),
            "position_error_m": pos_err,
            "attitude_error_deg": float(att_err_deg),
            "success": bool(pos_err <= POSITION_TOLERANCE_M),
            "latency_classify_ms": p["classify_ms"],
            "latency_trajectory_gen_ms": p["trajectory_ms"],
            "latency_control_dispatch_ms": (
                float(np.mean(self._step_latencies_ms)) if self._step_latencies_ms else 0.0),
        })
        self._pending = None
        self._write()  # persist immediately - don't wait for the demo to exit

    def on_exit(self, achieved_p: np.ndarray, att_err_deg: float) -> None:
        """Close out whatever leg was still in progress when the demo ends."""
        self._finalize(achieved_p, att_err_deg)

    def _write(self) -> None:
        if not self._rows:
            return
        self._out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._rows).to_parquet(self.path, index=False)

    def save(self) -> Path | None:
        """Idempotent final write; returns the path, or None if nothing was
        ever dispatched this session (nothing worth saving)."""
        self._write()
        return self.path if self._rows else None


def build_dashboard():
    """Three live strip charts (attitude error, position error, control
    effort), a compact mission-control HUD readout, and a row of
    click-to-fly buttons (one per voice command) - all in the same
    window as the MuJoCo 3D viewport's neighbor.

    The buttons dispatch through the exact same `fc.dispatch(cmd)` call
    as TextSource/VoskSource (see main()'s loop) - they are a THIRD input
    source alongside typed and spoken commands, not a separate path that
    bypasses the cascade. They work regardless of --source, so clicking
    "FORWARD" here works identically whether you launched with
    --source text or --source vosk; this exists specifically so control
    doesn't depend on a working microphone or a terminal window with
    live stdin, both of which have been unreliable in this environment."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    # Aerospace Mission-Control HUD Theme: dark cockpit palette with neon signals
    fig = plt.figure(figsize=(6.2, 9.2), facecolor="#0b0e14")
    # top=0.62 (was 0.80): the HUD text block now also carries a live
    # LISTENING line and a scrolling heard/dispatch log, so it needs more
    # of the figure's vertical space than the original 3-line readout did.
    gs = fig.add_gridspec(3, 1, hspace=0.28, left=0.14, right=0.94, top=0.62, bottom=0.14)

    ax_att = fig.add_subplot(gs[0, 0], facecolor="#121722")
    ax_pos = fig.add_subplot(gs[1, 0], facecolor="#121722", sharex=ax_att)
    ax_ctrl = fig.add_subplot(gs[2, 0], facecolor="#121722", sharex=ax_att)

    # Attitude Error Chart (Neon Crimson/Coral)
    line_att, = ax_att.plot([], [], color="#ff5370", linewidth=1.8, label="Attitude Error")
    ax_att.set_ylabel("Att Err (deg)", color="#cdd6f4", fontsize=9, fontweight="bold")
    ax_att.tick_params(colors="#899bb0", labelsize=8)
    ax_att.grid(True, color="#232d3f", linestyle="--", linewidth=0.7, alpha=0.8)
    for spine in ax_att.spines.values():
        spine.set_color("#2a364f")

    # Position Error Chart (Cyber Cyan) + 0.15m tolerance threshold line
    line_pos, = ax_pos.plot([], [], color="#00e5ff", linewidth=1.8, label="Position Error")
    line_tol = ax_pos.axhline(0.15, color="#ffb86c", linestyle=":", linewidth=1.2, alpha=0.8, label="Gate Tol (0.15m)")
    ax_pos.set_ylabel("Pos Err (m)", color="#cdd6f4", fontsize=9, fontweight="bold")
    ax_pos.tick_params(colors="#899bb0", labelsize=8)
    ax_pos.grid(True, color="#232d3f", linestyle="--", linewidth=0.7, alpha=0.8)
    for spine in ax_pos.spines.values():
        spine.set_color("#2a364f")

    # Control Effort Chart (Neon Emerald)
    line_ctrl, = ax_ctrl.plot([], [], color="#22da6e", linewidth=1.8, label="Control Effort")
    ax_ctrl.set_ylabel("Ctrl Effort (N)", color="#cdd6f4", fontsize=9, fontweight="bold")
    ax_ctrl.set_xlabel("Time (s)", color="#cdd6f4", fontsize=9, fontweight="bold")
    ax_ctrl.tick_params(colors="#899bb0", labelsize=8)
    ax_ctrl.grid(True, color="#232d3f", linestyle="--", linewidth=0.7, alpha=0.8)
    for spine in ax_ctrl.spines.values():
        spine.set_color("#2a364f")

    # HUD Status Block Header
    status_text = fig.text(
        0.05, 0.98, "", family="monospace", fontsize=8.5,
        color="#00e5ff", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#121722", edgecolor="#2a364f", linewidth=1.2)
    )

    if fig.canvas.manager is not None:
        try:
            fig.canvas.manager.set_window_title("Flight Telemetry HUD — Data-Driven Control")
        except Exception:
            pass

    # Click-to-fly button row (docs/CLAUDE.md's 6 fixed voice commands).
    # A plain dict-on-fig (not a return value) so build_dashboard()'s
    # signature stays (fig, axes, lines, status_text) for existing callers
    # (main(), tests) - main()'s loop polls fig._pending_command each
    # iteration exactly like it polls src.get_command().
    pending_command = {"cmd": None}
    fig._pending_command = pending_command  # noqa: SLF001 (intentional, see above)

    button_specs = [("FORWARD", "forward"), ("BACK", "back"), ("LEFT", "left"),
                    ("RIGHT", "right"), ("UP", "up"), ("DOWN", "down"),
                    ("HOVER", "hover"), ("STOP", "stop")]
    n = len(button_specs)
    btn_w, gap = 0.145, 0.012
    total_w = n * btn_w + (n - 1) * gap
    start_x = 0.5 - total_w / 2
    buttons = []
    for i, (label, cmd) in enumerate(button_specs):
        ax_btn = fig.add_axes((start_x + i * (btn_w + gap), 0.03, btn_w, 0.055))
        btn = Button(ax_btn, label, color="#1b2333", hovercolor="#2a4a5f")
        btn.label.set_color("#00e5ff")
        btn.label.set_fontsize(8)
        btn.label.set_fontweight("bold")

        def _make_callback(command):
            def _on_click(_event):
                pending_command["cmd"] = command
            return _on_click

        btn.on_clicked(_make_callback(cmd))
        buttons.append(btn)

    # Camera toggle as a CLICKABLE button, not just the "B" hotkey. The
    # hotkey only reaches this window's key_callback when the MuJoCo
    # viewer window itself has OS keyboard focus - if focus is on the
    # dashboard, terminal, or anywhere else (an easy, unannounced mistake
    # to make with two windows open), pressing B silently does nothing,
    # exactly the class of "invisible window" bug this project already
    # hit once with the old Tk button row. A mouse click on THIS window's
    # own button has no such dependency, so it works regardless of which
    # window currently has focus - same reasoning as the 8 command
    # buttons above.
    pending_view_toggle = {"toggle": False}
    ax_view_btn = fig.add_axes((0.28, 0.095, 0.44, 0.035))
    view_btn = Button(ax_view_btn, "TOGGLE CAMERA  (chase ⇄ bird's-eye)",
                       color="#2a1f0a", hovercolor="#4a3a1a")
    view_btn.label.set_color("#ffb86c")
    view_btn.label.set_fontsize(7.5)
    view_btn.label.set_fontweight("bold")

    def _on_view_click(_event):
        pending_view_toggle["toggle"] = True

    view_btn.on_clicked(_on_view_click)
    buttons.append(view_btn)
    fig._pending_view_toggle = pending_view_toggle  # noqa: SLF001

    # Buttons must be kept referenced (matplotlib drops the click handler
    # if the Button object itself is garbage-collected) - stash on fig
    # alongside pending_command rather than returning a 5th value.
    fig._buttons = buttons  # noqa: SLF001

    import matplotlib
    if "agg" not in matplotlib.get_backend().lower():
        fig.show()
    return fig, (ax_att, ax_pos, ax_ctrl), (line_att, line_pos, line_ctrl), status_text


def update_dashboard(fig, axes, lines, status_text, hist: TelemetryHistory,
                      fc: FlightController, last_cmd: str, meta: dict,
                      listening: str = "", heard_log=None, cam: str = "chase"):
    import matplotlib
    import matplotlib.pyplot as plt
    ax_att, ax_pos, ax_ctrl = axes
    line_att, line_pos, line_ctrl = lines

    if hist.t:
        t = np.asarray(hist.t)
        line_att.set_data(t, hist.att_err_deg)
        line_pos.set_data(t, hist.pos_err_m)
        line_ctrl.set_data(t, hist.control_effort_n)
        for ax, series in ((ax_att, hist.att_err_deg), (ax_pos, hist.pos_err_m),
                           (ax_ctrl, hist.control_effort_n)):
            ax.set_xlim(t[0], max(t[-1], t[0] + 1e-3))
            lo, hi = min(series), max(series)
            pad = 0.1 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - pad, hi + pad)

    p, q = fc.position(), fc.attitude()
    cur_pos_err = hist.pos_err_m[-1] if hist.pos_err_m else 0.0
    cur_att_err = hist.att_err_deg[-1] if hist.att_err_deg else 0.0
    lat = meta.get("latencies_ms", {}) if meta else {}
    transcript = meta.get("transcript", "") if meta else ""
    status_mode = "TRACKING" if cur_pos_err > 0.01 else "HOVER/STEADY"
    dist_tower = OBSTACLE_FACE_X - p[0]

    # LISTENING shows the in-progress transcript AS it's being spoken (see
    # voice_input.VoskSource.get_partial) - "(silence)" is a real, visible
    # state distinct from "mic dead", which the old UI could never show.
    listening_line = listening if listening else "(silence)"
    log_entries = list(heard_log)[-6:] if heard_log else []
    log_block = "\n".join(f"│   {e}" for e in log_entries) if log_entries else "│   (nothing heard yet)"

    status_text.set_text(
        f"┌─ TELEMETRY HUD ── MODE: {status_mode:<12s} ── CMD: {last_cmd.upper():<7s} ── CAM: {cam.upper()}\n"
        f"│ LAST DISPATCH TRANSCRIPT: {transcript!r}\n"
        f"│ LISTENING: {listening_line}\n"
        f"│ POS (m): X={p[0]:+5.2f} Y={p[1]:+5.2f} Z={p[2]:+5.2f}  │ ERR: pos={cur_pos_err:.4f}m  att={cur_att_err:5.2f}°  │ TOWER: d={dist_tower:+5.2f}m\n"
        f"│ QUAT   : W={q[0]:+5.2f} X={q[1]:+5.2f} Y={q[2]:+5.2f} Z={q[3]:+5.2f}  │ LAT: cls={lat.get('classify', 0):.2f}ms traj={lat.get('trajectory', 0):.2f}ms\n"
        f"│ HEARD LOG (most recent last):\n"
        f"{log_block}"
    )

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    if "agg" not in matplotlib.get_backend().lower():
        plt.pause(0.001)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["text", "vosk"], default="text")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--device", type=int, default=None,
                    help="Microphone device index from sounddevice")
    ap.add_argument("--duration", type=float, default=None,
                    help="Run for N seconds then exit cleanly (useful for automated validation)")
    ap.add_argument("--scripted", type=str, default=None,
                    help="Comma-separated scripted commands to dispatch in text mode")
    args = ap.parse_args()

    if args.source == "vosk":
        src = VoskSource(model_dir=args.model_dir, device=args.device)
        src.start()
    else:
        if args.scripted:
            cmd_list = [c.strip() for c in args.scripted.split(",") if c.strip()]
            src = TextSource(scripted=cmd_list)
        else:
            src = TextSource(interactive=True)

    fc = FlightController()
    last_cmd = "hover"
    last_meta = {}
    hist = TelemetryHistory()
    # Bounded scrolling log of every heard utterance, recognized or not -
    # the "log" half of "show me graphs AND logs in real time". Separate
    # from last_cmd/last_meta (which still mean "last successful dispatch
    # only", unchanged) so a misheard word shows up here without disturbing
    # that existing state.
    heard_log: deque[str] = deque(maxlen=8)
    get_partial = getattr(src, "get_partial", None)  # None for TextSource
    # Logs every dispatched command to a real parquet file so the "after"
    # plots (plot_2d_telemetry.py / plot_3d_trajectory.py) can show THIS
    # flight instead of only ever an old headless test run - previously
    # this interactive session logged nothing at all.
    flight_logger = LiveFlightLogger(SESSIONS_DIR)

    try:
        import mujoco.viewer as mjviewer
    except Exception as e:  # pragma: no cover
        print("MuJoCo viewer unavailable:", e)
        return

    fig, axes, lines, status_text = build_dashboard()
    t_start = time.perf_counter()

    import mujoco as _mj
    # key_callback must be passed to launch_passive() BEFORE the viewer
    # handle exists, so the B-toggle can't reach into `viewer` directly -
    # it just flips a flag here, and the main loop (which does have the
    # viewer handle) applies it every iteration. GLFW key codes for
    # letters match uppercase ASCII, so ord("B") == GLFW's B keycode.
    view_state = {"cam": "chase"}

    def _key_callback(keycode):
        if keycode == ord("B"):
            view_state["cam"] = "birdseye" if view_state["cam"] == "chase" else "chase"
            print(f"[view] switched to {view_state['cam']} camera", flush=True)

    with mjviewer.launch_passive(fc.model, fc.data, key_callback=_key_callback) as viewer:
        chase_id = _mj.mj_name2id(fc.model, _mj.mjtObj.mjOBJ_CAMERA, "chase")
        bird_id = _mj.mj_name2id(fc.model, _mj.mjtObj.mjOBJ_CAMERA, "birdseye")

        def _apply_camera():
            cid = chase_id if view_state["cam"] == "chase" else bird_id
            if cid >= 0:
                viewer.cam.type = _mj.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = cid

        _apply_camera()
        input_desc = ("speak" if args.source == "vosk" else "type")
        print(f"Live demo running. {input_desc.capitalize()} a command OR click a "
              f"button in the dashboard window - BOTH work at the same time, every "
              f"single control step, regardless of --source "
              f"({', '.join(['forward','back','left','right','up','down','hover','stop'])}). "
              f"Camera: press B (only works if the MuJoCo VIEWER window has "
              f"keyboard focus - click its titlebar first) OR click "
              f"'TOGGLE CAMERA' in the dashboard window (works regardless of "
              f"focus). Ctrl-C to quit.", flush=True)
        t_sim = 0.0
        t_start = time.perf_counter()
        next_redraw = time.perf_counter()
        while viewer.is_running():
            if args.duration is not None and (time.perf_counter() - t_start) >= args.duration:
                print(f"Target test duration of {args.duration}s reached. Exiting cleanly.", flush=True)
                viewer.close()
                break

            if fig._pending_view_toggle["toggle"]:
                fig._pending_view_toggle["toggle"] = False
                view_state["cam"] = "birdseye" if view_state["cam"] == "chase" else "chase"
                print(f"[view] switched to {view_state['cam']} camera (button)", flush=True)

            cmd, meta = src.get_command(timeout=0.0)
            if cmd is None and fig._pending_command["cmd"] is not None:
                # Button click - same dispatch path as text/voice, just a
                # third source of `cmd`. meta mirrors what TextSource
                # produces (transcript field) so the HUD readout and the
                # printed dispatch line look consistent across all three
                # input methods rather than needing special-casing.
                cmd = fig._pending_command["cmd"]
                fig._pending_command["cmd"] = None
                meta = {"transcript": f"[button] {cmd}", "asr_confidence": 1.0,
                         "latencies_ms": {"vad": 0.0, "asr": 0.0, "classify": 0.0}}
            # THE FIX: previously this whole block only ran `if cmd is not
            # None`, so a heard-but-unclassified transcript (mic works,
            # classify() just didn't recognize the word) was thrown away
            # with zero console/HUD output - indistinguishable from "mic
            # never heard anything at all". `meta` always carries the real
            # transcript (voice_input._BaseSource._emit), so show it either
            # way.
            if cmd is None and meta and meta.get("transcript"):
                transcript = meta.get("transcript", "")
                print(f"[heard] {format_heard_entry(transcript, None)}", flush=True)
                heard_log.append(format_heard_entry(transcript, None))
            if cmd is not None:
                last_cmd = cmd
                last_meta = meta
                p_before = fc.position()
                att_err_before = fc.attitude_error_deg(IDENTITY_Q)
                lat = fc.dispatch(cmd)
                if meta:
                    meta["latencies_ms"] = {**meta.get("latencies_ms", {}), "trajectory": lat["trajectory"]}
                transcript = meta.get("transcript", "") if meta else ""
                print(f"[dispatch] {cmd}  (transcript: {transcript!r})", flush=True)
                heard_log.append(format_heard_entry(transcript, cmd))
                classify_ms = (meta.get("latencies_ms", {}).get("classify", 0.0) if meta else 0.0)
                flight_logger.on_dispatch(cmd, transcript, p_before, lat["target"],
                                           classify_ms, lat["trajectory"], att_err_before)

            step_ms = fc.step()
            flight_logger.on_step(step_ms)
            _apply_camera()
            viewer.sync()

            att_err = fc.attitude_error_deg(IDENTITY_Q)
            pos_err = float(np.linalg.norm(fc.position() - fc.traj.position(fc.t_traj)))
            control_effort = float(np.sum(np.abs(fc.data.ctrl)))
            hist.push(t_sim, att_err, pos_err, control_effort)
            t_sim += DT

            if time.perf_counter() >= next_redraw:
                listening = get_partial() if get_partial else ""
                update_dashboard(fig, axes, lines, status_text, hist, fc, last_cmd, last_meta,
                                  listening, heard_log, view_state["cam"])
                next_redraw = time.perf_counter() + DASHBOARD_REDRAW_S
            time.sleep(DT)

        flight_logger.on_exit(fc.position(), fc.attitude_error_deg(IDENTITY_Q))
        saved_path = flight_logger.save()
        if saved_path:
            print(f"[log] saved this flight -> {saved_path.name}  "
                  f"(run plot_2d_telemetry.py / plot_3d_trajectory.py now to see it)",
                  flush=True)

    import matplotlib.pyplot as plt
    plt.close("all")


if __name__ == "__main__":
    main()
