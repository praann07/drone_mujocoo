"""Stage E offscreen 3D navigation demo (docs/ENGINEERING_PLAN.md Stage E
task 4). Renders the MuJoCo plant flying through a scripted sequence of
voice commands, driving it through the Stage D cascade, and writes an
animated GIF + telemetry overlay. This produces the "live 3D navigation"
artifact without requiring a display or microphone, so the visual demo is
verifiable in a headless environment; the interactive viewer lives in
demo.py.

    .venv\\Scripts\\python.exe 05_voice_interface\\render_offscreen.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
for _p in ("05_voice_interface", "01_simulation/models"):
    sys.path.insert(0, str(ROOT / _p))
from flight import DT, FlightController, IDENTITY_Q  # noqa: E402
from city import OBSTACLE_FACE_X  # noqa: E402
from run_preemption import PREEMPT_AT_S  # noqa: E402

WIDTH, HEIGHT = 640, 480
FPS = 25
STEPS_PER_FRAME = max(1, int(round(1.0 / (FPS * DT))))
COMMAND_SEQUENCE = ["forward", "left", "back", "right", "hover"]


def _font():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _overlay(rgb: np.ndarray, fc: FlightController, command: str,
             frame_idx: int) -> Image.Image:
    img = Image.fromarray(rgb).convert("RGB")
    p = fc.position()
    q = fc.attitude()
    att_err = fc.attitude_error_deg(IDENTITY_Q)
    dist_tower = OBSTACLE_FACE_X - p[0]
    txt = [
        f"cmd: {command}",
        f"pos  x={p[0]:+6.2f} y={p[1]:+6.2f} z={p[2]:+6.2f}",
        f"quat w={q[0]:+.2f} x={q[1]:+.2f} y={q[2]:+.2f} z={q[3]:+.2f}",
        f"att_err={att_err:6.2f} deg",
        f"tower_dist x={dist_tower:+6.2f} m",
    ]
    draw = ImageDraw.Draw(img)
    font = _font()
    y = 8
    for line in txt:
        draw.text((8, y), line, fill=(0, 255, 0), font=font)
        y += 16
    return img


def render(out_dir: Path | None = None, commands=None,
           flight_s: float = 5.0, preempt: bool = False) -> Path:
    out_dir = out_dir or (Path(__file__).resolve().parent.parent / "data"
                          / "processed" / "stage_e_demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    fc = FlightController()
    renderer = __import__("mujoco").Renderer(fc.model, width=WIDTH, height=HEIGHT)
    frames = []
    step = 0

    def _emit(disp_cmd: str, fc: FlightController, preempted_flag: str):
        nonlocal step
        fc.step()
        step += 1
        if step % STEPS_PER_FRAME == 0:
            renderer.update_scene(fc.data, camera="chase")
            rgb = renderer.render()
            frames.append(_overlay(rgb, fc, f"{disp_cmd}{preempted_flag}", step))

    if not preempt:
        seq = commands or COMMAND_SEQUENCE
        for cmd in seq:
            fc.dispatch(cmd)
            n_steps = int(round(flight_s / DT))
            for _ in range(n_steps):
                _emit(cmd, fc, "")
            print(f"  rendered command '{cmd}' -> pos {fc.position()}")
        gif_path = out_dir / "stage_e_navigation.gif"
    else:
        # Mid-flight preemption sequence: forward then back before the
        # waypoint, mirroring run_preemption.py (same PREEMPT_AT_S) so the
        # GIF, the logged parquet, and the robustness PNG tell one story.
        fc.dispatch("forward")
        preempted = False
        n_steps = int(round(flight_s / DT))
        for k in range(n_steps):
            t = k * DT
            if not preempted and t >= PREEMPT_AT_S:
                fc.dispatch("back")
                preempted = True
            _emit("forward", fc, "  -> BACK!" if preempted else "")
        print(f"  rendered preemption flight -> pos {fc.position()}"
              f" (preempted at t={PREEMPT_AT_S:.2f}s)")
        gif_path = out_dir / "robustness_preemption.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0)
    print(f"wrote {len(frames)} frames -> {gif_path}")
    return gif_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--preempt", action="store_true",
                    help="render the mid-flight preemption GIF "
                         "(robustness_preemption.gif) instead of the "
                         "standard navigation GIF")
    args = ap.parse_args()
    render(preempt=args.preempt)
