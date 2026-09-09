"""Single source of truth for the city environment drawn in scene.xml.

Everything here is VISUALIZATION ONLY: every object is generated as a
non-collidable (contype=0 conaffinity=0), zero-mass geom, exactly like the
reference boxes it replaces. Headless Stage A-D code loads quad.xml
directly and never touches this file or scene.xml, so changes here are
physics-inert by construction; the physics-identical guarantee is still
re-verified at the end of any change by the full test suite plus a
bit-for-bit trajectory comparison (see docs/CLAUDE.md precedent).

This one spec drives TWO artifacts so they can never drift apart:
  1. make_city.py -> city.xml  (the 3D geoms, <include>d by scene.xml)
  2. make_city.py -> map_texture.png (the printed top-down map on the floor,
     whose footprints/parks/pad are drawn from this same list)

plot_3d_trajectory.py also imports this file (instead of a hand-duplicated
constant) so the matplotlib "city bar" markers track the MuJoCo city.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Flight-envelope safety -------------------------------------------------
# Voice waypoints stay inside +-1.5 m lateral / ~+-0.5 m vertical (see
# commands.py). The obstacle tower's face sits at OBSTACLE_FACE_X, which a
# full un-preempted "forward" (waypoint +1.5 m) still stops ~0.8 m short of,
# and which the command-preemption demonstration (run_preemption.py) keeps
# the drone well away from by reversing mid-transit.
OBSTACLE_X = 3.0      # center of the front obstacle tower (world +X, "forward")
OBSTACLE_HALF = 0.7   # half footprint (square), so the face is at:
OBSTACLE_FACE_X = OBSTACLE_X - OBSTACLE_HALF  # 2.3 m - past every waypoint
# Radius of the dashed "warning zone" drawn around the tower on the map.
# A full un-preempted forward run (1.0 m/s planned, ~1.7 s transit) has its
# reversal decision point inside this ring; the preemption demo guarantees
# the drone never reaches it.
OBSTACLE_RING_R = 2.6


@dataclass(frozen=True)
class CityObject:
    """One world-frame object in the city.

    ``half_x``/``half_y`` are half-footprint extents (like MuJoCo box
    ``size``). ``height`` is the FULL height in meters; flat map-only marks
    (parks, home pad) carry a small height so they still read in 3D without
    ever touching the drone's flight altitude (~1.0 m).
    """

    name: str
    x: float
    y: float
    half_x: float
    half_y: float
    height: float
    rgba: tuple[float, float, float, float]
    kind: str = "building"  # building | tower | park | tree | pad


# Urban palette: muted slate/granite/terracotta, a single distinct
# orange-red tower for the forward obstacle, greens for parks/trees, cyan
# for the home pad - all cosmetic.
_SLATE = (0.42, 0.44, 0.52, 1.0)
_SLATE2 = (0.38, 0.40, 0.48, 1.0)
_SLATE3 = (0.45, 0.47, 0.55, 1.0)
_SLATE4 = (0.35, 0.37, 0.44, 1.0)
_TERRA = (0.58, 0.42, 0.36, 1.0)
_GRANITE = (0.50, 0.50, 0.54, 1.0)
_TOWER = (0.85, 0.40, 0.25, 1.0)
_PARK = (0.34, 0.56, 0.34, 1.0)
_TREE = (0.25, 0.50, 0.30, 1.0)
_PAD = (0.20, 0.80, 0.95, 1.0)

CITY_OBJECTS: tuple[CityObject, ...] = (
    # --- Forward obstacle tower (home of the preemption demo) --------------
    CityObject("obstacle_tower", OBSTACLE_X, 0.0, OBSTACLE_HALF, OBSTACLE_HALF,
               3.5, _TOWER, "tower"),
    # --- Mid ring (~4-5 m out) gives the near-field depth without ever
    #     intruding on the +-1.5 m command envelope -------------------------
    CityObject("block_n", 0.0, 4.6, 0.90, 0.55, 2.8, _SLATE),
    CityObject("block_s", 0.0, -4.6, 0.90, 0.55, 2.8, _SLATE2),
    CityObject("block_ne", 4.4, 3.4, 0.55, 0.55, 2.2, _TERRA),
    CityObject("block_se", 4.4, -3.4, 0.55, 0.55, 2.2, _GRANITE),
    CityObject("block_nw", -4.6, 3.2, 0.60, 0.50, 2.6, _SLATE3),
    CityObject("block_sw", -4.6, -3.2, 0.60, 0.50, 2.6, _SLATE4),
    # --- Far edge clusters at x=+-8 (mid-east/west) -------------------------
    CityObject("edge_e1", 7.5, 1.4, 0.60, 0.55, 3.0, _SLATE),
    CityObject("edge_e2", 8.7, -1.5, 0.45, 0.70, 2.6, _GRANITE),
    CityObject("edge_e3", 7.6, -3.4, 0.50, 0.50, 2.4, _TERRA),
    CityObject("edge_w1", -7.6, 1.5, 0.55, 0.55, 2.9, _SLATE2),
    CityObject("edge_w2", -8.7, -1.2, 0.50, 0.60, 3.4, _SLATE4),
    CityObject("edge_w3", -7.5, -3.5, 0.50, 0.50, 2.5, _SLATE),
    # --- Corner districts at (+-8, +-8) -------------------------------------
    CityObject("corner_ne1", 7.4, 8.6, 0.90, 0.50, 3.4, _SLATE),
    CityObject("corner_ne2", 8.8, 7.3, 0.55, 0.95, 4.2, _TERRA),
    CityObject("corner_ne3", 8.5, 8.9, 0.40, 0.40, 2.4, _GRANITE),
    CityObject("corner_nw1", -7.3, 8.7, 0.95, 0.55, 4.6, _SLATE3),
    CityObject("corner_nw2", -8.7, 7.4, 0.55, 0.95, 3.0, _SLATE2),
    CityObject("corner_nw3", -8.3, 9.0, 0.40, 0.40, 2.2, _TERRA),
    CityObject("corner_se1", 8.7, -7.4, 0.60, 0.90, 2.8, _GRANITE),
    CityObject("corner_se2", 7.4, -8.7, 0.90, 0.55, 3.8, _SLATE),
    CityObject("corner_se3", 8.9, -8.9, 0.35, 0.35, 2.0, _TERRA),
    CityObject("corner_sw1", -7.4, -8.7, 0.85, 0.55, 3.5, _SLATE4),
    CityObject("corner_sw2", -8.8, -7.5, 0.55, 0.85, 2.7, _SLATE),
    CityObject("corner_sw3", -7.7, -9.0, 0.55, 0.40, 3.2, _SLATE2),
    # --- Parks (flat green) + trees on the north/south edge ----------------
    CityObject("park_n", 0.0, 8.0, 1.60, 0.90, 0.10, _PARK, "park"),
    CityObject("park_s", 0.0, -8.0, 1.30, 1.00, 0.10, _PARK, "park"),
    CityObject("tree_n1", 1.3, 8.8, 0.25, 0.25, 1.0, _TREE, "tree"),
    CityObject("tree_n2", -1.3, 8.6, 0.25, 0.25, 0.9, _TREE, "tree"),
    CityObject("tree_n3", 0.2, 9.0, 0.22, 0.22, 1.1, _TREE, "tree"),
    CityObject("tree_s1", 1.4, -8.6, 0.25, 0.25, 1.0, _TREE, "tree"),
    CityObject("tree_s2", -1.4, -8.7, 0.22, 0.22, 0.9, _TREE, "tree"),
    CityObject("tree_s3", 0.1, -9.0, 0.24, 0.24, 1.0, _TREE, "tree"),
    # --- Home pad at the plaza center (flat, under the drone's spawn) ------
    CityObject("home_pad", 0.0, 0.0, 0.95, 0.95, 0.02, _PAD, "pad"),
)


def buildings() -> tuple[CityObject, ...]:
    """City objects that stand up as 3D landmarks (everything except the
    flat park marks and the home pad)."""
    return tuple(o for o in CITY_OBJECTS if o.kind in ("building", "tower"))


def obstacle() -> CityObject:
    return CITY_OBJECTS[0]


# --- Chained-command safety clamp -------------------------------------------
# `test_city_buildings_clear_flight_corridor` only ever proved a SINGLE
# voice command from the origin can't land inside a building. It says
# nothing about two commands in the SAME direction (e.g. "go straight"
# said twice) compounding past one - which is exactly what happened live:
# forward + forward pushed the target to x=3.0 m, dead center of the
# obstacle tower (face at OBSTACLE_FACE_X=2.3 m). Buildings are
# non-collidable (contype=0/conaffinity=0, see module docstring), so
# nothing in MuJoCo's physics would have stopped that target from being
# accepted - the drone just flew its cascade straight at/through the
# tower's visual mesh. This margin is the general fix: clamp at dispatch
# time (flight.py::FlightController.dispatch), not by making the scene
# collidable (which would be a physics/controller change, out of scope).
SAFETY_MARGIN_M = 0.3


def clamp_target_to_safe_zone(p0, p1, margin: float = SAFETY_MARGIN_M):
    """Given a commanded flight from `p0` to `p1`, return the furthest
    point along that straight segment that stays `margin` meters clear of
    every building/tower - or `p1` unchanged if the direct path never
    enters one. General ray-vs-AABB clipping (slab method), not just an
    axis-aligned special case, so it stays correct even if a future
    command ever moves on more than one axis at once."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    d = p1 - p0
    seg_len = float(np.linalg.norm(d))
    if seg_len < 1e-9:
        return p1.copy()  # hover/stop: no travel, nothing to clamp

    best_t = 1.0  # fraction of the p0->p1 segment that's actually clear
    for b in buildings():
        lo = np.array([b.x - b.half_x, b.y - b.half_y, 0.0])
        hi = np.array([b.x + b.half_x, b.y + b.half_y, b.height])
        t_enter, t_exit, hit = 0.0, 1.0, True
        for axis in range(3):
            if abs(d[axis]) < 1e-9:
                if p0[axis] < lo[axis] or p0[axis] > hi[axis]:
                    hit = False
                    break
                continue
            t1 = (lo[axis] - p0[axis]) / d[axis]
            t2 = (hi[axis] - p0[axis]) / d[axis]
            t1, t2 = min(t1, t2), max(t1, t2)
            t_enter, t_exit = max(t_enter, t1), min(t_exit, t2)
            if t_enter > t_exit:
                hit = False
                break
        if not hit or t_enter > 1.0 or t_exit < 0.0:
            continue  # segment never reaches this building's box
        safe_t = max(0.0, t_enter - margin / seg_len)
        best_t = min(best_t, safe_t)

    if best_t >= 1.0 - 1e-9:
        return p1.copy()
    return p0 + best_t * d