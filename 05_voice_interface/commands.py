"""Stage E command vocabulary, world-frame waypoint map, and rule-based
classifier (docs/ENGINEERING_PLAN.md Stage E task 2).

Eight fixed commands, world-frame directions (docs/TDD.md section 7b):
forward=+X, back=-X, left=+Y, right=-Y, up=+Z, down=-Z, hover/stop = hold
position. Yaw is never commanded, so directions are world-fixed regardless
of current attitude. up/down were added for the Mission Control UI
(docs/CLAUDE.md Stage E "Mission Control" entry) - PositionController and
PointToPointTrajectory already operate on arbitrary R^3 offsets, so this
is a pure vocabulary/waypoint-map addition, not a controller change.
"""
from __future__ import annotations

import difflib

import numpy as np

COMMANDS = ("forward", "back", "left", "right", "up", "down", "hover", "stop")
WAYPOINT_OFFSET_M = 1.5
# Vertical offset is deliberately smaller than the lateral one: hover
# altitude is ~1.0m (quad.xml body pos), and the visual floor in scene.xml
# is non-collidable (contype=0 conaffinity=0 - visualization only, see
# that file's own comment), so nothing physically stops "down" from
# flying the drone below z=0 if the offset is too large. 0.5m keeps
# "down" landing at ~0.5m - comfortably above the floor - while still
# being a visually obvious vertical move.
VERTICAL_OFFSET_M = 0.5

_COMMAND_OFFSET = {
    "forward": np.array([WAYPOINT_OFFSET_M, 0.0, 0.0]),
    "back": np.array([-WAYPOINT_OFFSET_M, 0.0, 0.0]),
    "left": np.array([0.0, WAYPOINT_OFFSET_M, 0.0]),
    "right": np.array([0.0, -WAYPOINT_OFFSET_M, 0.0]),
    "up": np.array([0.0, 0.0, VERTICAL_OFFSET_M]),
    "down": np.array([0.0, 0.0, -VERTICAL_OFFSET_M]),
    "hover": np.array([0.0, 0.0, 0.0]),
    "stop": np.array([0.0, 0.0, 0.0]),
}


def target_offset(command: str) -> np.ndarray:
    """World-frame position offset applied from the current position when
    `command` is dispatched."""
    return _COMMAND_OFFSET[command].copy()


def classify(transcript: str) -> str | None:
    """Map a raw ASR transcript to one of the fixed commands, or None if no
    command is confidently present. Rule-based: exact/close word match via
    difflib (cutoff 0.6), whole-transcript first, then the GLOBALLY
    best-scoring token in the transcript (not the first token to clear the
    cutoff).

    The "first token above cutoff wins, left-to-right" version of this
    function had a real bug: in "five meters to left", the short filler
    word "to" fuzzy-matches "stop" at ratio 0.667 (>= the 0.6 cutoff,
    purely from the shared 't'/'o' characters) and gets returned before
    the loop ever reaches "left", which would have scored a perfect 1.0.
    Scoring every token and keeping the best match fixes this class of bug
    generally, not just for these two phrases: a short, coincidentally-
    fuzzy-matching filler word can no longer pre-empt a later exact match.
    """
    if not transcript:
        return None
    text = transcript.lower().strip()
    if not text:
        return None
    whole = difflib.get_close_matches(text, COMMANDS, n=1, cutoff=0.6)
    if whole:
        return whole[0]

    best_cmd, best_ratio = None, 0.0
    for tok in text.split():
        hits = difflib.get_close_matches(tok, COMMANDS, n=1, cutoff=0.6)
        if not hits:
            continue
        ratio = difflib.SequenceMatcher(None, tok, hits[0]).ratio()
        if ratio > best_ratio:
            best_ratio, best_cmd = ratio, hits[0]
    return best_cmd
