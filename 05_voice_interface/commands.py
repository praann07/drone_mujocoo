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


def command_vocabulary() -> dict[str, list[str]]:
    """Every fixed command mapped to its recognized synonyms (may be an
    empty list). Built by inverting `_SYNONYMS` so a printed guide (see
    print_commands_guide.py) can never drift out of sync with the real
    vocabulary - there is exactly one place synonyms are defined."""
    out: dict[str, list[str]] = {cmd: [] for cmd in COMMANDS}
    for word, cmd in _SYNONYMS.items():
        out[cmd].append(word)
    return out


# Natural synonyms that mean a command word but share no useful letter
# overlap with it (so fuzzy matching alone can't find them, and sometimes
# actively mismatches them - see the "straight" bug below). Checked as
# exact token matches before any fuzzy scoring.
_SYNONYMS = {
    "straight": "forward", "ahead": "forward", "forwards": "forward",
    "backward": "back", "backwards": "back", "reverse": "back",
    "halt": "stop", "freeze": "stop", "wait": "stop", "pause": "stop",
    "ascend": "up", "climb": "up", "rise": "up", "higher": "up", "lift": "up",
    "descend": "down", "lower": "down", "land": "down", "drop": "down",
    "stay": "hover", "hold": "hover",
}

# Below this length, a token is too short for difflib ratio to be a
# trustworthy signal - short common words coincidentally share enough
# letters with a real command to clear even a fairly high cutoff. Real
# commands this short ("up") are still matched at ratio 1.0 (exact) or
# near it, which clears this stricter bar fine; it only blocks WEAK
# coincidental matches from short filler words.
# 0.7 (not the original 0.6): measured on real live-mic transcripts,
# genuine matches (typos/short forms of the actual command word, e.g.
# "foward"->forward 0.923, "lef"->left 0.857) score far above coincidental
# collisions from unrelated English words (e.g. "laptop"->stop, "their"->
# hover, both exactly 0.6) - there's a clean gap between the two
# populations, and 0.7 sits safely in it.
_SHORT_TOKEN_LEN = 4
_SHORT_TOKEN_CUTOFF = 0.8
_CUTOFF = 0.7


def classify(transcript: str) -> str | None:
    """Map a raw ASR transcript to one of the fixed commands, or None if no
    command is confidently present. Rule-based: known synonyms first
    (exact word match, see _SYNONYMS), then exact/close word match via
    difflib, whole-transcript first, then the GLOBALLY best-scoring token
    in the transcript (not the first token to clear the cutoff).

    Two real bugs found from actual live-mic transcripts (not hypothetical):
    1. "first token above cutoff wins, left-to-right" (an earlier version of
       this function): in "five meters to left", the short filler word "to"
       fuzzy-matches "stop" at ratio 0.667 and got returned before the loop
       ever reached "left" (a perfect 1.0 match). Fixed by scoring every
       token and keeping the best match, not the first.
    2. Scoring every token was NOT enough on its own: in "go straight" (a
       completely natural way to say "forward"), "straight" fuzzy-matches
       "right" at a HIGH ratio purely from sharing the letters r-i-g-h-t as
       a near-contiguous run ("stRAIGHT" / "RIGHT"), and nothing in the
       sentence scores higher - so "straight" was still the best-scoring
       token, just for the WRONG command. No amount of "pick the best
       match" fixes a coincidental match that IS the best-scoring one -
       this needs an explicit synonym table (word-level, checked before
       fuzzy matching) plus a stricter cutoff for short tokens, since most
       of these coincidental collisions involve a short (<=3 char) token
       riding a lenient 0.6 cutoff against a longer command word.
    """
    if not transcript:
        return None
    text = transcript.lower().strip()
    if not text:
        return None

    tokens = text.split()
    for tok in tokens:
        if tok in _SYNONYMS:
            return _SYNONYMS[tok]

    whole = difflib.get_close_matches(text, COMMANDS, n=1, cutoff=_CUTOFF)
    if whole:
        return whole[0]

    best_cmd, best_ratio = None, 0.0
    for tok in tokens:
        cutoff = _SHORT_TOKEN_CUTOFF if len(tok) < _SHORT_TOKEN_LEN else _CUTOFF
        hits = difflib.get_close_matches(tok, COMMANDS, n=1, cutoff=cutoff)
        if not hits:
            continue
        ratio = difflib.SequenceMatcher(None, tok, hits[0]).ratio()
        if ratio > best_ratio:
            best_ratio, best_cmd = ratio, hits[0]
    return best_cmd
