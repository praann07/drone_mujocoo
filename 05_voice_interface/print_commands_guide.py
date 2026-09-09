"""Prints a plain-English reference for the live demo: every voice command
(with its recognized synonyms), the dashboard buttons, and how the live
feedback works. Meant to be launched into its own window that stays open
for the whole demo (see RUN_EVERYTHING.bat) - this script just prints once
and exits; the launcher's `cmd /k` is what keeps the window up.

Vocabulary is imported from commands.py, never re-typed here, so this
guide can't silently drift out of sync with what the classifier actually
accepts.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from commands import COMMANDS, command_vocabulary  # noqa: E402


def main() -> None:
    vocab = command_vocabulary()
    width = 70
    print("=" * width)
    print("  VOICE-CONTROLLED QUADROTOR - COMMANDS GUIDE".center(width))
    print("=" * width)
    print()
    print("Say (or type) any of these words - exact word, a typo of it, or")
    print("one of its synonyms all work:")
    print()
    for cmd in COMMANDS:
        syns = vocab[cmd]
        line = f"  {cmd.upper():<8}"
        if syns:
            line += " - also: " + ", ".join(syns)
        print(line)
    print()
    print("-" * width)
    print("DASHBOARD WINDOW")
    print("-" * width)
    print("  - 8 click buttons (one per command above) - work even if your")
    print("    voice isn't recognized, or you're not using a mic at all.")
    print("  - TOGGLE CAMERA button - switches chase <-> bird's-eye view.")
    print("    Works by click regardless of which window has focus (the")
    print("    'B' keyboard shortcut only works if the 3D VIEWER window")
    print("    itself is focused).")
    print("  - LISTENING line - shows words appearing live as you speak.")
    print("  - HEARD LOG - every word you say shows up here, whether it")
    print("    was recognized or not - nothing is ever silently dropped.")
    print("  - 3 strip charts - attitude error, position error, control")
    print("    effort, updating live every control step.")
    print("-" * width)
    print()
    print("-" * width)
    print("AFTER YOU FLY")
    print("-" * width)
    print("  This window is just a reference - it stays open. When you're")
    print("  done flying, double-click SHOW_RESULTS.bat for a fresh graph")
    print("  of exactly what you just did (not an old test run - every")
    print("  command you fly gets logged automatically).")


if __name__ == "__main__":
    main()
