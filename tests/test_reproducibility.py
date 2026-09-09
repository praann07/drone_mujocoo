"""Regenerating a PRBS trial in a fresh process must reproduce the stored
log bit-for-bit.

This exists because an earlier version derived seeds from hash(), which
Python salts per process (PYTHONHASHSEED) - so the "seeded" PRBS data
silently differed on every run, which would have made every Stage B/C
number unreproducible and the stored fit/held-out split meaningless.
pytest runs in a separate process from whatever produced data/raw, so
this test genuinely exercises cross-process determinism.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "01_simulation"))

from run_excitation import DURATION_S, seed_for_trial  # noqa: E402
from sim_driver import run_trial  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"


@pytest.mark.parametrize("axis,signal_type,trial_index",
                          [("roll", "prbs", 0), ("combined", "prbs", 3)])
def test_prbs_trial_regenerates_identically(axis, signal_type, trial_index):
    trial_id = f"A_{axis}_{signal_type}_{trial_index:03d}"
    stored_path = RAW_DIR / f"{trial_id}.parquet"
    if not stored_path.exists():
        pytest.skip(f"{trial_id} not generated yet - run run_excitation.py first")

    stored = pd.read_parquet(stored_path)
    regenerated = run_trial(axis, signal_type, trial_id, duration_s=DURATION_S,
                            seed=seed_for_trial(axis, signal_type, trial_index))

    pd.testing.assert_frame_equal(stored, regenerated)
