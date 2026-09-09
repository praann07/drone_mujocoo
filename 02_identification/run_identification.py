"""Stage B entry point: preprocess, fit SINDy + DMDc, evaluate
one-step-ahead prediction error on genuinely held-out trials.

    .venv\\Scripts\\python.exe 02_identification\\run_identification.py

Stage B's gate is one-step-ahead error only. No open-loop rollout is
produced here - that is Stage C, and per the project rules it may not
even be attempted until this gate passes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dmdc_fit import DMDcModel  # noqa: E402
from preprocessing import CONTROL_COLS, DERIV_COLS, STATE_COLS, preprocess_trial  # noqa: E402
from sindy_fit import SINDyModel  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DT = 0.002  # s, docs/TDD.md section 1

# --- Stage B gate thresholds -----------------------------------------
# Reported error is normalised by the actual per-step state CHANGE, not
# by state magnitude. This matters: at dt=2ms the state barely moves
# between samples, so error normalised by state magnitude is tiny for ANY
# model - including a "predict no change" model that has learned nothing.
# That is precisely the degenerate-model failure this project exists to
# catch, so the metric is chosen to be un-gameable: a do-nothing model
# scores 1.0 on it by construction, and the trivial baseline is reported
# alongside every result as proof the number is meaningful.
GATE_STEP_NRMSE = 0.10     # predicted step must capture >=90% of true motion
GATE_DERIV_R2 = 0.99       # derivative fit quality


def load_split() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "trial_split.parquet")


def load_processed(trial_ids) -> dict[str, pd.DataFrame]:
    out = {}
    for tid in trial_ids:
        df = preprocess_trial(pd.read_parquet(RAW_DIR / f"{tid}.parquet"))
        if not df.empty:
            out[tid] = df
    return out


def stack(frames: dict[str, pd.DataFrame]):
    df = pd.concat(frames.values(), ignore_index=True)
    return (df[STATE_COLS].to_numpy(), df[CONTROL_COLS].to_numpy(), df[DERIV_COLS].to_numpy())


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean(axis=0)) ** 2)
    return float(1.0 - ss_res / ss_tot)


def consecutive_pairs(df: pd.DataFrame):
    """Yield (state_k, control_k, state_k+1) within each segment.

    Pairs are formed per segment_id, never across a reset boundary - see
    the segment_id note in preprocessing.preprocess_trial.
    """
    for _, seg in df.groupby("segment_id", sort=True):
        if len(seg) < 2:
            continue
        state = seg[STATE_COLS].to_numpy()
        control = seg[CONTROL_COLS].to_numpy()
        yield state[:-1], control[:-1], state[1:]


def evaluate_one_step(predict_next, frames: dict[str, pd.DataFrame]) -> dict:
    """One-step-ahead error, normalised by the true per-step state change.

    predict_next(state, control) -> predicted next state.
    """
    err_sq, change_sq = 0.0, 0.0
    for df in frames.values():
        for state, control, true_next in consecutive_pairs(df):
            pred = predict_next(state, control)
            err_sq += np.sum((pred - true_next) ** 2)
            change_sq += np.sum((true_next - state) ** 2)
    return {
        "step_nrmse": float(np.sqrt(err_sq / change_sq)),
        # Trivial reference: predict "nothing changes". Scores exactly 1.0.
        "baseline_step_nrmse": 1.0,
    }


def fit_models(verbose: bool = True) -> dict:
    """Fit SINDy and DMDc exactly as the Stage B gate did, and return
    everything Stage C needs (fitted models + all data splits) so Stage C
    validates the SAME fit rather than risking a silent re-derivation
    drift from re-running this logic independently."""
    p = print if verbose else (lambda *a, **k: None)
    split = load_split()
    p("Preprocessing (Savitzky-Golay smoothing + rotation-vector transform)...")

    full_fit_ids = split[(split.regime == "full_envelope") & (split.split == "fit")].trial_id
    full_held_ids = split[(split.regime == "full_envelope") & (split.split == "held_out")].trial_id
    hover_fit_ids = split[(split.regime == "near_hover") & (split.split == "fit")].trial_id
    hover_held_ids = split[(split.regime == "near_hover") & (split.split == "held_out")].trial_id

    full_fit = load_processed(full_fit_ids)
    full_held = load_processed(full_held_ids)
    hover_fit = load_processed(hover_fit_ids)
    hover_held = load_processed(hover_held_ids)

    overlap = set(full_fit) & set(full_held) | set(hover_fit) & set(hover_held)
    assert not overlap, f"fit/held-out leakage: {overlap}"
    p(f"  full-envelope: {len(full_fit)} fit / {len(full_held)} held-out trials")
    p(f"  near-hover:    {len(hover_fit)} fit / {len(hover_held)} held-out trials")

    # --- SINDy, full envelope ----------------------------------------
    # The STLSQ sparsity threshold is selected on a validation split
    # carved out of the FIT trials, never on held-out data. Choosing it by
    # held-out score would be selecting a hyperparameter on the same data
    # the gate is then judged against, which quietly turns the gate into
    # a training metric. The held-out set is touched exactly once, below.
    # (Stage C repeats this as a full documented ablation.)
    p("\nFitting SINDy (physically-motivated library, STLSQ)...")
    fit_ids = sorted(full_fit)
    val_ids = fit_ids[::3] or fit_ids[:1]          # ~1/3 of fit trials, trial-level
    core_ids = [i for i in fit_ids if i not in set(val_ids)]
    Xc, Uc, Dc = stack({i: full_fit[i] for i in core_ids})
    Xv, Uv, Dv = stack({i: full_fit[i] for i in val_ids})

    candidates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    scored = []
    for thr in candidates:
        cand = SINDyModel(threshold=thr).fit(Xc, Uc, Dc)
        scored.append((thr, r2(Dv, cand.predict_derivative(Xv, Uv)), cand.n_active()))

    # Parsimony rule: take the SPARSEST model whose validation R^2 is
    # within R2_TOLERANCE of the best. Maximising R^2 alone would always
    # pick the densest candidate - here it prefers 127 terms over 44 to
    # buy 0.0002 of R^2, which is noise, not signal, and defeats the
    # entire point of a *sparse* identification. Trading a negligible fit
    # difference for a 3x simpler model is the whole reason to run SINDy
    # rather than plain least squares.
    R2_TOLERANCE = 0.002
    best_r2 = max(s[1] for s in scored)
    viable = [s for s in scored if s[1] >= best_r2 - R2_TOLERANCE]
    best_thr, sel_r2, sel_active = min(viable, key=lambda s: s[2])
    p(f"  threshold sweep on validation split ({len(val_ids)} of {len(fit_ids)} fit trials):")
    for s_thr, s_r2, s_active in scored:
        mark = " <-- selected (sparsest within tolerance)" if s_thr == best_thr else ""
        p(f"    thr={s_thr:<6} val R^2={s_r2:.6f}  active={s_active}{mark}")

    Xs, Us, Ds = stack(full_fit)
    sindy = SINDyModel(threshold=best_thr).fit(Xs, Us, Ds)
    p(f"  refit on all fit trials: active terms {sindy.n_active()} of {sindy.coef_.size}")

    Xh, Uh, Dh = stack(full_held)
    sindy_deriv_r2 = r2(Dh, sindy.predict_derivative(Xh, Uh))
    sindy_step = evaluate_one_step(lambda s, c: sindy.step(s, c, DT), full_held)
    p(f"  held-out derivative R^2 : {sindy_deriv_r2:.6f}")
    p(f"  held-out step NRMSE     : {sindy_step['step_nrmse']:.4f} (trivial baseline 1.0)")

    # --- DMDc, near hover only ---------------------------------------
    p("\nFitting DMDc (near-hover trim regime only)...")
    dmdc = DMDcModel().fit([(seg[STATE_COLS].to_numpy(), seg[CONTROL_COLS].to_numpy())
                             for df in hover_fit.values()
                             for _, seg in df.groupby("segment_id", sort=True)
                             if len(seg) >= 2])
    p(f"  pydmd vs least-squares operator mismatch: {dmdc.pydmd_mismatch:.3e} "
      f"(cross-checked on a rank-{dmdc.pydmd_check_rank} segment)")
    dmdc_step = evaluate_one_step(dmdc.step, hover_held)
    dmdc_pred, dmdc_true_deriv = [], []
    for df in hover_held.values():
        for state, control, _ in consecutive_pairs(df):
            dmdc_pred.append((dmdc.step(state, control) - state) / DT)
        for _, seg in df.groupby("segment_id", sort=True):
            if len(seg) >= 2:
                dmdc_true_deriv.append(seg[DERIV_COLS].to_numpy()[:-1])
    dmdc_deriv_r2 = r2(np.vstack(dmdc_true_deriv), np.vstack(dmdc_pred))
    p(f"  held-out derivative R^2 : {dmdc_deriv_r2:.6f}")
    p(f"  held-out step NRMSE     : {dmdc_step['step_nrmse']:.4f} (trivial baseline 1.0)")

    # DMDc outside its declared regime - reported, never used as a pass.
    dmdc_out = evaluate_one_step(dmdc.step, full_held)
    p(f"  [outside declared regime] full-envelope step NRMSE: {dmdc_out['step_nrmse']:.4f}")

    results = pd.DataFrame([
        {"model": "sindy", "regime": "full_envelope", "deriv_r2": sindy_deriv_r2,
         "step_nrmse": sindy_step["step_nrmse"], "n_active_terms": sindy.n_active()},
        {"model": "dmdc", "regime": "near_hover", "deriv_r2": dmdc_deriv_r2,
         "step_nrmse": dmdc_step["step_nrmse"], "n_active_terms": int(dmdc.A.size + dmdc.B.size)},
    ])
    results.to_csv(PROCESSED_DIR / "identification_results.csv", index=False)

    passed = bool(((results.step_nrmse < GATE_STEP_NRMSE) & (results.deriv_r2 > GATE_DERIV_R2)).all())

    return {
        "split": split, "sindy": sindy, "dmdc": dmdc,
        "full_fit": full_fit, "full_held": full_held,
        "hover_fit": hover_fit, "hover_held": hover_held,
        "sindy_threshold": best_thr, "threshold_sweep": scored,
        "results": results, "dmdc_out_of_regime_nrmse": dmdc_out["step_nrmse"],
        "gate_passed": passed,
    }


def main() -> int:
    fitted = fit_models(verbose=True)
    results = fitted["results"]

    print("\n" + "=" * 62)
    print("STAGE B GATE")
    print("=" * 62)
    for _, r in results.iterrows():
        ok = r.step_nrmse < GATE_STEP_NRMSE and r.deriv_r2 > GATE_DERIV_R2
        print(f"  {r.model:6s} ({r.regime:13s}): step NRMSE {r.step_nrmse:.4f} "
              f"< {GATE_STEP_NRMSE} and R^2 {r.deriv_r2:.6f} > {GATE_DERIV_R2}  "
              f"-> {'PASS' if ok else 'FAIL'}")
    print(f"\nSTAGE B GATE: {'PASS' if fitted['gate_passed'] else 'FAIL'}")
    if not fitted["gate_passed"]:
        print("Stage C must NOT be started. Fix identification first.")
    return 0 if fitted["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
