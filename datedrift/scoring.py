"""Scoring: per-cell accuracy, flip rate vs the no-date baseline, paired bootstrap CIs.

Inputs are results rows (one per model x system_date x question). The parsed answer and
the ground truth are *recomputed* here from the raw response and question parameters, so
a parser fix never requires re-querying the models.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import NONE_REFERENCE_DATE
from .parser import parse
from .questions import truth

BASELINE = "none"
N_BOOT = 1000
BOOT_SEED = 0


def load_rows(path: str | Path) -> pd.DataFrame:
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return enrich(pd.DataFrame(rows))


def _ref(label: str) -> dt.date:
    return NONE_REFERENCE_DATE if label == BASELINE else dt.date.fromisoformat(label)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed = [parse(r["response"], r["question"]["answer_type"]) for _, r in df.iterrows()]
    df["parsed"] = pd.Series(parsed, index=df.index, dtype=object)
    df["truth"] = pd.Series([truth(r["question"], _ref(r["system_date"])) for _, r in df.iterrows()], index=df.index, dtype=object)
    df["unparsed"] = [p is None for p in parsed]
    df["correct"] = [(p is not None and p == t) for p, t in zip(df["parsed"], df["truth"])]
    df["kind"] = [q["kind"] for q in df["question"]]
    return df


def _same(a, b) -> bool:
    """Two parsed answers are 'the same' if equal; unparsed (None) only equals unparsed."""
    a_na = a is None or (isinstance(a, float) and np.isnan(a))
    b_na = b is None or (isinstance(b, float) and np.isnan(b))
    if a_na or b_na:
        return a_na and b_na
    return a == b


def matrices(df: pd.DataFrame, model: str) -> tuple[list[str], list[str], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return (dates, question_ids, correct[date], flip[date], truth_flip[date]) aligned on question id."""
    sub = df[df["model"] == model]
    dates = [BASELINE] + sorted(d for d in sub["system_date"].unique() if d != BASELINE)
    qids = sorted(sub["question_id"].unique())
    by = {d: sub[sub["system_date"] == d].set_index("question_id").reindex(qids) for d in dates}
    for d, t in by.items():
        if t["response"].isna().any():
            missing = list(t.index[t["response"].isna()])[:5]
            raise ValueError(f"{model} / {d}: missing rows for questions {missing} ...")
    base = by[BASELINE]
    correct = {d: by[d]["correct"].to_numpy(dtype=float) for d in dates}
    flip = {
        d: np.array([0.0 if _same(a, b) else 1.0 for a, b in zip(base["parsed"], by[d]["parsed"])])
        for d in dates
        if d != BASELINE
    }
    truth_flip = {
        d: np.array([0.0 if a == b else 1.0 for a, b in zip(base["truth"], by[d]["truth"])])
        for d in dates
        if d != BASELINE
    }
    return dates, qids, correct, flip, truth_flip


def boot_indices(n: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def ci(values: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    """Point estimate and 95% percentile CI of the mean, resampling questions with ``idx``."""
    boots = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def score(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cells, headline).

    cells: one row per model x system_date with accuracy, accuracy delta vs baseline,
    flip rate vs baseline (each with 95% paired-bootstrap CI), unparsed rate and the
    'truth flip' rate (how often the correct answer itself changes vs the baseline
    reference date, i.e. the flip rate a perfect model would show).

    headline: one row per model, flip rate averaged over the stated dates, with a paired
    bootstrap CI (the same resampled questions are used for every date in a replicate).
    """
    cells, heads = [], []
    for model in sorted(df["model"].unique()):
        dates, qids, correct, flip, truth_flip = matrices(df, model)
        idx = boot_indices(len(qids), n_boot, seed)
        sub = df[df["model"] == model]
        for d in dates:
            acc, acc_lo, acc_hi = ci(correct[d], idx)
            row = {
                "model": model,
                "system_date": d,
                "n": len(qids),
                "accuracy": acc,
                "acc_lo": acc_lo,
                "acc_hi": acc_hi,
                "unparsed": float(sub[sub["system_date"] == d]["unparsed"].mean()),
            }
            if d == BASELINE:
                row.update({k: np.nan for k in ["acc_delta", "delta_lo", "delta_hi", "flip", "flip_lo", "flip_hi", "truth_flip"]})
            else:
                dl, dlo, dhi = ci(correct[d] - correct[BASELINE], idx)
                fl, flo, fhi = ci(flip[d], idx)
                row.update(
                    {
                        "acc_delta": dl,
                        "delta_lo": dlo,
                        "delta_hi": dhi,
                        "flip": fl,
                        "flip_lo": flo,
                        "flip_hi": fhi,
                        "truth_flip": float(truth_flip[d].mean()),
                    }
                )
            cells.append(row)
        stated = [d for d in dates if d != BASELINE]
        per_q = np.mean([flip[d] for d in stated], axis=0)  # per-question mean flip across dates
        m, lo, hi = ci(per_q, idx)
        heads.append({"model": model, "n_dates": len(stated), "mean_flip": m, "flip_lo": lo, "flip_hi": hi})
    return pd.DataFrame(cells), pd.DataFrame(heads)


def _pct(x: float) -> str:
    return "" if pd.isna(x) else f"{100 * x:.1f}"


def format_tables(cells: pd.DataFrame, head: pd.DataFrame) -> str:
    lines = [
        "| model | system date | n | accuracy % [95% CI] | Δacc vs none (pp) [95% CI] | flip rate % [95% CI] | truth-flip % | unparsed % |",
        "|---|---|---:|---|---|---|---:|---:|",
    ]
    for _, r in cells.iterrows():
        acc = f"{_pct(r.accuracy)} [{_pct(r.acc_lo)}, {_pct(r.acc_hi)}]"
        if r.system_date == BASELINE:
            delta = flip = "— (baseline)"
            tf = "—"
        else:
            delta = f"{100 * r.acc_delta:+.1f} [{100 * r.delta_lo:+.1f}, {100 * r.delta_hi:+.1f}]"
            flip = f"{_pct(r.flip)} [{_pct(r.flip_lo)}, {_pct(r.flip_hi)}]"
            tf = _pct(r.truth_flip)
        lines.append(f"| {r.model} | {r.system_date} | {r.n} | {acc} | {delta} | {flip} | {tf} | {_pct(r.unparsed)} |")
    lines += [
        "",
        "Headline: answer flip rate vs no-date baseline, averaged over stated dates (paired bootstrap 95% CI)",
        "",
        "| model | dates averaged | mean flip rate % [95% CI] |",
        "|---|---:|---|",
    ]
    for _, r in head.iterrows():
        lines.append(f"| {r.model} | {r.n_dates} | {_pct(r.mean_flip)} [{_pct(r.flip_lo)}, {_pct(r.flip_hi)}] |")
    return "\n".join(lines)
