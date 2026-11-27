"""Offline end-to-end scoring test on tests/fixtures/cached_results.jsonl.

Expected numbers are derived by hand in tests/make_fixture.py's docstring/table.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from datedrift.scoring import load_rows, score

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "cached_results.jsonl"


@pytest.fixture(scope="module")
def scored():
    df = load_rows(FIXTURE)
    cells, head = score(df)
    return df, cells.set_index(["model", "system_date"]), head.set_index("model")


def test_fixture_is_current():
    """The committed fixture equals what make_fixture.py generates."""
    sys.path.insert(0, str(ROOT / "tests"))
    import json

    import make_fixture

    committed = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
    assert committed == make_fixture.rows()


def test_shape(scored):
    df, cells, head = scored
    assert len(df) == 2 * 5 * 4
    assert len(cells) == 10 and len(head) == 2
    assert (cells["n"] == 4).all()


def test_oracle_accuracy_and_flip(scored):
    _, cells, head = scored
    o = cells.loc["fake-oracle"]
    assert np.allclose(o["accuracy"], 1.0)
    assert np.allclose(o["unparsed"], 0.0)
    assert o.loc["2023-06-01", "flip"] == 1.0
    assert o.loc["2025-01-01", "flip"] == 1.0
    assert o.loc["2026-11-27", "flip"] == 0.0
    assert o.loc["2030-01-01", "flip"] == 1.0
    # A perfect model flips exactly when the truth flips.
    stated = o.drop("none")
    assert np.allclose(stated["flip"], stated["truth_flip"])
    assert head.loc["fake-oracle", "mean_flip"] == pytest.approx(0.75)


def test_sticky_numbers(scored):
    _, cells, head = scored
    s = cells.loc["fake-sticky"]
    assert s["accuracy"].to_dict() == pytest.approx(
        {"none": 1.0, "2023-06-01": 0.0, "2025-01-01": 0.25, "2026-11-27": 0.5, "2030-01-01": 1.0}
    )
    assert s.drop("none")["flip"].to_dict() == pytest.approx(
        {"2023-06-01": 0.0, "2025-01-01": 0.25, "2026-11-27": 0.5, "2030-01-01": 1.0}
    )
    assert s.drop("none")["acc_delta"].to_dict() == pytest.approx(
        {"2023-06-01": -1.0, "2025-01-01": -0.75, "2026-11-27": -0.5, "2030-01-01": 0.0}
    )
    assert s.loc["2026-11-27", "unparsed"] == 0.25
    assert head.loc["fake-sticky", "mean_flip"] == pytest.approx(0.4375)


def test_confidence_intervals(scored):
    _, cells, head = scored
    stated = cells.drop(index="none", level="system_date")
    assert (cells["acc_lo"] <= cells["accuracy"] + 1e-12).all()
    assert (cells["accuracy"] <= cells["acc_hi"] + 1e-12).all()
    assert (stated["flip_lo"] <= stated["flip"] + 1e-12).all()
    assert (stated["flip"] <= stated["flip_hi"] + 1e-12).all()
    # Degenerate cells have zero-width intervals.
    assert cells.loc[("fake-oracle", "none"), ["acc_lo", "acc_hi"]].tolist() == [1.0, 1.0]
    assert cells.loc[("fake-sticky", "2023-06-01"), ["flip_lo", "flip_hi"]].tolist() == [0.0, 0.0]
    # Non-degenerate cell has a proper interval.
    lo, hi = cells.loc[("fake-sticky", "2025-01-01"), ["flip_lo", "flip_hi"]]
    assert lo == 0.0 and hi > 0.25
    h = head.loc["fake-sticky"]
    assert h.flip_lo <= h.mean_flip <= h.flip_hi


def test_bootstrap_is_deterministic():
    df = load_rows(FIXTURE)
    a, ha = score(df)
    b, hb = score(df)
    assert a.equals(b) and ha.equals(hb)


def test_score_cli_prints_table():
    out = subprocess.run(
        [sys.executable, "score.py", str(FIXTURE)], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    assert "accuracy % [95% CI]" in out and "flip rate % [95% CI]" in out
    assert "| fake-sticky | 2025-01-01 | 4 | 25.0 [" in out
    assert "| fake-sticky | 4 | 43.8 [" in out  # headline mean flip
