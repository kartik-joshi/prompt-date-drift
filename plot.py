"""Plot flip rate (vs no-date baseline) per stated system date, with 95% bootstrap CIs.

Usage: python plot.py results/results.jsonl --out results/flip_rate.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from datedrift.scoring import load_rows, score

COLORS = ["#1f6fb4", "#e07b00"]  # blue / orange, distinguishable with common colour-vision deficiencies


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--out", default="results/flip_rate.png")
    args = ap.parse_args(argv)

    cells, _ = score(load_rows(args.results))
    cells = cells[cells.system_date != "none"]
    dates = sorted(cells.system_date.unique())
    models = sorted(cells.model.unique())
    x = np.arange(len(dates))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for i, m in enumerate(models):
        c = cells[cells.model == m].set_index("system_date").loc[dates]
        xs = x - 0.4 + width * (i + 0.5)
        err = np.vstack([c.flip - c.flip_lo, c.flip_hi - c.flip]) * 100
        ax.bar(xs, c.flip * 100, width * 0.9, color=COLORS[i % len(COLORS)], label=m)
        ax.errorbar(xs, c.flip * 100, yerr=err, fmt="none", ecolor="#333333", capsize=3, lw=1)
    truth = cells.groupby("system_date").truth_flip.first().loc[dates] * 100
    ax.scatter(x, truth, marker="_", s=900, color="#333333", zorder=3, label="% of true answers that differ from run-date (2026-11-27) truth")
    ax.set_xticks(x, dates)
    ax.set_xlabel("date stated in system prompt")
    ax.set_ylabel("answers changed vs no-date baseline (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Answer flip rate by system-prompt date (200 questions, 95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
