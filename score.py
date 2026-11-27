"""Score a results file: accuracy, accuracy delta and flip rate per model x date with 95% CIs.

Usage: python score.py results/results.jsonl [--json results/summary.json] [--by-kind]
"""

from __future__ import annotations

import argparse
import json

from datedrift.scoring import N_BOOT, format_tables, load_rows, score


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--json", help="also write the cell and headline tables as JSON here")
    ap.add_argument("--by-kind", action="store_true", help="also print flip rate per question kind")
    args = ap.parse_args(argv)

    df = load_rows(args.results)
    cells, head = score(df, n_boot=args.n_boot)
    print(format_tables(cells, head))

    if args.by_kind:
        print("\nFlip rate % by question kind (point estimates, vs none):\n")
        print("| model | kind | " + " | ".join(d for d in sorted(df.system_date.unique()) if d != "none") + " |")
        stated = [d for d in sorted(df.system_date.unique()) if d != "none"]
        print("|---|---|" + "---:|" * len(stated))
        for (model, kind), sub in df.groupby(["model", "kind"]):
            c, _ = score(sub, n_boot=10)
            vals = [f"{100 * c[c.system_date == d].flip.iloc[0]:.1f}" for d in stated]
            print(f"| {model} | {kind} | " + " | ".join(vals) + " |")
        print("\nAccuracy % by question kind:\n")
        print("| model | kind | none | " + " | ".join(stated) + " |")
        print("|---|---|---:|" + "---:|" * len(stated))
        for (model, kind), sub in df.groupby(["model", "kind"]):
            acc = sub.groupby("system_date")["correct"].mean()
            print(f"| {model} | {kind} | " + " | ".join(f"{100 * acc[d]:.1f}" for d in ["none"] + stated) + " |")

    if args.by_kind:
        # age_in_year asks "how old do they turn this calendar year?", so
        # birth_year + parsed answer is the year the model implicitly believes it is.
        print("\nImplied current year from age_in_year answers (top 3, count of 50):\n")
        print("| model | system date | implied year: count |")
        print("|---|---|---|")
        ages = df[(df.kind == "age_in_year") & ~df.unparsed]
        for (model, d), sub in ages.groupby(["model", "system_date"]):
            years = (sub["parsed"].astype(int) + [q["birth_year"] for q in sub["question"]]).value_counts()
            print(f"| {model} | {d} | " + ", ".join(f"{y}: {n}" for y, n in years.head(3).items()) + " |")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {"cells": json.loads(cells.to_json(orient="records")), "headline": json.loads(head.to_json(orient="records"))},
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
