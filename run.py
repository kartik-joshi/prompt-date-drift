"""Sweep 5 system dates x N models over the 200 questions; write results/results.jsonl.

Raw responses are cached in results/cache.jsonl keyed by (model, system_date, question_id),
so the sweep is resumable and re-running it is free.

Usage: python run.py [--models qwen2.5:3b-instruct llama3.2:3b] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from datedrift import MODELS, SYSTEM_DATES, date_label
from datedrift.client import OPTIONS, Cache, ask, system_prompt
from datedrift.parser import parse
from datedrift.questions import generate


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seed", type=int, default=0, help="question-generator seed")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke test)")
    ap.add_argument("--cache", default="results/cache.jsonl")
    ap.add_argument("--out", default="results/results.jsonl")
    args = ap.parse_args(argv)

    questions = generate(args.seed)[: args.limit]
    cache = Cache(args.cache)
    commit = git_commit()
    rows = []
    total = len(args.models) * len(SYSTEM_DATES) * len(questions)
    t0 = time.time()
    done = 0
    for model in args.models:
        for sd in SYSTEM_DATES:
            for q in questions:
                r = ask(cache, model, sd, q)
                rows.append(
                    {
                        "model": model,
                        "system_date": date_label(sd),
                        "system_prompt": system_prompt(sd),
                        "question_id": q["id"],
                        "question": q,
                        "response": r["response"],
                        "parsed": parse(r["response"], q["answer_type"]),
                        "options": OPTIONS,
                        "commit": commit,
                    }
                )
                done += 1
                if done % 50 == 0 or done == total:
                    el = time.time() - t0
                    print(f"[{done}/{total}] {model} {date_label(sd)} {el:.0f}s elapsed", file=sys.stderr, flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
