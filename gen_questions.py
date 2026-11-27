"""Emit the synthetic question set as JSONL on stdout (one question per line).

Usage: python gen_questions.py --seed 0 > data/questions.jsonl
"""

import argparse
import json
import sys

from datedrift.questions import generate


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    for q in generate(args.seed):
        sys.stdout.write(json.dumps(q) + "\n")


if __name__ == "__main__":
    main()
