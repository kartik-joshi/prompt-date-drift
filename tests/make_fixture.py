"""Regenerate tests/fixtures/cached_results.jsonl (hand-designed, no model needed).

Two fake models answer the first question of each kind (4 questions) under all 5 dates:

* fake-oracle: always right (responses written in varied formats to exercise the parser).
* fake-sticky: mostly repeats its no-date answers, with one refusal; flips chosen so the
  expected numbers are easy to verify by hand (see tests/test_score_fixture.py).

Hand-checked truths (none is scored against 2026-11-27):
                    none   2023-06-01  2025-01-01  2026-11-27  2030-01-01
  days_until-000     151   1426        846         151         -980
  weekday_of-000     Fri   Thu         Wed         Fri         Tue
  age_in_year-000    44    41          43          44          48
  years_since-000    59    56          57          59          62
"""

import json
from pathlib import Path

from datedrift.client import OPTIONS, system_prompt
from datedrift.questions import generate

QIDS = ["days_until-000", "weekday_of-000", "age_in_year-000", "years_since-000"]
DATES = ["none", "2023-06-01", "2025-01-01", "2026-11-27", "2030-01-01"]

RESPONSES = {
    "fake-oracle": {
        "none": ["151", "Friday", "44", "59"],
        "2023-06-01": ["1,426", "It is Thursday.", "Answer: 41", "56 years"],
        "2025-01-01": ["846 days", "Wednesday", "43", "**57**"],
        "2026-11-27": ["151", "friday", "44.", "Answer: 59"],
        "2030-01-01": ["-980", "Tue", "48", "There have been 62 full years."],
    },
    "fake-sticky": {
        "none": ["151", "Friday", "44", "Answer: 59"],
        "2023-06-01": ["151", "Friday", "44", "59"],
        "2025-01-01": ["846", "Friday", "44", "59"],
        "2026-11-27": ["I cannot know the current date.", "Monday", "44", "59"],
        "2030-01-01": ["-980", "Tuesday", "48", "62"],
    },
}


def rows() -> list[dict]:
    qs = {q["id"]: q for q in generate(0)}
    out = []
    for model, by_date in RESPONSES.items():
        for d in DATES:
            for qid, resp in zip(QIDS, by_date[d]):
                out.append(
                    {
                        "model": model,
                        "system_date": d,
                        "system_prompt": system_prompt(None if d == "none" else d),
                        "question_id": qid,
                        "question": qs[qid],
                        "response": resp,
                        "options": OPTIONS,
                        "commit": "fixture",
                    }
                )
    return out


if __name__ == "__main__":
    path = Path(__file__).parent / "fixtures" / "cached_results.jsonl"
    with path.open("w") as f:
        for r in rows():
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path}")
