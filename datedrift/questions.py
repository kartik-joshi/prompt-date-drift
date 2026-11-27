"""Seeded synthetic question generator with date-relative ground truth.

Each question is a plain dict (JSON-serialisable). Ground truth is never stored as a
fixed value: ``truth(question, ref_date)`` computes it from the question parameters and
whichever reference date the system prompt stated.
"""

from __future__ import annotations

import datetime as dt
import random

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
KINDS = ["days_until", "weekday_of", "age_in_year", "years_since"]
N_PER_KIND = 50

# Direct-answer format: short outputs keep 2000 local calls affordable (see README).
INT_INSTRUCTION = "Reply with only the answer, as a single integer."
DAY_INSTRUCTION = "Reply with only the answer, as a single weekday name."

EVENTS = [
    "the Riverside community library opened",
    "the Maple Street bridge was completed",
    "a small software company was founded",
    "the local chess club held its first meeting",
    "the town's first bakery opened",
    "a time capsule was buried at the school",
    "the harbour lighthouse was restored",
    "the city orchestra gave its first concert",
]


def _fmt(d: dt.date) -> str:
    return f"{d.strftime('%B')} {d.day}, {d.year}"


def _rand_date(rng: random.Random, start: dt.date, end: dt.date) -> dt.date:
    return start + dt.timedelta(days=rng.randint(0, (end - start).days))


def full_years_between(earlier: dt.date, later: dt.date) -> int:
    """Completed years from ``earlier`` to ``later`` (negative if later < earlier)."""
    if later < earlier:
        return -full_years_between(later, earlier)
    years = later.year - earlier.year
    if (later.month, later.day) < (earlier.month, earlier.day):
        years -= 1
    return years


def generate(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    items: list[dict] = []
    for kind in KINDS:
        for i in range(N_PER_KIND):
            qid = f"{kind}-{i:03d}"
            if kind == "days_until":
                target = _rand_date(rng, dt.date(2023, 1, 1), dt.date(2031, 12, 31))
                text = (
                    f"How many days are there from today until {_fmt(target)}? "
                    "If that date is already in the past, give a negative number. " + INT_INSTRUCTION
                )
                q = {"id": qid, "kind": kind, "answer_type": "int", "target": target.isoformat(), "text": text}
            elif kind == "weekday_of":
                offset = 0 if i < 5 else rng.randint(1, 60)
                if offset == 0:
                    text = "What day of the week is it today? " + DAY_INSTRUCTION
                else:
                    text = f"What day of the week will it be {offset} days from today? " + DAY_INSTRUCTION
                q = {"id": qid, "kind": kind, "answer_type": "weekday", "offset": offset, "text": text}
            elif kind == "age_in_year":
                birth_year = rng.randint(1940, 2020)
                text = (
                    f"A person was born in {birth_year}. How old do they turn this calendar year? "
                    + INT_INSTRUCTION
                )
                q = {"id": qid, "kind": kind, "answer_type": "int", "birth_year": birth_year, "text": text}
            else:  # years_since
                event_date = _rand_date(rng, dt.date(1950, 1, 1), dt.date(2022, 12, 31))
                event = rng.choice(EVENTS)
                text = (
                    f"On {_fmt(event_date)}, {event}. How many full years have passed since then? "
                    + INT_INSTRUCTION
                )
                q = {"id": qid, "kind": kind, "answer_type": "int", "event_date": event_date.isoformat(), "text": text}
            items.append(q)
    return items


def truth(q: dict, ref: dt.date) -> int | str:
    kind = q["kind"]
    if kind == "days_until":
        return (dt.date.fromisoformat(q["target"]) - ref).days
    if kind == "weekday_of":
        return WEEKDAYS[(ref + dt.timedelta(days=q["offset"])).weekday()]
    if kind == "age_in_year":
        return ref.year - q["birth_year"]
    if kind == "years_since":
        return full_years_between(dt.date.fromisoformat(q["event_date"]), ref)
    raise ValueError(f"unknown kind {kind!r}")
