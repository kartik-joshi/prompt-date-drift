import datetime as dt
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from datedrift import SYSTEM_DATES, reference_date
from datedrift.client import Cache, ask, build_messages
from datedrift.questions import KINDS, full_years_between, generate, truth

ROOT = Path(__file__).resolve().parents[1]


def test_generate_200_balanced_unique():
    qs = generate(0)
    assert len(qs) == 200
    assert len({q["id"] for q in qs}) == 200
    assert Counter(q["kind"] for q in qs) == {k: 50 for k in KINDS}


def test_seed_determinism():
    assert generate(0) == generate(0)
    assert generate(0) != generate(1)


def test_cli_emits_200_lines():
    out = subprocess.run([sys.executable, "gen_questions.py", "--seed", "0"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    assert len(lines) == 200
    assert all(json.loads(l)["text"] for l in lines)


def test_truth_hand_checked():
    q = {"kind": "days_until", "target": "2027-04-27"}
    assert truth(q, dt.date(2026, 11, 27)) == 151
    assert truth(q, dt.date(2030, 1, 1)) == -980
    assert truth({"kind": "weekday_of", "offset": 0}, dt.date(2026, 11, 27)) == "Friday"
    assert truth({"kind": "weekday_of", "offset": 3}, dt.date(2025, 1, 1)) == "Saturday"
    assert truth({"kind": "age_in_year", "birth_year": 1982}, dt.date(2023, 6, 1)) == 41
    assert truth({"kind": "years_since", "event_date": "1967-01-30"}, dt.date(2025, 1, 1)) == 57
    assert full_years_between(dt.date(2000, 2, 29), dt.date(2001, 2, 28)) == 0
    assert full_years_between(dt.date(2000, 2, 29), dt.date(2001, 3, 1)) == 1


def test_every_question_truth_depends_on_some_date():
    """Every item is date-sensitive: its truth differs across the stated dates."""
    for q in generate(0):
        values = {truth(q, reference_date(d)) for d in SYSTEM_DATES}
        assert len(values) > 1, q["id"]


def test_system_prompt_only_differs_by_date():
    q = generate(0)[0]
    base = build_messages(None, q["text"])
    dated = build_messages("2025-01-01", q["text"])
    assert base[1] == dated[1]
    assert dated[0]["content"] == base[0]["content"] + " Today's date is 2025-01-01."


def test_cache_hits_and_invalidation(tmp_path):
    calls = []

    def fake_chat(model, messages, options):
        calls.append(messages)
        return "42"

    q = generate(0)[0]
    cache = Cache(tmp_path / "c.jsonl")
    r1 = ask(cache, "m", "2025-01-01", q, chat_fn=fake_chat)
    r2 = ask(cache, "m", "2025-01-01", q, chat_fn=fake_chat)
    assert r1 == r2 and len(calls) == 1
    # Reload from disk: still a hit.
    ask(Cache(tmp_path / "c.jsonl"), "m", "2025-01-01", q, chat_fn=fake_chat)
    assert len(calls) == 1
    # Different system date or model is a different key.
    ask(cache, "m", None, q, chat_fn=fake_chat)
    ask(cache, "m2", "2025-01-01", q, chat_fn=fake_chat)
    assert len(calls) == 3
    # Changed prompt text -> stale hash -> re-query.
    ask(cache, "m", "2025-01-01", dict(q, text=q["text"] + " Please."), chat_fn=fake_chat)
    assert len(calls) == 4
