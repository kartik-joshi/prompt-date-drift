"""Extract an integer or weekday from free-form model output.

Strategy: prefer an explicit ``Answer: <value>`` line (the format the prompt asks for,
last one wins); otherwise fall back to the last integer / weekday mentioned anywhere.
Returns ``None`` when nothing can be extracted; callers log that as "unparsed".
"""

from __future__ import annotations

import re

from .questions import WEEKDAYS

_MINUS = "−–—"  # unicode minus, en dash, em dash
_INT = rf"[-{_MINUS}]?\s?\d[\d,]*"
_ANSWER_INT = re.compile(rf"answer\s*(?:is)?\s*[:=]?\s*\**\s*({_INT})", re.IGNORECASE)
_ANY_INT = re.compile(rf"(?<![\w.])({_INT})(?![\d.]\d)")

_DAY_ALIASES = {}
for _d in WEEKDAYS:
    _DAY_ALIASES[_d.lower()] = _d
    _DAY_ALIASES[_d[:3].lower()] = _d
_DAY_ALIASES.update({"tues": "Tuesday", "wed": "Wednesday", "thur": "Thursday", "thurs": "Thursday"})
_DAY_WORD = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tues?|wed|thu(?:rs?)?|fri|sat|sun)\b"
_ANSWER_DAY = re.compile(rf"answer\s*(?:is)?\s*[:=]?\s*\**\s*{_DAY_WORD}", re.IGNORECASE)
_ANY_DAY = re.compile(rf"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)


def _to_int(s: str) -> int | None:
    s = s.strip()
    neg = s[:1] in "-" + _MINUS
    digits = s.lstrip("-" + _MINUS).strip().replace(",", "")
    if not digits.isdigit():
        return None
    v = int(digits)
    return -v if neg else v


def parse_int(text: str) -> int | None:
    if not text:
        return None
    hits = _ANSWER_INT.findall(text)
    if hits:
        return _to_int(hits[-1])
    hits = _ANY_INT.findall(text)
    if hits:
        return _to_int(hits[-1])
    return None


def parse_weekday(text: str) -> str | None:
    if not text:
        return None
    hits = _ANSWER_DAY.findall(text)
    if hits:
        return _DAY_ALIASES.get(hits[-1].lower())
    hits = _ANY_DAY.findall(text)
    if hits:
        return _DAY_ALIASES[hits[-1].lower()]
    return None


def parse(text: str, answer_type: str) -> int | str | None:
    if answer_type == "int":
        return parse_int(text)
    if answer_type == "weekday":
        return parse_weekday(text)
    raise ValueError(f"unknown answer_type {answer_type!r}")
