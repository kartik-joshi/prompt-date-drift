"""prompt-date-drift: does the system-prompt date change a local model's answers?"""

import datetime as _dt

# The five system-prompt conditions swept by run.py. ``None`` means no date is stated.
SYSTEM_DATES: list[str | None] = [None, "2023-06-01", "2025-01-01", "2026-11-27", "2030-01-01"]

# Ground truth is undefined when no date is stated. For the accuracy column of the
# "none" cell we score against the real calendar date the sweep was run on.
NONE_REFERENCE_DATE = _dt.date(2026, 11, 27)

MODELS = ["qwen2.5:3b-instruct", "llama3.2:3b"]


def date_label(system_date: str | None) -> str:
    return "none" if system_date is None else system_date


def reference_date(system_date: str | None) -> _dt.date:
    if system_date is None:
        return NONE_REFERENCE_DATE
    return _dt.date.fromisoformat(system_date)
