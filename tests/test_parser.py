"""Parser unit tests on 20+ hand-written model outputs."""

import pytest

from datedrift.parser import parse

INT_CASES = [
    ("112", 112),
    ("-119", -119),
    ("-1,046", -1046),
    ("1,426 days", 1426),
    ("Answer: 41", 41),
    ("**57**", 57),
    ("There are 151 days until April 27, 2027.\nAnswer: 151", 151),
    ("The person turns 44 this year.", 44),
    ("−980", -980),  # unicode minus
    ("112 días", 112),
    ("It has been 59 full years.", 59),
    ("Answer: -3", -3),
    ("From 2025-01-01 to 2027-04-27 there are 846 days.", 846),
    ("I'm sorry, but I cannot provide a date that is in the future.", None),
    ("", None),
    ("I don't know today's date.", None),
]

DAY_CASES = [
    ("Friday", "Friday"),
    ("friday.", "Friday"),
    ("Tue", "Tuesday"),
    ("It will be a Wednesday.", "Wednesday"),
    ("Today is Monday, so 3 days from today is Thursday.", "Thursday"),
    ("Answer: Sunday", "Sunday"),
    ("Answer: Sat", "Saturday"),
    ("**Thursday**", "Thursday"),
    ("I don't have access to the current date.", None),
    ("", None),
]


@pytest.mark.parametrize("text,expected", INT_CASES)
def test_parse_int(text, expected):
    assert parse(text, "int") == expected


@pytest.mark.parametrize("text,expected", DAY_CASES)
def test_parse_weekday(text, expected):
    assert parse(text, "weekday") == expected


def test_case_count():
    assert len(INT_CASES) + len(DAY_CASES) >= 20


def test_unknown_type():
    with pytest.raises(ValueError):
        parse("1", "float")
