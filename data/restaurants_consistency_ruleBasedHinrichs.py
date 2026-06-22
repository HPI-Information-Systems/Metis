"""Hinrichs consistency rules for the demo restaurants dataset."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

_DATA_PATH: Path = Path(__file__).resolve().parent / "restaurants.csv"
_TYPE_MIN_COUNT: int = 5

_PHONE_REGEX: re.Pattern[str] = re.compile(r"^\d{3}[-./\s]\d{3}[-./\s]\d{4}$")


def _load_type_vocab(path: Path = _DATA_PATH, min_count: int = _TYPE_MIN_COUNT) -> set[str]:
    df = pd.read_csv(path, usecols=["type"])
    counts = df["type"].dropna().str.lower().value_counts()
    return set(counts[counts >= min_count].index)


_TYPE_VOCAB: set[str] = _load_type_vocab()


def phone_format_violation(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return 0.0 if _PHONE_REGEX.fullmatch(str(value).strip()) else 1.0


def type_vocab_violation(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return 0.0 if str(value).strip().lower() in _TYPE_VOCAB else 1.0


def avg_rating_range_violation(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(max(0.0, value - 5.0)) + float(max(0.0, 1.0 - value))


def reviews_non_negative_violation(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    return float(max(0.0, -value))


def date_order_violation(row: pd.Series) -> float:
    first = row.get("first_review_date")
    last = row.get("last_review_date")
    if pd.isna(first) or pd.isna(last):
        return 0.0
    first_ts = pd.Timestamp(first)
    last_ts = pd.Timestamp(last)
    if last_ts >= first_ts:
        return 0.0
    return float((first_ts - last_ts).days) / 365.0


attribute_rules: dict[str, list] = {
    "phone": [phone_format_violation],
    "type": [type_vocab_violation],
    "avg_rating": [avg_rating_range_violation],
    "total_reviews_count": [reviews_non_negative_violation],
}

tuple_rules: list = [date_order_violation]
