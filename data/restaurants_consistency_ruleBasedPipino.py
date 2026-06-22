"""Pipino consistency rules for the demo restaurants dataset."""
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


def phone_format_ok(value: Any) -> bool:
    if pd.isna(value):
        return True
    return _PHONE_REGEX.fullmatch(str(value).strip()) is not None


def type_in_vocab(value: Any) -> bool:
    if pd.isna(value):
        return True
    return str(value).strip().lower() in _TYPE_VOCAB


def avg_rating_in_range(value: Any) -> bool:
    if pd.isna(value):
        return True
    return 1.0 <= float(value) <= 5.0


def reviews_non_negative(value: Any) -> bool:
    if pd.isna(value):
        return True
    return float(value) >= 0.0


def date_order_ok(row: pd.Series) -> bool:
    first = row.get("first_review_date")
    last = row.get("last_review_date")
    if pd.isna(first) or pd.isna(last):
        return True
    return pd.Timestamp(last) >= pd.Timestamp(first)


def reviews_imply_date(row: pd.Series) -> bool:
    count = row.get("total_reviews_count")
    last = row.get("last_review_date")
    if pd.isna(count):
        return True
    if float(count) <= 0:
        return True
    return not pd.isna(last)


attribute_rules: dict[str, list] = {
    "phone": [phone_format_ok],
    "type": [type_in_vocab],
    "avg_rating": [avg_rating_in_range],
    "total_reviews_count": [reviews_non_negative],
}

tuple_rules: list = [date_order_ok, reviews_imply_date]
