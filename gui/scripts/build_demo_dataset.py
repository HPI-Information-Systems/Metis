#!/usr/bin/env python3
"""Build the demo restaurants CSV: source rows + synthetic columns + seeded noise. See README."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SEED: int = 42
SYNTHETIC_NULL_PCT: float = 0.10
ORIGINAL_NULL_PCT: float = 0.03
INVERTED_DATE_PCT: float = 0.02
INVALID_RATING_PCT: float = 0.02
NEGATIVE_REVIEWS_PCT: float = 0.02

RATING_ALPHA: float = 8.0
RATING_BETA: float = 2.0
REVIEW_COUNT_SCALE: float = 60.0
DATE_RANGE_START: date = date(2010, 1, 1)
DATE_RANGE_END: date = date(2022, 1, 1)
LAST_REVIEW_GAP_MIN_DAYS: int = 30
LAST_REVIEW_GAP_MAX_DAYS: int = 365 * 5

SYNTHETIC_COLS: tuple[str, ...] = (
    "avg_rating",
    "total_reviews_count",
    "first_review_date",
    "last_review_date",
)
ORIGINAL_COLS: tuple[str, ...] = ("name", "address", "city", "phone", "type")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.source)
    df = _add_synthetic_columns(df, rng)
    df = _inject_value_violations(df, rng)
    df = _inject_nulls(df, rng)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows × {df.shape[1]} cols → {args.output}")


def _add_synthetic_columns(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    df = df.copy()
    df["avg_rating"] = (1.0 + rng.beta(RATING_ALPHA, RATING_BETA, n) * 4.0).round(1)
    df["total_reviews_count"] = (
        rng.exponential(REVIEW_COUNT_SCALE, n).round().astype(int)
    )
    span_days = (DATE_RANGE_END - DATE_RANGE_START).days
    first_offsets = rng.integers(0, span_days, n)
    gap_offsets = rng.integers(
        LAST_REVIEW_GAP_MIN_DAYS, LAST_REVIEW_GAP_MAX_DAYS, n
    )
    first_dates = [DATE_RANGE_START + timedelta(days=int(o)) for o in first_offsets]
    last_dates = [
        f + timedelta(days=int(g)) for f, g in zip(first_dates, gap_offsets)
    ]
    df["first_review_date"] = [d.isoformat() for d in first_dates]
    df["last_review_date"] = [d.isoformat() for d in last_dates]
    return df


def _inject_value_violations(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    inv_mask = rng.random(n) < INVERTED_DATE_PCT
    df.loc[inv_mask, ["first_review_date", "last_review_date"]] = (
        df.loc[inv_mask, ["last_review_date", "first_review_date"]].values
    )

    bad_rating_mask = rng.random(n) < INVALID_RATING_PCT
    df.loc[bad_rating_mask, "avg_rating"] = rng.choice(
        [-0.5, 0.0, 5.5, 6.0, 7.0], size=int(bad_rating_mask.sum())
    )

    neg_reviews_mask = rng.random(n) < NEGATIVE_REVIEWS_PCT
    df.loc[neg_reviews_mask, "total_reviews_count"] = -rng.integers(
        1, 20, size=int(neg_reviews_mask.sum())
    )

    return df


def _inject_nulls(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    for col in SYNTHETIC_COLS:
        mask = rng.random(len(df)) < SYNTHETIC_NULL_PCT
        df.loc[mask, col] = np.nan
    for col in ORIGINAL_COLS:
        mask = rng.random(len(df)) < ORIGINAL_NULL_PCT
        df.loc[mask, col] = np.nan
    return df


if __name__ == "__main__":
    main()
