"""Seeded synthetic data for the scalability sweep.

No dataset in ``data/`` is large enough to reach a million rows, and metrics
have type requirements, so the sweep generates frames with a fixed mix of
column kinds. Generation is deterministic per seed so measurements taken on
different days remain comparable.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

COLUMN_KINDS: tuple[str, ...] = (
    "integer",
    "float",
    "categorical",
    "text",
    "date",
    "sparse",
)

_WORDS: tuple[str, ...] = (
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
)

_CONSONANTS: str = "bcdfghjklmnpqrstvwxyz"
_VOWELS: str = "aeiou"

_CATEGORIES: tuple[str, ...] = ("north", "south", "east", "west", "central")

NULL_FRACTION: float = 0.05
DUPLICATE_FRACTION: float = 0.10


def make_frame(rows: int, cols: int, seed: int = 13) -> pd.DataFrame:
    """
    Build a synthetic frame with a repeating mix of column kinds.

    The frame deliberately contains nulls and duplicate rows so completeness
    and minimality metrics have something to measure rather than trivially
    scoring 1.0.

    :param rows: Number of rows.
    :param cols: Number of columns.
    :param seed: Seed controlling all randomness.
    :return: The generated DataFrame.
    """
    if rows < 1 or cols < 1:
        raise ValueError(f"rows and cols must be positive, got {rows} and {cols}.")

    rng = np.random.default_rng(seed)
    unique_rows = max(1, rows - int(rows * DUPLICATE_FRACTION))

    data: dict[str, np.ndarray] = {}
    for i in range(cols):
        kind = COLUMN_KINDS[i % len(COLUMN_KINDS)]
        name = f"{kind}_{i}"
        data[name] = _make_column(kind, unique_rows, rng)

    frame = pd.DataFrame(data)

    if rows > unique_rows:
        extra = frame.iloc[rng.integers(0, unique_rows, rows - unique_rows)]
        frame = pd.concat([frame, extra], ignore_index=True)

    return frame.reset_index(drop=True)


def _make_column(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate one column of the given kind.

    :param kind: One of :data:`COLUMN_KINDS`.
    :param n: Number of values.
    :param rng: The seeded generator.
    :return: The column values.
    """
    if kind == "integer":
        return rng.integers(0, 1_000_000, n)

    if kind == "float":
        return rng.normal(100.0, 25.0, n)

    if kind == "categorical":
        return rng.choice(_CATEGORIES, n)

    if kind == "text":
        # _WORDS alone caps two-word cells at 16**2 == 256 distinct values.
        # Real corpora grow their vocabulary roughly per Heaps' law (a
        # sublinear power of the corpus length), so a fixed-size pool makes
        # a per-token cache (e.g. readability_wordnet's WordNet lookups)
        # saturate within a few thousand rows and look far more scalable
        # than it is on real text. sqrt(n) distinct base words keeps the
        # pool -- and so the number of distinct two-word combinations --
        # growing with the row count.
        vocab_size = max(len(_WORDS), math.isqrt(n) + 1)
        pool = _word_pool(vocab_size, rng)
        left = rng.choice(pool, n)
        right = rng.choice(pool, n)
        return np.char.add(np.char.add(left.astype(str), " "), right.astype(str))

    if kind == "date":
        offsets = rng.integers(0, 3_650, n)
        return (pd.Timestamp("2015-01-01") + pd.to_timedelta(offsets, unit="D")).values

    if kind == "sparse":
        values = rng.choice(_CATEGORIES, n).astype(object)
        mask = rng.random(n) < NULL_FRACTION * 4
        values[mask] = None
        return values

    raise ValueError(f"Unknown column kind {kind!r}.")


def _word_pool(size: int, rng: np.random.Generator) -> list[str]:
    """
    Build a pronounceable word pool of at least ``size`` distinct tokens.

    :data:`_WORDS` is the base of the pool; when more words are needed the
    pool is extended with generated consonant-vowel syllables so the tokens
    stay word-like (and thus behave sensibly under WordNet-based scoring)
    while remaining fully reproducible from ``rng``'s state.

    :param size: Minimum number of distinct words required.
    :param rng: The seeded generator; its state advances by roughly
        ``size - len(_WORDS)`` draws.
    :return: A list of at least ``size`` distinct lowercase words.
    """
    words = list(_WORDS)
    seen = set(words)
    while len(words) < size:
        syllables = int(rng.integers(2, 4))
        word = "".join(
            str(rng.choice(list(_CONSONANTS))) + str(rng.choice(list(_VOWELS)))
            for _ in range(syllables)
        )
        if word not in seen:
            words.append(word)
            seen.add(word)
    return words
