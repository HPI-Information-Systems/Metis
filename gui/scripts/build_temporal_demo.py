#!/usr/bin/env python3
"""Generate two extra precomputed demo runs for the Comparison-over-time tab.

Slices the next two 1k-row chunks out of ``data/restaurants.csv`` (rows
1000-2000 and 2000-3000), runs every demo metric on each chunk, and writes two
backdated JSON snapshots into ``gui/demo/precomputed/`` alongside the existing
``restaurant_results.json``. All three snapshots share the same ``dataset``
name (``restaurant_sample``) so the temporal chart groups them into a single
time series — only the experiment tag and timestamp differ.

Usage:
    python gui/scripts/build_temporal_demo.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "gui" / "scripts"))

import metis.metric  # noqa: F401 — populate Metric.registry
from metis.metric.metric import Metric

from run_demo_pipeline import (  # type: ignore
    _get_metric_config,
    _json_default,
    _load_config_module,
    _preprocess_df,
    _result_to_dict,
)


SOURCE_CSV: Path = _root / "data" / "restaurants.csv"
OUT_DIR: Path = _root / "gui" / "demo" / "precomputed"
SLICE_CSV_DIR: Path = _root / "gui" / "demo"

# Match the existing demo's dataset name so all snapshots group together in
# load_temporal_data() and show up as one time series in the chart.
DATASET_NAME: str = "restaurant_sample"

# Each snapshot: (slice_start, slice_end, experiment_tag, timestamp, csv_filename,
#                  json_filename, null_pct, dup_pct, noise_seed).
# Older snapshots get heavier random null injection and row duplication so the
# temporal chart shows a clear "data quality improving over time" arc:
#   completeness rises, minimality rises, consistency rises.
# The existing restaurant_results.json (current) carries no noise.
SNAPSHOTS: list[tuple] = [
    (
        1000, 2000,
        "demo-2026-03-08",
        datetime(2026, 3, 8, 9, 0, 0),
        "restaurant_sample_t1.csv",
        "restaurant_results_t1.json",
        0.20, 0.05, 11,
    ),
    (
        2000, 3000,
        "demo-2026-03-22",
        datetime(2026, 3, 22, 9, 0, 0),
        "restaurant_sample_t2.csv",
        "restaurant_results_t2.json",
        0.10, 0.02, 22,
    ),
]


def main() -> None:
    """
    Generate the bundled temporal-demo snapshots.

    :return: None.
    """
    if not SOURCE_CSV.exists():
        sys.exit(f"Source CSV not found: {SOURCE_CSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {SOURCE_CSV}")
    full_df = pd.read_csv(SOURCE_CSV, low_memory=False)
    print(f"  Total rows: {len(full_df):,}")

    config_mod = _load_config_module(None)
    metric_names: list[str] = list(config_mod.DEMO_METRICS)
    print(f"  Demo metrics: {', '.join(metric_names)}")

    for start, end, tag, ts, csv_name, json_name, null_pct, dup_pct, seed in SNAPSHOTS:
        if end > len(full_df):
            print(f"\n!!! Not enough rows for {tag} (need {end}, have {len(full_df)}). Skipping.")
            continue
        df = full_df.iloc[start:end].reset_index(drop=True)
        df = _apply_noise(df, null_pct, dup_pct, seed)
        csv_path = SLICE_CSV_DIR / csv_name
        df.to_csv(csv_path, index=False)
        print(f"\n=== {json_name} ===")
        print(
            f"  rows {start}-{end} → {len(df):,} after noise "
            f"(null_pct={null_pct:.0%}, dup_pct={dup_pct:.0%}), tag={tag}, ts={ts.isoformat()}"
        )
        print(f"  wrote slice: {csv_path}")

        results = _compute_all(df, config_mod, metric_names, tag, ts)

        out_path = OUT_DIR / json_name
        payload = {
            "_generated_by": "build_temporal_demo.py",
            "results": results,
        }
        with out_path.open("w") as f:
            json.dump(payload, f, default=_json_default, indent=2)
        print(f"  wrote {len(results):,} results → {out_path}")


def _apply_noise(
    df: pd.DataFrame,
    null_pct: float,
    dup_pct: float,
    seed: int,
) -> pd.DataFrame:
    """
    Randomly null individual cells and duplicate whole rows.

    Deterministic given ``seed``. Nulls are injected by sampling ``(row, col)``
    pairs uniformly across the whole frame; duplicates by sampling rows and
    concatenating them at the end.

    :param df: The input dataframe.
    :param null_pct: Fraction of cells to null out.
    :param dup_pct: Fraction of rows to duplicate.
    :param seed: RNG seed for determinism.
    :return: A new dataframe with the noise applied.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    if null_pct > 0:
        n_cells = out.size
        n_null = int(n_cells * null_pct)
        if n_null > 0:
            row_idx = rng.integers(0, len(out), size=n_null)
            col_idx = rng.integers(0, out.shape[1], size=n_null)
            for r, c in zip(row_idx, col_idx):
                out.iat[int(r), int(c)] = None

    if dup_pct > 0:
        n_dup = int(len(out) * dup_pct)
        if n_dup > 0:
            dup_idx = rng.integers(0, len(out), size=n_dup)
            out = pd.concat([out, out.iloc[dup_idx]], ignore_index=True)

    return out


def _compute_all(
    df: pd.DataFrame,
    config_mod,
    metric_names: list[str],
    tag: str,
    ts: datetime,
) -> list[dict]:
    """
    Run all demo metrics on a snapshot dataframe.

    :param df: The (already noisy) snapshot dataframe.
    :param config_mod: The loaded config module.
    :param metric_names: Metric names to run.
    :param tag: Experiment tag stored in the results.
    :param ts: Backdated timestamp stored in the results.
    :return: A list of serialized result dicts.
    """
    out: list[dict] = []
    for name in metric_names:
        if name not in Metric.registry:
            print(f"  SKIP  {name}: not in registry")
            continue

        print(f"  RUN   {name} ...", end=" ", flush=True)
        tmp_path: str | None = None
        try:
            run_df = _preprocess_df(name, df, config_mod)
            cfg = _get_metric_config(name, config_mod)

            if name == "consistency_countFDViolations" and isinstance(cfg, dict):
                tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
                json.dump(cfg, tmp)
                tmp.close()
                tmp_path = tmp.name
                cfg = tmp_path

            batch = Metric.registry[name]().assess(run_df, None, cfg)
            for r in batch:
                d = _result_to_dict(r, tag, DATASET_NAME)
                d["timestamp"] = ts.isoformat()
                out.append(d)
            print(f"{len(batch):,} results")
        except Exception as exc:
            print(f"ERROR: {exc}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    return out


if __name__ == "__main__":
    main()
