#!/usr/bin/env python3
"""Pre-compute DQ metrics on a demo dataset.

Writes results to a JSON file compatible with the GUI's import format. Supports
hash-based selective recomputation: metrics whose source code and input data
have not changed since the last run are skipped.

Usage:
    python gui/scripts/run_demo_pipeline.py \\
        --dataset gui/demo/restaurant_sample.csv \\
        --metrics completeness_nullRatio,minimality_duplicateCount \\
        --output gui/demo/precomputed/restaurant_results.json

    # Merge additional metrics into an existing file:
    python gui/scripts/run_demo_pipeline.py \\
        --dataset gui/demo/restaurant_sample.csv \\
        --metrics correctness_heinrich \\
        --max-rows 5000 \\
        --merge-into gui/demo/precomputed/restaurant_results.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path
from types import ModuleType

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

import pandas as pd

import metis.metric  # noqa: F401 — populate Metric.registry
from metis.metric.metric import Metric


def main() -> None:
    """
    CLI entry point.

    :return: None.
    """
    parser = _build_parser()
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    merge_path = Path(args.merge_into) if args.merge_into else None
    if args.output:
        output_path = Path(args.output)
    elif merge_path:
        output_path = merge_path
    else:
        parser.error("--output is required when --merge-into is not provided.")

    metric_names = _resolve_metric_names(args.metrics)

    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path, low_memory=False)
    if args.max_rows and len(df) > args.max_rows:
        df = df.head(args.max_rows)
        print(f"  Capped to {args.max_rows:,} rows.")
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")

    existing, existing_hashes = _load_existing(merge_path)
    if merge_path and merge_path.exists():
        print(f"Loaded {len(existing):,} existing results from {merge_path}.")

    data_hash = _hash_csv(dataset_path)
    config_mod = _load_config_module(args.config_module)

    new_results = _run_metrics(
        metric_names, df, args, config_mod,
        data_hash, existing_hashes, dataset_path,
    )

    run_metric_names = {r["DQmetric"] for r in new_results}
    kept = [r for r in existing if r["DQmetric"] not in run_metric_names]
    final_results = kept + new_results

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_hashes": existing_hashes,
        "_generated_by": "run_demo_pipeline.py",
        "results": final_results,
    }
    with output_path.open("w") as f:
        json.dump(payload, f, default=_json_default, indent=2)

    print(f"\nWrote {len(final_results):,} results → {output_path}")


def _build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser.

    :return: The configured argparse parser.
    """
    parser = argparse.ArgumentParser(description="Pre-compute Metis demo data.")
    parser.add_argument("--dataset", required=True, help="Path to input CSV.")
    parser.add_argument(
        "--metrics",
        required=True,
        help="Comma-separated list of metric names, or 'all'.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Required unless --merge-into is provided.",
    )
    parser.add_argument(
        "--merge-into",
        metavar="PATH",
        help="If supplied, merge new results into this existing JSON file "
             "(existing results for the same metric are replaced).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap input rows (useful for slow cell-level metrics).",
    )
    parser.add_argument(
        "--experiment-tag",
        default="demo",
        help="Experiment tag stored in results (default: demo).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached hashes and re-run everything.",
    )
    parser.add_argument(
        "--config-module",
        default=None,
        metavar="PATH",
        help="Path to a Python module that provides get_*_config() functions "
             "and FD_CONFIG for metrics that require configuration. "
             "Defaults to gui/demo/demo_metric_configs.py.",
    )
    return parser


def _resolve_metric_names(metrics_arg: str) -> list[str]:
    """
    Resolve the ``--metrics`` argument into a list of metric names.

    :param metrics_arg: The raw value of ``--metrics`` (``"all"`` or a comma-separated list).
    :return: The expanded list of metric names.
    """
    if metrics_arg.strip().lower() == "all":
        return list(Metric.registry.keys())
    return [m.strip() for m in metrics_arg.split(",") if m.strip()]


def _load_existing(merge_path: Path | None) -> tuple[list[dict], dict[str, str]]:
    """
    Load existing results and stored hashes for hash-based selective recomputation.

    :param merge_path: Path passed via ``--merge-into``, or ``None``.
    :return: ``(existing_results, existing_hashes)``. Both empty when no merge file exists.
    """
    existing: list[dict] = []
    existing_hashes: dict[str, str] = {}
    if merge_path and merge_path.exists():
        with merge_path.open() as f:
            payload = json.load(f)
        existing = payload.get("results", payload) if isinstance(payload, dict) else payload
        existing_hashes = payload.get("_hashes", {}) if isinstance(payload, dict) else {}
    return existing, existing_hashes


def _run_metrics(
    metric_names: list[str],
    df: pd.DataFrame,
    args: argparse.Namespace,
    config_mod: ModuleType,
    data_hash: str,
    existing_hashes: dict[str, str],
    dataset_path: Path,
) -> list[dict]:
    """
    Run the selected metrics, skipping those whose source/data hashes are unchanged.

    :param metric_names: Metric names to run.
    :param df: The (possibly capped) input dataframe.
    :param args: Parsed CLI args (used for ``force``, ``max_rows``, ``experiment_tag``).
    :param config_mod: The loaded config module.
    :param data_hash: Hash of the input CSV.
    :param existing_hashes: Mutable dict of cached ``cache_key -> hash`` (updated in place).
    :param dataset_path: Path of the input CSV (used to derive a dataset name).
    :return: A list of serialized result dicts.
    """
    new_results: list[dict] = []
    for name in metric_names:
        if name not in Metric.registry:
            print(f"  SKIP  {name}: not in registry")
            continue

        metric_hash = _hash_metric_source(name) + data_hash
        cache_key = f"{name}:{args.max_rows}"
        if not args.force and existing_hashes.get(cache_key) == metric_hash:
            print(f"  SKIP  {name}: unchanged (hash match)")
            continue

        print(f"  RUN   {name} ...", end=" ", flush=True)
        tmp_fd_path: str | None = None
        try:
            run_df = _preprocess_df(name, df, config_mod)
            metric_config = _get_metric_config(name, config_mod)
            if name == "consistency_countFDViolations" and isinstance(metric_config, dict):
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False,
                )
                json.dump(metric_config, tmp)
                tmp.close()
                tmp_fd_path = tmp.name
                metric_config = tmp_fd_path
            batch = Metric.registry[name]().assess(run_df, None, metric_config)
            serialized = [
                _result_to_dict(r, args.experiment_tag, dataset_path.stem) for r in batch
            ]
            new_results.extend(serialized)
            existing_hashes[cache_key] = metric_hash
            print(f"{len(serialized):,} results")
        except Exception as exc:
            print(f"ERROR: {exc}")
            if os.getenv("CI"):
                traceback.print_exc()
        finally:
            if tmp_fd_path and os.path.exists(tmp_fd_path):
                os.unlink(tmp_fd_path)

    return new_results


def _load_config_module(path: str | None) -> ModuleType:
    """
    Load a config module that provides ``get_*_config`` functions and ``FD_CONFIG``.

    :param path: Path to a Python file. Defaults to ``gui/demo/demo_metric_configs.py``.
    :return: The loaded module.
    """
    if path is None:
        path = str(Path(__file__).parent.parent / "demo" / "demo_metric_configs.py")
    spec = importlib.util.spec_from_file_location("_demo_configs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _preprocess_df(metric_name: str, df: pd.DataFrame, config_mod: ModuleType) -> pd.DataFrame:
    """
    Apply an optional per-metric dataframe preprocessor from the config module.

    Convention: ``preprocess_<suffix>(df)`` in the config module.

    :param metric_name: Metric the preprocessor applies to.
    :param df: The input dataframe.
    :param config_mod: The loaded config module.
    :return: The (possibly filtered) dataframe.
    """
    suffix = metric_name.split("_", 1)[1] if "_" in metric_name else metric_name
    fn = getattr(config_mod, f"preprocess_{suffix}", None)
    if callable(fn):
        return fn(df)
    return df


def _get_metric_config(metric_name: str, config_mod: ModuleType):
    """
    Resolve the metric config from the config module, falling back to a default.

    Convention in ``config_mod``:

    - ``consistency_countFDViolations`` → ``FD_CONFIG`` (dict)
    - ``consistency_ruleBasedHinrichs`` → ``get_hinrichs_config()``
    - ``consistency_ruleBasedPipino``  → ``get_pipino_config()``
    - ``timeliness_heinrich``          → ``get_timeliness_config()``

    :param metric_name: Metric to look up.
    :param config_mod: The loaded config module.
    :return: A config object (or a dict for the FD metric), or ``None``.
    """
    if metric_name == "consistency_countFDViolations":
        fd = getattr(config_mod, "FD_CONFIG", None)
        if fd is not None:
            return fd

    suffix = metric_name.split("_", 1)[1] if "_" in metric_name else metric_name
    for fn_name in (f"get_{suffix}_config", f"get_{metric_name}_config"):
        fn = getattr(config_mod, fn_name, None)
        if callable(fn):
            return fn()

    return _default_config(metric_name)


def _default_config(metric_name: str):
    """
    Return a default config instance for metrics that need one, or ``None``.

    Mirrors the logic in :func:`gui.core.metric_runner._prepare_config`.

    :param metric_name: Metric to look up.
    :return: A default config instance, or ``None`` if the metric has no config class.
    """
    cls = Metric.registry[metric_name]
    config_module_name = f"{cls.__module__}_config"
    try:
        mod = importlib.import_module(config_module_name)
        config_class = getattr(mod, f"{metric_name}_config", None)
        if config_class is not None:
            return config_class()
    except (ImportError, TypeError):
        pass
    return None


def _result_to_dict(r, experiment_tag: str, dataset_name: str) -> dict:
    """
    Serialize a :class:`DQResult` to a dict with explicit experiment/dataset annotations.

    :param r: The DQResult.
    :param experiment_tag: Experiment tag annotation.
    :param dataset_name: Dataset annotation.
    :return: A JSON-serializable dict.
    """
    value = r.DQvalue
    if value is not None and isinstance(value, float) and math.isnan(value):
        value = None
    return {
        "timestamp": r.timestamp.isoformat() if r.timestamp is not None else None,
        "DQdimension": str(r.DQdimension),
        "DQmetric": r.DQmetric,
        "DQgranularity": str(r.DQgranularity),
        "DQvalue": value,
        "DQexplanation": r.DQexplanation,
        "runtime": r.runtime,
        "tableName": r.tableName,
        "columnNames": r.columnNames,
        "rowIndex": r.rowIndex,
        "configJson": r.configJson,
        "experimentTag": experiment_tag,
        "dataset": dataset_name,
    }


def _hash_csv(path: Path) -> str:
    """Hash a CSV file for change-detection (truncated SHA-256)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _hash_metric_source(metric_name: str) -> str:
    """Hash a metric's source code for change-detection (truncated SHA-256)."""
    cls = Metric.registry[metric_name]
    try:
        src = inspect.getsource(cls)
    except OSError:
        src = metric_name
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _json_default(obj):
    """JSON ``default=`` hook: serialize datetime-like values via their ``isoformat``."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


if __name__ == "__main__":
    main()
