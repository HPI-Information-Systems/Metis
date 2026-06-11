"""Restaurant demo configurations.

Used by:

- ``metrics_page.render(demo_mode=True)`` — read-only config display
- ``run_demo_pipeline.py`` — precompute results with proper configs
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd

_DATA_DIR: Path = Path(__file__).resolve().parents[2] / "data"

# Ordered list of metrics that have precomputed results in restaurant_results.json.
# Metrics absent from this list are shown as disabled in the demo.
DEMO_METRICS: list[str] = [
    "completeness_nullRatio",
    "minimality_duplicateCount",
    "validity_outOfVocabulary",
    "consistency_countFDViolations",
    "consistency_ruleBasedHinrichs",
    "consistency_ruleBasedPipino",
    "timeliness_heinrich",
]

# FD config — used as the consistency_countFDViolations config dict.
FD_CONFIG: dict[str, list[str]] = {
    "phone":   ["city", "type"],
    "address": ["city"],
    "name":    ["phone"],
}


def get_nullRatio_config():
    """
    Return the demo config for ``completeness_nullRatio``.

    Per-column completeness — per-cell would be a binary indicator that
    duplicates the input dataframe.

    :return: A ``completeness_nullRatio_config`` instance.
    """
    from metis.metric.completeness.completeness_nullRatio_config import (
        completeness_nullRatio_config,
    )
    return completeness_nullRatio_config(aggregation_axis="index")


def get_hinrichs_config():
    """
    Return the demo config for ``consistency_ruleBasedHinrichs`` loaded from the data/ rule file.

    :return: A ``consistency_ruleBasedHinrichs_config`` instance.
    """
    mod = _load_rule_module("restaurants_consistency_ruleBasedHinrichs")
    from metis.metric.consistency.consistency_ruleBasedHinrichs_config import (
        consistency_ruleBasedHinrichs_config,
    )
    return consistency_ruleBasedHinrichs_config(
        attribute_rules=mod.attribute_rules,
        tuple_rules=mod.tuple_rules,
    )


def get_pipino_config():
    """
    Return the demo config for ``consistency_ruleBasedPipino`` loaded from the data/ rule file.

    :return: A ``consistency_ruleBasedPipino_config`` instance.
    """
    mod = _load_rule_module("restaurants_consistency_ruleBasedPipino")
    from metis.metric.consistency.consistency_ruleBasedPipino_config import (
        consistency_ruleBasedPipino_config,
    )
    return consistency_ruleBasedPipino_config(
        attribute_rules=mod.attribute_rules,
        tuple_rules=mod.tuple_rules,
    )


def get_timeliness_config():
    """
    Return the demo config for ``timeliness_heinrich`` over the relevant restaurant columns.

    :return: A ``timeliness_heinrich_config`` instance.
    """
    from metis.metric.timeliness.timeliness_heinrich_config import (
        timeliness_heinrich_column_config,
        timeliness_heinrich_config,
    )
    date_kwargs = {"errors": "coerce"}
    return timeliness_heinrich_config(
        timeliness_config_per_column={
            "avg_rating": timeliness_heinrich_column_config(
                decline_rate=1.0,
                ingestion_date_column="last_review_date",
                to_datetime_kwargs=date_kwargs,
                simulated_timestamp_precision="day",
            ),
            "total_reviews_count": timeliness_heinrich_column_config(
                decline_rate=0.5,
                ingestion_date_column="last_review_date",
                to_datetime_kwargs=date_kwargs,
                simulated_timestamp_precision="day",
            ),
        }
    )


# Human-readable config display data for read-only rendering in metrics_page.
DEMO_CONFIG_DISPLAY: dict[str, dict] = {
    "consistency_countFDViolations": {
        "type": "fd",
        "rules": FD_CONFIG,
    },
    "consistency_ruleBasedHinrichs": {
        "type": "callable",
        "source_file": str(_DATA_DIR / "restaurants_consistency_ruleBasedHinrichs.py"),
        "description": "Attribute rules grade phone-format, type-vocabulary, rating-range and review-count violations. Tuple rule penalises rows where last_review_date precedes first_review_date.",
    },
    "consistency_ruleBasedPipino": {
        "type": "callable",
        "source_file": str(_DATA_DIR / "restaurants_consistency_ruleBasedPipino.py"),
        "description": "Attribute rules check phone format, type vocabulary, rating range, and non-negative review counts. Tuple rules check date ordering and that rows with reviews carry a last_review_date.",
    },
    "timeliness_heinrich": {
        "type": "timeliness",
        "columns": {
            "avg_rating":          {"decline_rate": 1.0, "ingestion_date_column": "last_review_date", "precision": "day"},
            "total_reviews_count": {"decline_rate": 0.5, "ingestion_date_column": "last_review_date", "precision": "day"},
        },
    },
}


def preprocess_heinrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows ``timeliness_heinrich`` cannot handle: NaN ``last_review_date``
    crashes the certainty computation, and future-dated values make
    ``exp(-decline_rate * age)`` exceed 1 because the age turns negative.

    :param df: The raw restaurant dataframe.
    :return: A filtered dataframe with valid, non-future timestamp values.
    """
    df = df.dropna(subset=["last_review_date"])
    dates = pd.to_datetime(df["last_review_date"], errors="coerce")
    return df[dates.notna() & (dates <= pd.Timestamp.now())]


# Aliases used by run_demo_pipeline._get_metric_config() — the pipeline strips
# the dimension prefix (e.g. "consistency_") to get the suffix and then calls
# get_{suffix}_config().
get_ruleBasedHinrichs_config = get_hinrichs_config
get_ruleBasedPipino_config = get_pipino_config
get_heinrich_config = get_timeliness_config


def _load_rule_module(module_stem: str) -> ModuleType:
    """
    Dynamically load a rule file from the ``data/`` directory by stem name.

    :param module_stem: The module's filename stem.
    :return: The loaded module.
    """
    path = _DATA_DIR / f"{module_stem}.py"
    spec = importlib.util.spec_from_file_location(module_stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
