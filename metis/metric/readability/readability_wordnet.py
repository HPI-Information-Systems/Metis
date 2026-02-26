# metis/metric/readability/readability_wordnet.py
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from metis.metric.metric import Metric
from metis.utils.result import DQResult
from .readability_wordnet_config import readability_wordnet_config

from .tokenization import split_identifier, split_text, compute_case_consistency_scores
from .scorers import (
    load_abbreviations,
    WordNetScorer,
    WordNetOnlyAdapter,
    schema_label_score,
    content_cell_score,
)
from ...utils.dq_dimension import DQDimension


def _select_text_columns(df: pd.DataFrame, ignore_numeric: bool) -> List[str]:
    if not ignore_numeric:
        return [str(c) for c in df.columns]
    cols: List[str] = []
    for c in df.columns:
        dt = str(df[c].dtype)
        if dt == "object" or dt.startswith("string"):
            cols.append(str(c))
    return cols


def _sample_df(df: pd.DataFrame, sample_size: Optional[int], rng: random.Random) -> pd.DataFrame:
    if sample_size is None or len(df) <= sample_size:
        return df
    idx = list(df.index)
    sampled_idx = rng.sample(idx, int(sample_size))
    return df.loc[sampled_idx]


class ReadabilityWordNet(Metric):
    """WordNet-only readability metric (no LLM / no HF dependencies)."""

    def assess(
        self,
        data: pd.DataFrame,
        reference: Union[pd.DataFrame, None] = None,
        metric_config: Union[str, None] = None,
    ) -> List[DQResult]:

        if metric_config is None:
            raise ValueError(
                f"Metric configuration is required for metric {readability_wordnet_config.__name__} but None was provided."
            )

        cfg = self.load_config(metric_config, readability_wordnet_config)
        rng = random.Random(cfg.random_seed)

        text_cols = _select_text_columns(data, cfg.ignore_numeric_columns)
        df = _sample_df(data, cfg.sample_size, rng)

        abbreviations = load_abbreviations(cfg.abbr_csv)
        wordnet = WordNetScorer(abbreviations=abbreviations)
        baseline = WordNetOnlyAdapter(wordnet)

        # A) SCHEMA
        schema_wordnet = 0.0
        schema_label_scores: Dict[str, float] = {}

        if cfg.compute_schema:
            labels = [str(c) for c in data.columns]
            case_scores = compute_case_consistency_scores(labels)
            for label in labels:
                toks = [t for t in split_identifier(label) if len(t) >= cfg.min_token_length]
                s_case = float(case_scores.get(label, 1.0))
                schema_label_scores[label] = schema_label_score(toks, s_case, baseline)
            schema_wordnet = float(sum(schema_label_scores.values()) / len(schema_label_scores)) if schema_label_scores else 0.0

        # B) CONTENT
        col_scores: Dict[str, float] = {}
        col_ann: Dict[str, Dict[str, Any]] = {}

        for col in text_cols:
            series = df[col].dropna()
            if series.empty:
                col_scores[col] = 0.0
                col_ann[col] = {"content_readability_wordnet_only": 0.0}
                continue

            cell_scores = []
            for v in series:
                toks = [t for t in split_text(v) if len(t) >= cfg.min_token_length]
                if not toks:
                    continue
                cell_scores.append(content_cell_score(toks, baseline, None, None))

            s = float(sum(cell_scores) / len(cell_scores)) if cell_scores else 0.0
            col_scores[col] = s
            col_ann[col] = {
                "content_readability_wordnet_only": float(s),
                "use_llm_fallback": False,
                "llm_mode": "none",
                "llm_tokens_count": 0,
                "unique_tokens_count": 0,
                "llm_tokens_share": 0.0,
                "schema_readability_column_name_wordnet_only": float(schema_label_scores.get(col, 0.0)) if cfg.compute_schema else None,
            }

        content_wordnet = float(sum(col_scores.values()) / len(col_scores)) if col_scores else 0.0

        now = pd.Timestamp.now()
        results: List[DQResult] = []

        results.append(
            DQResult(
                mesTime=now,
                DQvalue=float(content_wordnet),
                DQdimension=DQDimension.READABILITY,
                DQmetric="readability_wordnet_content",
                columnNames=None,
                rowIndex=None,
                DQgranularity="table",
                DQexplanation={
                    "content_readability_wordnet_only": float(content_wordnet),
                    "use_llm_fallback": False,
                    "sample_size": cfg.sample_size,
                    "random_seed": cfg.random_seed,
                    "min_token_length": cfg.min_token_length,
                },
                dataset=None,
                tableName=None,
            )
        )

        if cfg.compute_schema:
            results.append(
                DQResult(
                    mesTime=now,
                    DQvalue=float(schema_wordnet),
                    DQdimension=DQDimension.READABILITY,
                    DQmetric="readability_wordnet_schema",
                    columnNames=None,
                    rowIndex=None,
                    DQgranularity="table",
                    DQexplanation={
                        "schema_readability_wordnet_only": float(schema_wordnet),
                        "use_llm_fallback": False,
                    },
                    dataset=None,
                    tableName=None,
                )
            )

        for col in text_cols:
            results.append(
                DQResult(
                    mesTime=now,
                    DQvalue=float(col_scores.get(col, 0.0)),
                    DQdimension=DQDimension.READABILITY,
                    DQmetric="readability_wordnet_content_column",
                    columnNames=[col],
                    rowIndex=None,
                    DQgranularity="column",
                    DQexplanation=col_ann.get(col, {}),
                    dataset=None,
                    tableName=None,
                )
            )

        return results


class readability_wordnet(ReadabilityWordNet):
    """snake_case alias for METIS registry."""
    pass
