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

from metis.utils.readability.tokenization import (split_identifier, split_text, compute_case_consistency_scores)
from metis.utils.readability.scorers import (load_abbreviations, WordNetScorer, WordNetOnlyAdapter, schema_label_score, content_cell_score)

@dataclass
class ReadabilityWordNetConfig:
    # Core
    sample_size: Optional[int] = None
    random_seed: int = 13
    min_token_length: int = 2
    abbr_csv: Optional[str] = None
    ignore_numeric_columns: bool = True

    # Schema (separate)
    compute_schema: bool = True
    # Output toggles
    output_cells: bool = False
    output_columns: bool = True
    output_table: bool = True

    @staticmethod
    def from_metric_config(metric_config: Optional[str]) -> "ReadabilityWordNetConfig":
        cfg = ReadabilityWordNetConfig()
        if metric_config is None:
            return cfg

        metric_config = metric_config.strip()
        if metric_config.startswith("{"):
            data = json.loads(metric_config)
        else:
            if not os.path.exists(metric_config):
                raise ValueError(f"metric_config is neither JSON nor an existing path: {metric_config}")
            with open(metric_config, "r", encoding="utf-8") as f:
                data = json.load(f)
        if isinstance(data, dict) and ("common" in data or "wordnet" in data or "llm" in data):
            common = data.get("common", {})
            wordnet = data.get("wordnet", {})
            if isinstance(common, dict) and isinstance(wordnet, dict):
                merged = dict(common)
                merged.update(wordnet)
                data = merged

        cfg.sample_size = data.get("sample_size", cfg.sample_size)
        cfg.random_seed = int(data.get("random_seed", cfg.random_seed))
        cfg.min_token_length = int(data.get("min_token_length", cfg.min_token_length))
        cfg.abbr_csv = data.get("abbr_csv", cfg.abbr_csv)
        cfg.ignore_numeric_columns = bool(data.get("ignore_numeric_columns", cfg.ignore_numeric_columns))
        cfg.compute_schema = bool(data.get("compute_schema", cfg.compute_schema))
        cfg.output_cells = bool(data.get("output_cells", cfg.output_cells))
        cfg.output_columns = bool(data.get("output_columns", cfg.output_columns))
        cfg.output_table = bool(data.get("output_table", cfg.output_table))
        return cfg


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


class readability_wordnet(Metric):
    """WordNet-only readability metric (no LLM / no HF dependencies)."""

    def assess(
        self,
        data: pd.DataFrame,
        reference: Union[pd.DataFrame, None] = None,
        metric_config: Union[str, None] = None,
    ) -> List[DQResult]:
        """
        Assess the readability of a tabular dataset using the WordNet-only readability metric.

        This metric evaluates the readability of schema labels and textual content
        without using LLM-based fallback or hybrid scoring. Depending on the
        configuration, it can produce readability results for schema labels, table-level
        text content, individual columns, and optionally individual cells.

        Parameters
        - data: pd.DataFrame
                The DataFrame to assess. This is the primary dataset whose schema labels
                and textual cell values are evaluated for readability.

        - reference: Optional[pd.DataFrame]
                Optional reference DataFrame. This metric does not use a reference
                dataset and accepts this parameter only to conform to the framework-wide
                metric interface.

        - metric_config: Optional[str]
                Optional path or JSON string containing readability-specific
                configuration. The configuration is parsed via
                `ReadabilityWordNetConfig.from_metric_config(...)` and controls
                sampling, schema scoring, and output granularity.

        Returns
        - List[DQResult]
                A list of readability assessment results. Depending on the configuration,
                the method may return `DQResult` objects for schema-level, table-level,
                column-level, and optional cell-level readability scores.

        Notes
        - The input DataFrame is not modified in-place.
        - This implementation uses WordNet-based scoring only and does not initialize
        or call any LLM backend.
        - The exact output granularity depends on the metric configuration.
        """
        cfg = ReadabilityWordNetConfig.from_metric_config(metric_config)
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
            schema_vocab = set()
            for lab in labels:
                schema_vocab.update(
                    t.strip().lower()
                    for t in split_identifier(lab)
                    if len(t) >= cfg.min_token_length and str(t).strip()
                )
            for label in labels:
                toks = [t for t in split_identifier(label) if len(t) >= cfg.min_token_length]
                s_case = float(case_scores.get(label, 1.0))
                schema_label_scores[label] = schema_label_score(toks, s_case, baseline, schema_vocab=schema_vocab)
            schema_wordnet = float(sum(schema_label_scores.values()) / len(schema_label_scores)) if schema_label_scores else 0.0

        # B) CONTENT
        col_scores: Dict[str, float] = {}
        col_ann: Dict[str, Dict[str, Any]] = {}
        all_cell_results: List[DQResult] = []

        for col in text_cols:
            series = df[col].dropna()
            if series.empty:
                col_scores[col] = 0.0
                col_ann[col] = {"content_readability_wordnet_only": 0.0}
                continue

            cell_scores: List[float] = []
            cell_results: List[DQResult] = []

            for row_pos, (src_idx, v) in enumerate(series.items()):
                toks = [t for t in split_text(v) if len(t) >= cfg.min_token_length]
                if not toks:
                    continue

                z = float(content_cell_score(toks, baseline, None, None))
                cell_scores.append(z)

                if cfg.output_cells:
                    cell_results.append(
                        DQResult(
                            mesTime=pd.Timestamp.now(),
                            DQdimension="Readability",
                            DQmetric="WordNet",
                            DQgranularity="cell",
                            DQvalue=z,
                            columnNames=[col],
                            rowIndex=row_pos,  # stable integer position (never crashes)
                            DQexplanation={
                                "content_readability_wordnet_only": z,
                                "use_llm_fallback": False,
                                "source_row_index": (None if pd.isna(src_idx) else str(src_idx)),
                            },
                            dataset=None,
                            tableName=None,
                        )
                    )                 
            if cfg.output_cells:
                all_cell_results.extend(cell_results)

            s = float(sum(cell_scores) / len(cell_scores)) if cell_scores else 0.0
            col_scores[col] = s
            col_ann[col] = {
                "content_readability_wordnet_only": float(s),
                "use_llm_fallback": False,
                "schema_readability_column_name_wordnet_only": float(schema_label_scores.get(col, 0.0)) if cfg.compute_schema else None,
            }

        content_wordnet = float(sum(col_scores.values()) / len(col_scores)) if col_scores else 0.0

        now = pd.Timestamp.now()
        results: List[DQResult] = []
        if cfg.output_cells:
            results.extend(all_cell_results)

        if cfg.output_table:
            results.append(
                DQResult(
                    mesTime=now,
                    DQvalue=float(content_wordnet),
                    DQdimension="Readability",
                    DQmetric="WordNet",
                    columnNames=None,
                    rowIndex=None,
                    DQgranularity="table",
                    DQexplanation={
                        "content_readability_wordnet_only": float(content_wordnet),
                        "use_llm_fallback": False},
                    dataset=None,
                    tableName=None
                )
            )

        if cfg.compute_schema:
            results.append(
                DQResult(
                    mesTime=now,
                    DQvalue=float(schema_wordnet),
                    DQdimension="Readability",
                    DQmetric="WordNet",
                    columnNames=None,
                    rowIndex=None,
                    DQgranularity="schema",
                    DQexplanation={
                        "schema_readability_wordnet_only": float(schema_wordnet),
                        "use_llm_fallback": False,
                    },
                    dataset=None,
                    tableName=None,
                )
            )

        if cfg.output_columns:
            for col in text_cols:
                results.append(
                    DQResult(
                        mesTime=now,
                        DQvalue=float(col_scores.get(col, 0.0)),
                        DQdimension="Readability",
                        DQmetric="WordNet",
                        columnNames=[col],
                        rowIndex=None,
                        DQgranularity="column",
                        DQexplanation=col_ann.get(col, {}),
                        dataset=None,
                        tableName=None,
                    )
                )

        return results