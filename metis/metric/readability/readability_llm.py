from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.metric.readability.readability_llm_config import readability_llm_config
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.readability.llm_backend import HFTransformersBackend, LLMBackend
from metis.utils.readability.scorers import (
    HybridScorer,
    WordNetOnlyAdapter,
    WordNetScorer,
    content_cell_score,
    load_abbreviations,
    schema_label_score,
)
from metis.utils.readability.tokenization import (
    compute_case_consistency_scores,
    split_identifier,
    split_text,
)
from metis.utils.result import DQResult

# ---------------- Helpers ----------------

_BACKEND_CACHE: Dict[tuple, LLMBackend] = {}

def _build_backend(cfg: readability_llm_config) -> Optional[LLMBackend]:
    if not cfg.use_llm_fallback:
        return None

    key = (cfg.hf_model_id, cfg.hf_device, cfg.hf_dtype, cfg.hf_max_new_tokens)
    if key in _BACKEND_CACHE:
        return _BACKEND_CACHE[key]

    backend = HFTransformersBackend(
        model_id=cfg.hf_model_id,
        device=cfg.hf_device,
        dtype=cfg.hf_dtype,
        max_new_tokens=cfg.hf_max_new_tokens,
    )
    _BACKEND_CACHE[key] = backend
    return backend

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

# ---------------- Metric ----------------

class readability_llm(Metric):
    """Hybrid readability metric: WordNet-first with LLM fallback (lazy backend loading)."""

    def assess(
        self,
        data: pd.DataFrame,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        Assess the readability of a tabular dataset using the hybrid readability metric.

        This metric combines WordNet-based readability scoring with optional LLM support.
        Depending on the configuration, it can produce readability results for schema
        labels, table-level text content, individual columns, and optionally individual
        cells.

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
                `ReadabilityLLMConfig.from_metric_config(...)` and controls sampling,
                schema scoring, output granularity, and LLM-related behavior.

        Returns
        - List[DQResult]
                A list of readability assessment results. Depending on the configuration,
                the method may return `DQResult` objects for schema-level, table-level,
                column-level, and optional cell-level readability scores.

        Notes
        - The input DataFrame is not modified in-place.
        - Only textual columns are considered for content readability scoring when
        numeric columns are ignored by configuration.
        - The exact output granularity depends on the metric configuration.
        """
        if metric_config is None:
            raise ValueError(
                f"Metric configuration is required for metric {readability_llm_config.__name__} but None was provided."
            )

        cfg = self.load_config(metric_config, readability_llm_config)
        rng = random.Random(cfg.random_seed)

        text_cols = _select_text_columns(data, cfg.ignore_numeric_columns)
        df = _sample_df(data, cfg.sample_size, rng)

        abbreviations = load_abbreviations(cfg.abbr_csv)
        wordnet = WordNetScorer(abbreviations=abbreviations)
        baseline = WordNetOnlyAdapter(wordnet)
        hybrid = HybridScorer(cfg, wordnet=wordnet, backend=None)

        llm_mode = str(getattr(cfg, "llm_mode", "fallback")).lower()
        total_llm_tokens_used = 0
        total_unique_tokens_seen = 0

        # A) SCHEMA
        schema_wordnet = 0.0
        schema_hybrid = 0.0
        schema_label_scores_wordnet: Dict[str, float] = {}
        schema_label_scores_hybrid: Dict[str, float] = {}

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
                schema_label_scores_wordnet[label] = schema_label_score(toks, s_case, baseline, schema_vocab=schema_vocab)
            schema_wordnet = float(sum(schema_label_scores_wordnet.values()) / len(schema_label_scores_wordnet)) if schema_label_scores_wordnet else 0.0

            schema_llm_tokens = set()
            for label in labels:
                toks = [t for t in split_identifier(label) if len(t) >= cfg.min_token_length]
                for t in toks:
                    hybrid.score_fast(t)
                    if hybrid.needs_llm(t):
                        schema_llm_tokens.add(t)

            # Lazy backend build + batch scoring only if needed
            if cfg.use_llm_fallback and schema_llm_tokens:
                if hybrid.backend is None:
                    hybrid.backend = _build_backend(cfg)
                if hybrid.backend is not None:
                    hybrid.score_llm_batch(sorted(schema_llm_tokens))

            for label in labels:
                toks = [t for t in split_identifier(label) if len(t) >= cfg.min_token_length]
                s_case = float(case_scores.get(label, 1.0))
                schema_label_scores_hybrid[label] = schema_label_score(toks, s_case, hybrid, schema_vocab=schema_vocab)
            schema_hybrid = float(sum(schema_label_scores_hybrid.values()) / len(schema_label_scores_hybrid)) if schema_label_scores_hybrid else 0.0

        # B) CONTENT
        col_wordnet: Dict[str, float] = {}
        col_combined: Dict[str, float] = {}
        col_ann: Dict[str, Dict[str, Any]] = {}
        cell_results: List[DQResult] = []

        for col in text_cols:
            series = df[col].dropna()
            if series.empty:
                col_wordnet[col] = 0.0
                col_combined[col] = 0.0
                col_ann[col] = {"content_readability_wordnet_only": 0.0}
                continue

            uniq_tokens = set()
            llm_tokens = set()

            for v in series:
                toks = [t for t in split_text(v) if len(t) >= cfg.min_token_length]
                for t in toks:
                    uniq_tokens.add(t)
                    hybrid.score_fast(t)
                    if hybrid.needs_llm(t):
                        llm_tokens.add(t)

            if cfg.use_llm_fallback and llm_tokens:
                if hybrid.backend is None:
                    hybrid.backend = _build_backend(cfg)
                if hybrid.backend is not None:
                    hybrid.score_llm_batch(sorted(llm_tokens))
                    print("LLM TOKENS:", sorted(llm_tokens))
                    for t in sorted(llm_tokens):
                        E, _, _ = wordnet.score(t)
                        print("  WordNet existence (pure):", t, E)

            total_unique_tokens_seen += len(uniq_tokens)
            total_llm_tokens_used += len(llm_tokens)

            cell_scores_wordnet = []
            cell_scores_hybrid = []
            unknown_counter = Counter()
            difficult_counter = Counter()

            for row_pos, (src_idx, v) in enumerate(series.items()):
                toks = [t for t in split_text(v) if len(t) >= cfg.min_token_length]
                if not toks:
                    continue

                z_wordnet = float(content_cell_score(toks, baseline, None, None))
                z_hybrid = float(content_cell_score(toks, hybrid, unknown_counter, difficult_counter))

                cell_scores_wordnet.append(z_wordnet)
                cell_scores_hybrid.append(z_hybrid)

                if cfg.output_cells:
                    cell_results.append(
                        DQResult(
                            timestamp=pd.Timestamp.now(),
                            DQvalue=float(z_hybrid),
                            DQdimension=DQDimension.READABILITY,
                            DQmetric="LLM",
                            columnNames=[col],
                            rowIndex=row_pos,  # ✅ stable int position (never crashes)
                            DQgranularity=DQGranularity.CELL,
                            DQexplanation={
                                "content_readability_wordnet_only": float(z_wordnet),
                                "content_readability": float(z_hybrid),
                                "llm_mode": llm_mode,
                                "use_llm_fallback": bool(cfg.use_llm_fallback),
                                "source_row_index": (None if pd.isna(src_idx) else str(src_idx)),
                            },
                            dataset=None,
                            tableName=None,
                        )
                    )

            s_wordnet = float(sum(cell_scores_wordnet) / len(cell_scores_wordnet)) if cell_scores_wordnet else 0.0
            s_bottomup = float(sum(cell_scores_hybrid) / len(cell_scores_hybrid)) if cell_scores_hybrid else 0.0

            # optional top-down
            s_combined = s_bottomup

            col_wordnet[col] = s_wordnet
            col_combined[col] = float(s_combined)

            col_ann[col] = {
                "content_readability_wordnet_only": float(s_wordnet),
                "content_readability_bottom_up": float(s_bottomup),
                "content_readability_combined": float(s_combined),
                "top_unknown_words": [w for w, _ in unknown_counter.most_common(10)],
                "top_difficult_words": [w for w, _ in difficult_counter.most_common(10)],
                "llm_tokens_count": int(len(llm_tokens)),
                "unique_tokens_count": int(len(uniq_tokens)),
                "llm_mode": llm_mode,
                "llm_tokens_share": float(len(llm_tokens) / len(uniq_tokens)) if len(uniq_tokens) else 0.0,
                "schema_readability_column_name_wordnet_only": float(schema_label_scores_wordnet.get(col, 0.0)) if cfg.compute_schema else None,
                "schema_readability_column_name_hybrid": float(schema_label_scores_hybrid.get(col, 0.0)) if cfg.compute_schema else None,
                "use_llm_fallback": bool(cfg.use_llm_fallback),
                "hf_model_id": cfg.hf_model_id if cfg.use_llm_fallback else None,
                "hf_device": cfg.hf_device if cfg.use_llm_fallback else None,
                "hf_dtype": cfg.hf_dtype if cfg.use_llm_fallback else None,
            }

        content_wordnet = float(sum(col_wordnet.values()) / len(col_wordnet)) if col_wordnet else 0.0
        content_hybrid = float(sum(col_combined.values()) / len(col_combined)) if col_combined else 0.0
        content_uplift = float(content_hybrid - content_wordnet)

        llm_tokens_share_total = float(total_llm_tokens_used / total_unique_tokens_seen) if total_unique_tokens_seen else 0.0

        results: List[DQResult] = []
        if cfg.output_cells:
            results.extend(cell_results)

        if cfg.output_table:
            results.append(
                DQResult(
                    timestamp=pd.Timestamp.now(),
                    DQvalue=float(content_hybrid),
                    DQdimension=DQDimension.READABILITY,
                    DQmetric="LLM",
                    columnNames=None,
                    rowIndex=None,
                    DQgranularity=DQGranularity.TABLE,
                    DQexplanation={
                        "content_readability": float(content_hybrid),
                        "content_readability_wordnet_only": float(content_wordnet),
                        "llm_uplift_content": float(content_uplift),
                        "llm_mode": llm_mode,
                        "llm_tokens_count_total": int(total_llm_tokens_used),
                        "unique_tokens_count_total": int(total_unique_tokens_seen),
                        "llm_tokens_share_total": float(llm_tokens_share_total),
                        "sample_size": cfg.sample_size,
                        "random_seed": cfg.random_seed,
                        "min_token_length": cfg.min_token_length,
                        "use_llm_fallback": bool(cfg.use_llm_fallback),
                        "hf_model_id": cfg.hf_model_id if cfg.use_llm_fallback else None,
                        "hf_device": cfg.hf_device if cfg.use_llm_fallback else None,
                        "hf_dtype": cfg.hf_dtype if cfg.use_llm_fallback else None,
                    },
                    dataset=None,
                    tableName=None,
                    configJson=cfg.to_json(),
                )
            )

        if cfg.compute_schema:
            results.append(
                DQResult(
                    timestamp=pd.Timestamp.now(),
                    DQvalue=float(schema_hybrid),
                    DQdimension=DQDimension.READABILITY,
                    DQmetric="LLM",
                    columnNames=None,
                    rowIndex=None,
                    DQgranularity=DQGranularity.SCHEMA,
                    DQexplanation={
                        "schema_readability": float(schema_hybrid),
                        "schema_readability_wordnet_only": float(schema_wordnet),
                        "llm_uplift_schema": float(schema_hybrid - schema_wordnet),
                        "llm_mode": llm_mode,
                        "use_llm_fallback": bool(cfg.use_llm_fallback),
                    },
                    dataset=None,
                    tableName=None,
                    configJson=cfg.to_json(),
                )
            )

        if cfg.output_columns:
            for col in text_cols:
                results.append(
                    DQResult(
                        timestamp=pd.Timestamp.now(),
                        DQvalue=float(col_combined.get(col, 0.0)),
                        DQdimension=DQDimension.READABILITY,
                        DQmetric="LLM",
                        columnNames=[col],
                        rowIndex=None,
                        DQgranularity=DQGranularity.COLUMN,
                        DQexplanation=col_ann.get(col, {}),
                        dataset=None,
                        tableName=None,
                        configJson=cfg.to_json(),
                    )
                )
        return results
