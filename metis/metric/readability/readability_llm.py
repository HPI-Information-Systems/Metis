# metis/metric/readability/readability_llm.py
from __future__ import annotations

import json
import os
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from metis.metric.metric import Metric
from metis.utils.result import DQResult

from .tokenization import split_identifier, split_text, compute_case_consistency_scores
from .llm_backend import HFTransformersBackend, LLMBackend
from .scorers import (
    load_abbreviations,
    WordNetScorer,
    WordNetOnlyAdapter,
    HybridScorer,
    schema_label_score,
    content_cell_score,
)

# ---------------- Config (moved here; former config.py) ----------------

@dataclass
class LLMTriggerConfig:
    wordnet_unknown_only: bool = True
    also_if_contains_digit_or_symbol: bool = True

@dataclass
class ReadabilityLLMConfig:
    # Core
    sample_size: Optional[int] = None
    random_seed: int = 13
    min_token_length: int = 2
    abbr_csv: Optional[str] = None
    ignore_numeric_columns: bool = True

    compute_schema: bool = True
    # cell output (optional)
    output_cells: bool = False
    output_columns: bool = True
    output_table: bool = True

    # HF LLM
    use_llm_fallback: bool = True
    hf_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    hf_device: str = "auto"
    hf_dtype: str = "auto"
    hf_max_new_tokens: int = 512

    # Mode
    llm_mode: str = "fallback"  # Ziel-2 default

    # Shared LLM params
    llm_batch_size: int = 80
    llm_trigger: LLMTriggerConfig = field(default_factory=LLMTriggerConfig)

    # Optional top-down column scoring
    column_level_llm_score: bool = False
    column_level_llm_sample_values: int = 100
    column_level_llm_gamma: float = 0.5

    @staticmethod
    def from_metric_config(metric_config: Optional[str]) -> "ReadabilityLLMConfig":
        cfg = ReadabilityLLMConfig()
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
            llm = data.get("llm", {})
            if isinstance(common, dict) and isinstance(llm, dict):
                merged = dict(common)
                merged.update(llm)
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

        cfg.llm_mode = str(data.get("llm_mode", cfg.llm_mode)).strip().lower()
        if cfg.llm_mode not in ("strict", "fallback"):
            cfg.llm_mode = "fallback"

        cfg.use_llm_fallback = bool(data.get("use_llm_fallback", cfg.use_llm_fallback))
        cfg.hf_model_id = str(data.get("hf_model_id", cfg.hf_model_id))
        cfg.hf_device = str(data.get("hf_device", cfg.hf_device))
        cfg.hf_dtype = str(data.get("hf_dtype", cfg.hf_dtype))
        cfg.hf_max_new_tokens = int(data.get("hf_max_new_tokens", cfg.hf_max_new_tokens))

        cfg.llm_batch_size = int(data.get("llm_batch_size", cfg.llm_batch_size))

        trig = data.get("llm_trigger", None)
        if isinstance(trig, dict):
            cfg.llm_trigger = LLMTriggerConfig(
                wordnet_unknown_only=bool(trig.get("wordnet_unknown_only", cfg.llm_trigger.wordnet_unknown_only)),
                also_if_contains_digit_or_symbol=bool(trig.get("also_if_contains_digit_or_symbol", cfg.llm_trigger.also_if_contains_digit_or_symbol)),
            )

        cfg.column_level_llm_score = bool(data.get("column_level_llm_score", cfg.column_level_llm_score))
        cfg.column_level_llm_sample_values = int(data.get("column_level_llm_sample_values", cfg.column_level_llm_sample_values))
        cfg.column_level_llm_gamma = float(data.get("column_level_llm_gamma", cfg.column_level_llm_gamma))
        cfg.column_level_llm_gamma = max(0.0, min(1.0, cfg.column_level_llm_gamma))
        return cfg

# ---------------- Helpers ----------------

_BACKEND_CACHE: Dict[tuple, LLMBackend] = {}

def _build_backend(cfg: ReadabilityLLMConfig) -> Optional[LLMBackend]:
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

class ReadabilityLLM(Metric):
    """Hybrid readability metric: WordNet-first with LLM fallback (lazy backend loading)."""

    def assess(
        self,
        data: pd.DataFrame,
        reference: Union[pd.DataFrame, None] = None,
        metric_config: Union[str, None] = None,
    ) -> List[DQResult]:
        cfg = ReadabilityLLMConfig.from_metric_config(metric_config)
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

            for label in labels:
                toks = [t for t in split_identifier(label) if len(t) >= cfg.min_token_length]
                s_case = float(case_scores.get(label, 1.0))
                schema_label_scores_wordnet[label] = schema_label_score(toks, s_case, baseline)
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
                schema_label_scores_hybrid[label] = schema_label_score(toks, s_case, hybrid)
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
                            mesTime=pd.Timestamp.now(),
                            DQvalue=float(z_hybrid),
                            DQdimension="Readability",
                            DQmetric="readability_llm",
                            columnNames=[col],
                            rowIndex=row_pos,  # ✅ stable int position (never crashes)
                            DQgranularity="cell",
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
            s_topdown = None
            if cfg.column_level_llm_score and cfg.use_llm_fallback:
                if hybrid.backend is None:
                    hybrid.backend = _build_backend(cfg)
                if hybrid.backend is not None:
                    values = [str(x) for x in series.unique().tolist() if str(x).strip() != ""]
                    values.sort()
                    k = max(1, int(cfg.column_level_llm_sample_values))
                    if len(values) > k:
                        idxs = rng.sample(range(len(values)), k)
                        sample_vals = [values[i] for i in sorted(idxs)]
                    else:
                        sample_vals = values
                    s_topdown = float(hybrid.backend.score_column(col, sample_vals))

            if cfg.column_level_llm_score and s_topdown is not None:
                gamma = cfg.column_level_llm_gamma
                s_combined = gamma * s_bottomup + (1.0 - gamma) * s_topdown
            else:
                s_combined = s_bottomup

            col_wordnet[col] = s_wordnet
            col_combined[col] = float(s_combined)

            col_ann[col] = {
                "content_readability_wordnet_only": float(s_wordnet),
                "content_readability_bottom_up": float(s_bottomup),
                "content_readability_top_down": (float(s_topdown) if s_topdown is not None else None),
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
                "column_level_llm_score_enabled": bool(cfg.column_level_llm_score),
                "column_level_llm_gamma": float(cfg.column_level_llm_gamma),
            }

        content_wordnet = float(sum(col_wordnet.values()) / len(col_wordnet)) if col_wordnet else 0.0
        content_hybrid = float(sum(col_combined.values()) / len(col_combined)) if col_combined else 0.0
        content_uplift = float(content_hybrid - content_wordnet)

        llm_tokens_share_total = float(total_llm_tokens_used / total_unique_tokens_seen) if total_unique_tokens_seen else 0.0

        now = pd.Timestamp.now()
        results: List[DQResult] = []
        if cfg.output_cells:
            results.extend(cell_results)

        if cfg.output_table:
            results.append(
                DQResult(
                    mesTime=now,
                    DQvalue=float(content_hybrid),
                    DQdimension="Readability",
                    DQmetric="readability_llm",
                    columnNames=None,
                    rowIndex=None,
                    DQgranularity="table",
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
                        "column_level_llm_score_enabled": bool(cfg.column_level_llm_score),
                        "column_level_llm_sample_values": int(cfg.column_level_llm_sample_values),
                        "column_level_llm_gamma": float(cfg.column_level_llm_gamma),
                    },
                    dataset=None,
                    tableName=None,
                )
            )

        if cfg.compute_schema:
            results.append(
                DQResult(
                    mesTime=now,
                    DQvalue=float(schema_hybrid),
                    DQdimension="Readability",
                    DQmetric="readability_llm",
                    columnNames=None,
                    rowIndex=None,
                    DQgranularity="schema",
                    DQexplanation={
                        "schema_readability": float(schema_hybrid),
                        "schema_readability_wordnet_only": float(schema_wordnet),
                        "llm_uplift_schema": float(schema_hybrid - schema_wordnet),
                        "llm_mode": llm_mode,
                        "use_llm_fallback": bool(cfg.use_llm_fallback),
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
                        DQvalue=float(col_combined.get(col, 0.0)),
                        DQdimension="Readability",
                        DQmetric="readability_llm",
                        columnNames=[col],
                        rowIndex=None,
                        DQgranularity="column",
                        DQexplanation=col_ann.get(col, {}),
                        dataset=None,
                        tableName=None,
                    )
                )
        return results


class readability_llm(ReadabilityLLM):
    """snake_case alias for METIS registry."""
    pass
