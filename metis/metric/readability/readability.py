# metis/metric/readability/readability.py
import random
from collections import Counter
from typing import Dict, List, Optional, Union, Any

import pandas as pd
from sympy import series
from metis.metric.metric import Metric
from metis.utils.result import DQResult

from .config import ReadabilityConfig
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

# Cache for LLM backends to avoid redundant loading.
_BACKEND_CACHE = {}

# Helper functions for readability metric, optionally building LLM backend,locally, configurable. (LLM-Backend)
def _build_backend(cfg: ReadabilityConfig) -> Optional[LLMBackend]:
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

# Helper to select text columns, optionally ignoring numeric columns if the dataframe is mixed-type.

def _select_text_columns(df: pd.DataFrame, ignore_numeric: bool) -> List[str]:
    if not ignore_numeric:
        return [str(c) for c in df.columns]
    cols = []
    for c in df.columns:
        dt = str(df[c].dtype)
        if dt == "object" or dt.startswith("string"):
            cols.append(str(c))
    return cols


# Sampling helper, for large datasets to control runtime and cost, especially when using LLMs.

def _sample_df(df: pd.DataFrame, sample_size: Optional[int], rng: random.Random) -> pd.DataFrame:
    if sample_size is None or len(df) <= sample_size:
        return df
    idx = list(df.index)
    sampled_idx = rng.sample(idx, int(sample_size))
    return df.loc[sampled_idx]



class Readability(Metric):
    """Thin Metis-style wrapper around readability scoring components."""
    # Main assessment function that computes readability scores for schema and content levels, using WordNet and LLM fallback as configured. Deterministic reproducibility (sampling + LLM subsampling for top-down).
    
    def assess(
        self,
        data: pd.DataFrame,
        reference: Union[pd.DataFrame, None] = None,
        metric_config: Union[str, None] = None,
    ) -> List[DQResult]:
        cfg = ReadabilityConfig.from_metric_config(metric_config)
        rng = random.Random(cfg.random_seed)

        # If you want to use sampling and text column selection, uncomment these lines:
        text_cols = _select_text_columns(data, cfg.ignore_numeric_columns)
        df = _sample_df(data, cfg.sample_size, rng)
        

        # (You currently evaluate all columns + full data)
        # text_cols = [str(c) for c in data.columns]
        # df = data

        # Initialize scorers
        abbreviations = load_abbreviations(cfg.abbr_csv)
        wordnet = WordNetScorer(abbreviations=abbreviations)

        # Build scorers. The token formulas are located in scorers.py.
        baseline = WordNetOnlyAdapter(wordnet)  # (2019 Ehrlinger) baseline idea
        hybrid = HybridScorer(cfg, wordnet=wordnet, backend=None)

        # used for debugging & DQ4AI comparability
        llm_mode = str(getattr(cfg, "llm_mode", "strict")).lower() 
        total_llm_tokens_used = 0  
        total_unique_tokens_seen = 0

        # -------------------------
        # A) SCHEMA (separate result)
        # -------------------------
        schema_wordnet = 0.0
        schema_hybrid = 0.0
        schema_label_scores_wordnet: Dict[str, float] = {}
        schema_label_scores_hybrid: Dict[str, float] = {}

        # Label tokenization + case consistency, formula is in tokenization.py (only used here).
        if cfg.compute_schema:
            labels = [str(c) for c in data.columns]
            case_scores = compute_case_consistency_scores(labels)

            # WordNet-only schema
            for label in labels:
                toks = [t for t in split_identifier(label) if len(t) >= cfg.min_token_length]
                s_case = float(case_scores.get(label, 1.0))
                schema_label_scores_wordnet[label] = schema_label_score(toks, s_case, baseline)

            schema_wordnet = float(sum(schema_label_scores_wordnet.values()) / len(schema_label_scores_wordnet)) if schema_label_scores_wordnet else 0.0

            # the token set depends on llm_mode:
            # - strict: all tokens are LLM-scored (needs_llm() returns True for all non-empty tokens)
            # - fallback: only WordNet-unknown / triggers are LLM-scored
            schema_llm_tokens = set() 
            for label in labels:
                toks = [t for t in split_identifier(label) if len(t) >= cfg.min_token_length]
                for t in toks:
                    hybrid.score_fast(t)
                    if hybrid.needs_llm(t):
                        schema_llm_tokens.add(t)

            # LLM batch call for unknown tokens
            # Lazy LLM: only build + call if tokens exist
            if cfg.use_llm_fallback and schema_llm_tokens:
                if hybrid.backend is None:
                    hybrid.backend = _build_backend(cfg)
                if hybrid.backend is not None:
                    hybrid.score_llm_batch(sorted(schema_llm_tokens))

            # Hybrid schema scoring
            for label in labels:
                toks = [t for t in split_identifier(label) if len(t) >= cfg.min_token_length]
                s_case = float(case_scores.get(label, 1.0))
                schema_label_scores_hybrid[label] = schema_label_score(toks, s_case, hybrid)

            schema_hybrid = (
                float(sum(schema_label_scores_hybrid.values()) / len(schema_label_scores_hybrid))
                if schema_label_scores_hybrid else 0.0
            )

        # -------------------------
        # B) CONTENT (primary result)
        # -------------------------
        col_ann: Dict[str, Dict[str, Any]] = {}
        col_wordnet: Dict[str, float] = {}
        col_bottomup: Dict[str, float] = {}
        col_combined: Dict[str, float] = {}

        #Loop over columns, defines the set of non-null cells per column 𝑘
        for col in text_cols:
            series = df[col].dropna()
            if series.empty:
                col_wordnet[col] = 0.0
                col_bottomup[col] = 0.0
                col_combined[col] = 0.0
                col_ann[col] = {"content_readability_wordnet_only": 0.0}
                continue
            if col == "country":
                germany_rows = series.astype(str)[series.astype(str).str.contains("Germany", na=False)]
                           

            if len(germany_rows) > 0:
                sample = germany_rows.iloc[-1]  # nimmt "Germany xyzqweasd"
                
                toks = [t for t in split_text(sample) if len(t) >= cfg.min_token_length]
                for t in toks:
                    hybrid.score_fast(t)
                    

            ## Pre-scan tokens to decide which need LLM (WordNet-first)
            uniq_tokens = set()
            llm_tokens = set()  # clearer name than "unknown_tokens"

            # Pre-scan tokens to decide which need LLM
            for v in series:
                toks = [t for t in split_text(v) if len(t) >= cfg.min_token_length]
                for t in toks:
                    uniq_tokens.add(t)
                    hybrid.score_fast(t)
                    if hybrid.needs_llm(t):
                        llm_tokens.add(t)

            # Lazy LLM: build + call only if needed
            if cfg.use_llm_fallback and llm_tokens:
                if hybrid.backend is None:
                    hybrid.backend = _build_backend(cfg)
                if hybrid.backend is not None:
                    hybrid.score_llm_batch(sorted(llm_tokens))

            # track global usage
            total_unique_tokens_seen += len(uniq_tokens)      
            total_llm_tokens_used += len(llm_tokens)    

            cell_scores_wordnet = []
            cell_scores_hybrid = []
            unknown_counter = Counter()
            difficult_counter = Counter()

            # Bottom-up: cell-level aggregation 
            for v in series:
                toks = [t for t in split_text(v) if len(t) >= cfg.min_token_length]
                if not toks:
                    continue
                cell_scores_wordnet.append(content_cell_score(toks, baseline, None, None))
                cell_scores_hybrid.append(content_cell_score(toks, hybrid, unknown_counter, difficult_counter))

            s_wordnet = float(sum(cell_scores_wordnet) / len(cell_scores_wordnet)) if cell_scores_wordnet else 0.0
            s_bottomup = float(sum(cell_scores_hybrid) / len(cell_scores_hybrid)) if cell_scores_hybrid else 0.0

            # Top-down (optional): whole-column judge
            s_topdown = None
            if cfg.column_level_llm_score:
                # Only attempt top-down if LLM is allowed
                if cfg.use_llm_fallback:
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
                        # backend API lives on the backend (not on hybrid)
                        s_topdown = float(hybrid.backend.score_column(col, sample_vals))

            # Combine BU + TD (content-only) (F5)
            if cfg.column_level_llm_score and s_topdown is not None:
                gamma = cfg.column_level_llm_gamma
                s_combined = gamma * s_bottomup + (1.0 - gamma) * s_topdown
            else:
                s_combined = s_bottomup

            col_wordnet[col] = s_wordnet
            col_bottomup[col] = s_bottomup
            col_combined[col] = float(s_combined)

            # Column annotations (schema values are reported but NOT mixed into content score)
            col_ann[col] = {
                "content_readability_wordnet_only": float(s_wordnet),
                "content_readability_bottom_up": float(s_bottomup),
                "content_readability_top_down": (float(s_topdown) if s_topdown is not None else None),
                "content_readability_combined": float(s_combined),

                # optional diagnostics
                "top_unknown_words": [w for w, _ in unknown_counter.most_common(10)],
                "top_difficult_words": [w for w, _ in difficult_counter.most_common(10)],
                
                # make semantics match strict-mode as well
                "llm_tokens_count": int(len(llm_tokens)),             
                "unique_tokens_count": int(len(uniq_tokens)),
                "llm_mode": llm_mode,                                
                "llm_tokens_share": float(len(llm_tokens) / len(uniq_tokens)) if len(uniq_tokens) else 0.0, 

                # schema reported separately; keep as context only
                "schema_readability_column_name_wordnet_only": float(schema_label_scores_wordnet.get(col, 0.0)) if cfg.compute_schema else None,
                "schema_readability_column_name_hybrid": float(schema_label_scores_hybrid.get(col, 0.0)) if cfg.compute_schema else None,

                "use_llm_fallback": bool(cfg.use_llm_fallback),
                "hf_model_id": cfg.hf_model_id if cfg.use_llm_fallback else None,
                "hf_device": cfg.hf_device if cfg.use_llm_fallback else None,
                "hf_dtype": cfg.hf_dtype if cfg.use_llm_fallback else None,
                "column_level_llm_score_enabled": bool(cfg.column_level_llm_score),
                "column_level_llm_gamma": float(cfg.column_level_llm_gamma),
            }

        # Table-level CONTENT scores (F6)
        content_wordnet = float(sum(col_wordnet.values()) / len(col_wordnet)) if col_wordnet else 0.0
        content_hybrid = float(sum(col_combined.values()) / len(col_combined)) if col_combined else 0.0
        content_uplift = float(content_hybrid - content_wordnet)

        # overall usage diagnostics
        llm_tokens_share_total = float(total_llm_tokens_used / total_unique_tokens_seen) if total_unique_tokens_seen else 0.0  

        now = pd.Timestamp.now()
        results: List[DQResult] = []

        # (1) Primary: content result
        results.append(
            DQResult(
                mesTime=now,
                DQvalue=float(content_hybrid),
                DQdimension="Readability",
                DQmetric="readability_content_wordnetFirst_llmFallback",
                columnNames=None,
                rowIndex=None,
                DQgranularity="table",
                DQexplanation={
                    # content-only reporting
                    "content_readability": float(content_hybrid),
                    "content_readability_wordnet_only": float(content_wordnet),
                    "llm_uplift_content": float(content_uplift),

                    # DQ4AI comparability diagnostics
                    "llm_mode": llm_mode,  
                    "llm_tokens_count_total": int(total_llm_tokens_used), 
                    "unique_tokens_count_total": int(total_unique_tokens_seen),  
                    "llm_tokens_share_total": float(llm_tokens_share_total),  

                    # config/context
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

                    # provenance (paper mapping)
                    "paper_baseline": "2019_DQ-metric-readability (WordNet/lexical baseline)",
                    "paper_llm": "2025-02_DQ4AI-report-Automatic-readability-assessment (LLM assessment/fallback)",
                },
                dataset=None,
                tableName=None,
            )
        )

        # (2) Secondary: schema result (only if enabled)
        if cfg.compute_schema:
            schema_uplift = float(schema_hybrid - schema_wordnet)
            results.append(
                DQResult(
                    mesTime=now,
                    DQvalue=float(schema_hybrid),
                    DQdimension="Readability",
                    DQmetric="readability_schema_wordnetFirst_llmFallback",
                    columnNames=None,
                    rowIndex=None,
                    DQgranularity="table",
                    DQexplanation={
                        "schema_readability": float(schema_hybrid),
                        "schema_readability_wordnet_only": float(schema_wordnet),
                        "llm_uplift_schema": float(schema_uplift),

                        # show which mode produced schema scoring
                        "llm_mode": llm_mode,

                        "random_seed": cfg.random_seed,
                        "min_token_length": cfg.min_token_length,
                        "use_llm_fallback": bool(cfg.use_llm_fallback),
                        "hf_model_id": cfg.hf_model_id if cfg.use_llm_fallback else None,

                        "paper_baseline": "2019_DQ-metric-readability (schema label readability)",
                        "paper_llm": "2025-02 DQ4AI report (LLM fallback)",
                    },
                    dataset=None,
                    tableName=None,
                )
            )

        # Column results (content primary)
        for col in text_cols:
            results.append(
                DQResult(
                    mesTime=now,
                    DQvalue=float(col_combined.get(col, 0.0)),
                    DQdimension="Readability",
                    DQmetric="readability_content_wordnetFirst_llmFallback_column",
                    columnNames=[col],
                    rowIndex=None,
                    DQexplanation=col_ann.get(col, {}),
                    dataset=None,
                    tableName=None,
                    DQgranularity="column",
                )
            )

        return results

class readability_content(Readability):
    """
    Alias class to expose the metric under the framework-mandated
    snake_case naming convention.
    """
    pass
