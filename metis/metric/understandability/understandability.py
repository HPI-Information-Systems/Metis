from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from metis.metric.metric import Metric
from metis.utils.understandability.embedding_context import SentenceTransformerEmbeddingBackend
from metis.utils.understandability.llm_backend import HFTransformersBackend, LLMBackend
from metis.utils.understandability.word_understandability import content_word_understandability_score
from metis.utils.result import DQResult


@dataclass
class UnderstandabilityConfig:
    min_token_length: int = 2
    pipeline_mode: str = "resource_based"
    ignore_numeric_columns: bool = True
    include_token_details: bool = True
    include_cell_scores: bool = True
    include_column_scores: bool = True
    include_row_scores: bool = True
    output_cell_results: bool = True

    hf_model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    hf_model_revision: Optional[str] = None
    hf_device: str = "auto"
    hf_dtype: str = "auto"
    hf_max_new_tokens: int = 768
    hf_context_batch_size: int = 1
    disable_llm_backend: bool = False

    embedding_enabled: bool = True
    embedding_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "auto"
    embedding_max_column_context_values: int = 50
    include_embedding_context_text: bool = False

    # Globally fixed metric parameters. They are recorded in every result and
    # must remain unchanged across datasets within one experimental evaluation.
    context_lambda: float = 0.10
    context_negative_evidence_threshold: float = -0.40
    context_positive_evidence_threshold: float = 0.60
    language_hint: Optional[str] = None
    hybrid_minimum_available_criteria: int = 2
    hybrid_severe_notation_threshold: float = 0.35
    hybrid_low_lexical_threshold: float = 0.40
    hybrid_weak_context_threshold: float = 0.55

    table_description: Optional[str] = None
    target_user_group: str = "general adult data users"
    experiment_tag: Optional[str] = None

    @staticmethod
    def from_metric_config(metric_config: Optional[str]) -> "UnderstandabilityConfig":
        cfg = UnderstandabilityConfig()
        if metric_config is None:
            return cfg
        metric_config = metric_config.strip()
        if metric_config.startswith("{"):
            data = json.loads(metric_config)
        else:
            if not os.path.exists(metric_config):
                raise ValueError(f"metric_config is neither JSON nor an existing path: {metric_config}")
            with open(metric_config, "r", encoding="utf-8") as file:
                data = json.load(file)

        cfg.min_token_length = int(data.get("min_token_length", cfg.min_token_length))
        cfg.pipeline_mode = str(data.get("pipeline_mode", cfg.pipeline_mode)).strip().lower()
        if "use_llm" in data and "pipeline_mode" not in data:
            cfg.pipeline_mode = "hybrid" if bool(data.get("use_llm")) else "resource_based"
        cfg.ignore_numeric_columns = bool(data.get("ignore_numeric_columns", cfg.ignore_numeric_columns))
        cfg.include_token_details = bool(data.get("include_token_details", cfg.include_token_details))
        cfg.include_cell_scores = bool(data.get("include_cell_scores", cfg.include_cell_scores))
        cfg.include_column_scores = bool(data.get("include_column_scores", cfg.include_column_scores))
        cfg.include_row_scores = bool(data.get("include_row_scores", cfg.include_row_scores))
        cfg.output_cell_results = bool(data.get("output_cell_results", cfg.output_cell_results))

        cfg.hf_model_id = str(data.get("hf_model_id", cfg.hf_model_id))
        model_revision = data.get("hf_model_revision", cfg.hf_model_revision)
        cfg.hf_model_revision = None if model_revision in (None, "") else str(model_revision)
        cfg.hf_device = str(data.get("hf_device", cfg.hf_device))
        cfg.hf_dtype = str(data.get("hf_dtype", cfg.hf_dtype))
        cfg.hf_max_new_tokens = int(data.get("hf_max_new_tokens", cfg.hf_max_new_tokens))
        cfg.hf_context_batch_size = int(data.get("hf_context_batch_size", cfg.hf_context_batch_size))
        cfg.disable_llm_backend = bool(data.get("disable_llm_backend", cfg.disable_llm_backend))

        cfg.embedding_enabled = bool(data.get("embedding_enabled", cfg.embedding_enabled))
        cfg.embedding_model_id = str(data.get("embedding_model_id", cfg.embedding_model_id))
        cfg.embedding_device = str(data.get("embedding_device", cfg.embedding_device))
        cfg.embedding_max_column_context_values = int(
            data.get(
                "embedding_max_column_context_values",
                data.get("embedding_max_column_context_tokens", cfg.embedding_max_column_context_values),
            )
        )
        cfg.include_embedding_context_text = bool(
            data.get("include_embedding_context_text", cfg.include_embedding_context_text)
        )

        cfg.context_lambda = float(data.get("context_lambda", cfg.context_lambda))
        cfg.context_negative_evidence_threshold = float(
            data.get("context_negative_evidence_threshold", cfg.context_negative_evidence_threshold)
        )
        cfg.context_positive_evidence_threshold = float(
            data.get("context_positive_evidence_threshold", cfg.context_positive_evidence_threshold)
        )
        language_hint = data.get("language_hint", cfg.language_hint)
        cfg.language_hint = None if language_hint in (None, "", "auto") else str(language_hint).lower()
        cfg.hybrid_minimum_available_criteria = int(
            data.get("hybrid_minimum_available_criteria", cfg.hybrid_minimum_available_criteria)
        )
        cfg.hybrid_severe_notation_threshold = float(
            data.get("hybrid_severe_notation_threshold", cfg.hybrid_severe_notation_threshold)
        )
        cfg.hybrid_low_lexical_threshold = float(
            data.get("hybrid_low_lexical_threshold", cfg.hybrid_low_lexical_threshold)
        )
        cfg.hybrid_weak_context_threshold = float(
            data.get("hybrid_weak_context_threshold", cfg.hybrid_weak_context_threshold)
        )
        cfg.table_description = data.get("table_description", cfg.table_description)
        cfg.target_user_group = str(data.get("target_user_group", cfg.target_user_group))
        cfg.experiment_tag = data.get("experiment_tag", cfg.experiment_tag)
        return cfg

    def as_dict(self) -> Dict[str, Any]:
        return {
            "min_token_length": self.min_token_length,
            "pipeline_mode": self.pipeline_mode,
            "ignore_numeric_columns": self.ignore_numeric_columns,
            "include_token_details": self.include_token_details,
            "include_cell_scores": self.include_cell_scores,
            "include_column_scores": self.include_column_scores,
            "include_row_scores": self.include_row_scores,
            "output_cell_results": self.output_cell_results,
            "hf_model_id": self.hf_model_id,
            "hf_model_revision": self.hf_model_revision,
            "hf_device": self.hf_device,
            "hf_dtype": self.hf_dtype,
            "hf_max_new_tokens": self.hf_max_new_tokens,
            "hf_context_batch_size": self.hf_context_batch_size,
            "disable_llm_backend": self.disable_llm_backend,
            "embedding_enabled": self.embedding_enabled,
            "embedding_model_id": self.embedding_model_id,
            "embedding_device": self.embedding_device,
            "embedding_max_column_context_values": self.embedding_max_column_context_values,
            "include_embedding_context_text": self.include_embedding_context_text,
            "context_lambda": self.context_lambda,
            "context_negative_evidence_threshold": self.context_negative_evidence_threshold,
            "context_positive_evidence_threshold": self.context_positive_evidence_threshold,
            "language_hint": self.language_hint,
            "hybrid_minimum_available_criteria": self.hybrid_minimum_available_criteria,
            "hybrid_severe_notation_threshold": self.hybrid_severe_notation_threshold,
            "hybrid_low_lexical_threshold": self.hybrid_low_lexical_threshold,
            "hybrid_weak_context_threshold": self.hybrid_weak_context_threshold,
            "table_description": self.table_description,
            "target_user_group": self.target_user_group,
            "experiment_tag": self.experiment_tag,
        }


_LLM_BACKEND_CACHE: Dict[Tuple[str, Optional[str], str, str, int, int], LLMBackend] = {}
_EMBEDDING_BACKEND_CACHE: Dict[Tuple[str, str], SentenceTransformerEmbeddingBackend] = {}


def _build_llm_backend(cfg: UnderstandabilityConfig) -> Tuple[Optional[LLMBackend], Optional[str]]:
    if cfg.pipeline_mode not in {"llm", "hybrid"}:
        return None, None
    if cfg.disable_llm_backend:
        return None, "LLM backend disabled by metric config."
    key = (
        cfg.hf_model_id,
        cfg.hf_model_revision,
        cfg.hf_device,
        cfg.hf_dtype,
        cfg.hf_max_new_tokens,
        cfg.hf_context_batch_size,
    )
    if key in _LLM_BACKEND_CACHE:
        return _LLM_BACKEND_CACHE[key], None
    try:
        backend = HFTransformersBackend(
            model_id=cfg.hf_model_id,
            model_revision=cfg.hf_model_revision,
            device=cfg.hf_device,
            dtype=cfg.hf_dtype,
            max_new_tokens=cfg.hf_max_new_tokens,
            context_batch_size=cfg.hf_context_batch_size,
        )
    except Exception as exc:
        return None, str(exc)
    _LLM_BACKEND_CACHE[key] = backend
    return backend, None


def _build_embedding_backend(
    cfg: UnderstandabilityConfig,
) -> Tuple[Optional[SentenceTransformerEmbeddingBackend], Optional[str]]:
    if not cfg.embedding_enabled or cfg.pipeline_mode == "llm":
        return None, None
    key = (cfg.embedding_model_id, cfg.embedding_device)
    if key in _EMBEDDING_BACKEND_CACHE:
        return _EMBEDDING_BACKEND_CACHE[key], None
    try:
        backend = SentenceTransformerEmbeddingBackend(
            model_id=cfg.embedding_model_id,
            device=cfg.embedding_device,
        )
    except Exception as exc:
        return None, str(exc)
    _EMBEDDING_BACKEND_CACHE[key] = backend
    return backend, None


def _to_int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _llm_usage_statistics(metric_result: Dict[str, Any]) -> Dict[str, Any]:
    cell_details = metric_result.get("cell_details", {}) or {}
    llm_called_cells: List[str] = []
    fallback_cells: List[str] = []
    failed_cells: List[str] = []
    context_char_count = 0

    for cell_key, detail in cell_details.items():
        if detail.get("llm_called"):
            llm_called_cells.append(str(cell_key))
            context_char_count += len(json.dumps(detail.get("row_context") or {}, ensure_ascii=False))
            context_char_count += len(json.dumps(detail.get("column_context_examples") or [], ensure_ascii=False))
            context_char_count += len(str(detail.get("cell_value") or ""))
        if detail.get("fallback_triggered"):
            fallback_cells.append(str(cell_key))
        if detail.get("llm_called") and str(detail.get("score_source", "")).endswith("missing"):
            failed_cells.append(str(cell_key))

    approx_input_tokens = (
        int((300 * max(1, len(llm_called_cells)) + context_char_count + 3) // 4)
        if llm_called_cells
        else 0
    )
    return {
        "hybrid_fallback_cell_count": len(set(fallback_cells)),
        "llm_requested_cell_count": len(set(llm_called_cells)),
        "llm_scored_cell_count": max(0, len(set(llm_called_cells)) - len(set(failed_cells))),
        "llm_failed_cell_count": len(set(failed_cells)),
        "estimated_llm_call_count_min": len(set(llm_called_cells)),
        "estimated_llm_input_tokens_chars4": approx_input_tokens,
        "llm_usage_note": (
            "Tokenization is completed before every pipeline. For each requested cell, the LLM receives the exact preselected word occurrences plus the complete cell, row, and column context. It returns one score per word occurrence and a separate holistic score for the complete cell. The cell score is not derived from the word scores."
        ),
    }


class understandability(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        reference: Union[pd.DataFrame, None] = None,
        metric_config: Union[str, None] = None,
    ) -> List[DQResult]:
        cfg = UnderstandabilityConfig.from_metric_config(metric_config)
        now = pd.Timestamp.now()
        llm_backend, llm_backend_error = _build_llm_backend(cfg)
        embedding_backend, embedding_error = _build_embedding_backend(cfg)

        metric_result = content_word_understandability_score(
            data=data,
            min_token_length=cfg.min_token_length,
            pipeline_mode=cfg.pipeline_mode,
            backend=llm_backend,
            ignore_numeric_columns=cfg.ignore_numeric_columns,
            backend_error=llm_backend_error,
            embedding_enabled=cfg.embedding_enabled,
            embedding_backend=embedding_backend,
            embedding_model_id=cfg.embedding_model_id,
            embedding_error=embedding_error,
            embedding_max_column_context_values=cfg.embedding_max_column_context_values,
            include_embedding_context_text=cfg.include_embedding_context_text,
            table_name=cfg.experiment_tag,
            table_description=cfg.table_description,
            target_user_group=cfg.target_user_group,
            context_lambda=cfg.context_lambda,
            context_negative_evidence_threshold=cfg.context_negative_evidence_threshold,
            context_positive_evidence_threshold=cfg.context_positive_evidence_threshold,
            language_hint=cfg.language_hint,
            hybrid_minimum_available_criteria=cfg.hybrid_minimum_available_criteria,
            hybrid_severe_notation_threshold=cfg.hybrid_severe_notation_threshold,
            hybrid_low_lexical_threshold=cfg.hybrid_low_lexical_threshold,
            hybrid_weak_context_threshold=cfg.hybrid_weak_context_threshold,
        )

        score = metric_result["score"]
        llm_usage_stats = _llm_usage_statistics(metric_result)
        explanation: Dict[str, Any] = {
            "feature": "U(T)",
            "feature_name": "Content-Level Understandability of Tabular Data",
            "description": (
                "Assesses non-empty textual cells in their original surface form. All pipelines use the same pre-tokenized word occurrences. The resource-based estimator uses a fixed four-criterion mean in which unavailable intrinsic evidence contributes zero, followed by a strongly bounded conservative context modifier. The LLM estimator scores each supplied word occurrence and independently evaluates the complete cell in its row and column context. The hybrid estimator replaces the resource-based cell score only under globally fixed uncertainty conditions."
            ),
            "score": score,
            "cell_count": metric_result.get("cell_count"),
            "token_count": metric_result.get("token_count"),
            "unique_token_count": metric_result.get("unique_token_count"),
            "token_frequencies": metric_result.get("token_frequencies"),
            "table_components": metric_result.get("table_components"),
            "aggregation": metric_result.get("aggregation"),
            "features": metric_result.get("features", {}),
            "global_parameters": metric_result.get("global_parameters", {}),
            "pipeline_mode": metric_result.get("pipeline_mode"),
            "pipeline_feature_parity": metric_result.get("pipeline_feature_parity"),
            "tokenization": metric_result.get("tokenization"),
            "original_spelling_preserved": True,
            "target_user_group": cfg.target_user_group,
            "selected_columns": metric_result.get("selected_columns", []),
            "ignored_numeric_columns": metric_result.get("ignored_numeric_columns", []),
            "schema_excluded": True,
            "embedding_enabled": metric_result.get("embedding_enabled"),
            "embedding_model_id": metric_result.get("embedding_model_id"),
            "embedding_backend_loaded": metric_result.get("embedding_backend_loaded"),
            "embedding_error": metric_result.get("embedding_error"),
            "embedding_context_level": "cell",
            "embedding_cells_with_context": metric_result.get("embedding_cells_with_context"),
            "embedding_max_column_context_values": metric_result.get("embedding_max_column_context_values"),
            "llm_context_aware_backend": metric_result.get("llm_context_aware_backend"),
            "llm_backend": cfg.hf_model_id if cfg.pipeline_mode in {"llm", "hybrid"} else None,
            "llm_backend_revision": cfg.hf_model_revision if cfg.pipeline_mode in {"llm", "hybrid"} else None,
            "llm_backend_loaded": llm_backend is not None,
            "llm_backend_error": llm_backend_error,
            "language_hint": metric_result.get("language_hint"),
            "language_detector_available": metric_result.get("language_detector_available"),
            "language_detector_error": metric_result.get("language_detector_error"),
            "lexical_resource_status": metric_result.get("lexical_resource_status"),
            **llm_usage_stats,
            "warnings": metric_result.get("warnings", []),
        }
        if metric_result.get("reason") is not None:
            explanation["reason"] = metric_result["reason"]
        if cfg.include_token_details:
            explanation["token_scores"] = metric_result.get("token_scores", {})
            explanation["word_occurrence_scores"] = metric_result.get("word_occurrence_scores", {})
            explanation["llm_word_occurrence_scores"] = metric_result.get("llm_word_occurrence_scores", {})
        if cfg.include_cell_scores:
            explanation["cell_scores"] = metric_result.get("cell_scores", {})
            explanation["lexical_cell_scores"] = metric_result.get("lexical_cell_scores", {})
            explanation["cell_details"] = metric_result.get("cell_details", {})
        if cfg.include_column_scores:
            explanation["column_scores"] = metric_result.get("column_scores", {})
        if cfg.include_row_scores:
            explanation["row_scores"] = metric_result.get("row_scores", {})

        results = [
            DQResult(
                mesTime=now,
                DQdimension="Understandability",
                DQmetric="CellContextualUnderstandability",
                DQgranularity="table",
                DQvalue=0.0 if score is None else float(score),
                DQexplanation=explanation,
                experimentTag=cfg.experiment_tag,
                configJson=cfg.as_dict(),
            )
        ]

        if cfg.output_cell_results:
            for cell_key, detail in metric_result.get("cell_details", {}).items():
                if detail.get("score") is None:
                    continue
                cell_explanation = {
                    "feature": "U(cell)",
                    "aggregation": (
                        "Resource-based: fixed four-criterion mean with unavailable evidence represented by zero, followed by a conservative bounded context modification. LLM: independent holistic assessment of the complete cell in its row and column context, alongside validated contextual word-occurrence scores. Hybrid: complete replacement by the holistic LLM cell score when global uncertainty conditions are met."
                    ),
                    "cell_key": cell_key,
                    "cell_value": detail.get("cell_value"),
                    "language": detail.get("language"),
                    "language_confidence": detail.get("language_confidence"),
                    "language_detection_source": detail.get("language_detection_source"),
                    "feature_scores": detail.get("features"),
                    "intrinsic_score": detail.get("intrinsic_score"),
                    "context_modifier": detail.get("context_modifier"),
                    "context_effect": detail.get("context_effect"),
                    "context_lambda": detail.get("context_lambda"),
                    "context_positive_evidence_threshold": cfg.context_positive_evidence_threshold,
                    "available_criterion_count": detail.get("available_criterion_count"),
                    "possible_criterion_count": detail.get("possible_criterion_count"),
                    "assessment_coverage": detail.get("assessment_coverage"),
                    "lexical_applicable_token_count": detail.get("lexical_applicable_token_count"),
                    "lexical_evaluable_token_count": detail.get("lexical_evaluable_token_count"),
                    "tokens": detail.get("tokens"),
                    "word_features": detail.get("word_features"),
                    "score_source": detail.get("score_source"),
                    "llm_called": detail.get("llm_called"),
                    "fallback_triggered": detail.get("fallback_triggered"),
                    "fallback_reasons": detail.get("fallback_reasons"),
                    "unknown_tokens": detail.get("unknown_tokens"),
                    "resource_score_before_fallback": detail.get("resource_score_before_fallback"),
                    "llm_confidence": detail.get("llm_confidence"),
                    "llm_reason": detail.get("llm_reason"),
                    "llm_cell_understandability_score": detail.get("llm_cell_understandability_score"),
                    "llm_word_scores": detail.get("llm_word_scores", []),
                    "llm_expected_word_count": detail.get("llm_expected_word_count"),
                    "llm_returned_word_count": detail.get("llm_returned_word_count"),
                    "llm_assessment_coverage": detail.get("llm_assessment_coverage"),
                    "context_feature_method": detail.get("context_feature_method"),
                    "row_context_cosine": detail.get("row_context_cosine"),
                    "column_context_cosine": detail.get("column_context_cosine"),
                    "original_spelling_preserved": True,
                    "pipeline_mode": cfg.pipeline_mode,
                    "llm_backend_loaded": llm_backend is not None,
                    "llm_backend_error": llm_backend_error,
                    "embedding_backend_loaded": embedding_backend is not None,
                    "embedding_error": embedding_error,
                    "warnings": detail.get("warnings", []),
                }
                results.append(
                    DQResult(
                        mesTime=now,
                        DQdimension="Understandability",
                        DQmetric="CellContextualUnderstandability",
                        DQgranularity="cell",
                        DQvalue=float(detail["score"]),
                        DQexplanation=cell_explanation,
                        columnNames=[str(detail.get("column"))],
                        rowIndex=_to_int_or_none(detail.get("row_index")),
                        experimentTag=cfg.experiment_tag,
                        configJson=cfg.as_dict(),
                    )
                )
        return results
