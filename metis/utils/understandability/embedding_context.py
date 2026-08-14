from __future__ import annotations

"""Cell-level contextual support for the understandability metric.

Embeddings are used only by the resource-based and hybrid pipelines. They are
calculated for complete cell values, never for individual words. The target cell
is compared with the remaining assessable values in its row and with a
reproducible sample of values from its column.

Cosine similarity is treated as contextual evidence, not as an intrinsic
understandability criterion. A cosine similarity of zero maps to neutral support
(0.5). Small negative similarities are also treated as neutral because semantic
non-relatedness between different attributes is not contradictory evidence.
Only similarities below a globally fixed negative-evidence threshold reduce the
support score.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class SentenceTransformerEmbeddingBackend:
    """Lazy SentenceTransformer backend for cell-level contextual support."""

    model_id: str
    device: Optional[str] = None

    def __post_init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Cell-level embedding context requires the 'sentence-transformers' package."
            ) from exc

        kwargs: Dict[str, Any] = {}
        if self.device and str(self.device).lower() != "auto":
            kwargs["device"] = self.device
        self.model = SentenceTransformer(self.model_id, **kwargs)

    def encode_texts(self, texts: List[str]):
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")


def _has_visible_content(text: Any) -> bool:
    return text is not None and bool(str(text).strip())


def cosine_to_context_support(
    cosine_value: Optional[float],
    negative_evidence_threshold: float = -0.40,
    positive_evidence_threshold: float = 0.60,
) -> Optional[float]:
    """Transform cosine similarity into conservative contextual support.

    Context is deliberately neutral over a broad interval because multilingual
    sentence embeddings often assign positive cosine values even to weakly
    related texts. Only sufficiently strong positive similarity is interpreted
    as supporting evidence. Only sufficiently strong negative similarity is
    interpreted as contradictory evidence.

    Mapping:

    * cosine values in ``[negative_threshold, positive_threshold]`` -> ``0.5``
    * values above the positive threshold -> linear support from ``0.5`` to ``1``
    * values below the negative threshold -> linear support from ``0.5`` to ``0``

    The neutral value ``0.5`` has no effect after centering in the cell-score
    aggregation.
    """

    if cosine_value is None:
        return None
    try:
        cosine = float(cosine_value)
    except Exception:
        return None
    if math.isnan(cosine):
        return None

    cosine = max(-1.0, min(1.0, cosine))
    negative = max(-0.99, min(0.0, float(negative_evidence_threshold)))
    positive = max(0.0, min(0.99, float(positive_evidence_threshold)))

    if negative <= cosine <= positive:
        return 0.5

    if cosine > positive:
        denominator = 1.0 - positive
        if denominator <= 0.0:
            return 1.0
        return max(0.5, min(1.0, 0.5 + 0.5 * (cosine - positive) / denominator))

    denominator = negative + 1.0
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(0.5, 0.5 * (cosine + 1.0) / denominator))


def _clean_cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _deterministic_sample(values: List[str], limit: int) -> List[str]:
    """Return a deterministic, order-preserving sample with at most ``limit`` values."""

    if len(values) <= limit:
        return values
    if limit <= 1:
        return [values[0]]

    # Evenly spaced positions including the first and last available value.
    positions = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    out: List[str] = []
    seen_positions = set()
    for position in positions:
        if position in seen_positions:
            continue
        seen_positions.add(position)
        out.append(values[position])
    return out[:limit]


def build_cell_context_records(
    data: Any,
    columns: Sequence[str],
    max_column_context_values: int = 50,
) -> Dict[str, Dict[str, Any]]:
    """Build row and column context for every selected table cell.

    The target cell is excluded from both contexts. Row context contains the
    other non-empty assessable values of the same record. Column context is a
    deterministic sample of other non-empty values in the same column. Column
    names are retained only as labels in the row-context payload. They are not
    themselves assessment objects.
    """

    limit = max(1, int(max_column_context_values))
    records: Dict[str, Dict[str, Any]] = {}

    for row_position, row_index in enumerate(data.index):
        for column_position, column in enumerate(columns):
            if column not in data.columns:
                continue
            cell_value = _clean_cell_text(data.loc[row_index, column])
            cell_key = f"{row_index}::{column}"

            row_context_parts: List[str] = []
            row_context_dict: Dict[str, str] = {}
            for other_column in columns:
                if other_column == column or other_column not in data.columns:
                    continue
                other_value = _clean_cell_text(data.loc[row_index, other_column])
                if other_value:
                    row_context_dict[str(other_column)] = other_value
                    row_context_parts.append(f"{other_column}: {other_value}")

            column_values: List[str] = []
            for other_index, other_value in data[column].items():
                if other_index == row_index:
                    continue
                text = _clean_cell_text(other_value)
                if text:
                    column_values.append(text)
            column_values = _deterministic_sample(column_values, limit)

            records[cell_key] = {
                "cell_key": cell_key,
                "row_index": row_index,
                "row_position": row_position,
                "column": str(column),
                "column_position": column_position,
                "cell_value": cell_value,
                "row_context": row_context_dict,
                "row_context_text": " | ".join(row_context_parts),
                "column_context_examples": column_values,
                "column_context_text": " | ".join(column_values),
            }
    return records


def compute_cell_embedding_context_scores(
    context_records: Dict[str, Dict[str, Any]],
    backend: Any,
    negative_evidence_threshold: float = -0.40,
    positive_evidence_threshold: float = 0.60,
) -> Dict[str, Dict[str, Any]]:
    """Calculate row- and column-context support for every cell record."""

    if backend is None:
        return {key: dict(value) for key, value in context_records.items()}

    unique_texts: Dict[str, None] = {}
    for record in context_records.values():
        for text in (
            record.get("cell_value"),
            record.get("row_context_text"),
            record.get("column_context_text"),
        ):
            if _has_visible_content(text):
                unique_texts[str(text)] = None

    text_list = list(unique_texts.keys())
    if not text_list:
        return {
            key: {
                **record,
                "row_context_support": None,
                "row_context_cosine": None,
                "column_context_support": None,
                "column_context_cosine": None,
                "context_feature_method": "cell_embedding_cosine_with_conservative_neutral_zone",
            }
            for key, record in context_records.items()
        }

    embeddings = backend.encode_texts(text_list)
    embedding_by_text = {text: embeddings[index] for index, text in enumerate(text_list)}

    out: Dict[str, Dict[str, Any]] = {}
    for key, record in context_records.items():
        cell_text = str(record.get("cell_value", ""))
        row_context = str(record.get("row_context_text", ""))
        column_context = str(record.get("column_context_text", ""))

        row_cosine: Optional[float] = None
        row_support: Optional[float] = None
        if _has_visible_content(cell_text) and _has_visible_content(row_context):
            row_cosine = float(embedding_by_text[cell_text] @ embedding_by_text[row_context])
            row_support = cosine_to_context_support(
                row_cosine,
                negative_evidence_threshold,
                positive_evidence_threshold,
            )

        column_cosine: Optional[float] = None
        column_support: Optional[float] = None
        if _has_visible_content(cell_text) and _has_visible_content(column_context):
            column_cosine = float(embedding_by_text[cell_text] @ embedding_by_text[column_context])
            column_support = cosine_to_context_support(
                column_cosine,
                negative_evidence_threshold,
                positive_evidence_threshold,
            )

        out[key] = {
            **record,
            "row_context_support": row_support,
            "row_context_cosine": row_cosine,
            "column_context_support": column_support,
            "column_context_cosine": column_cosine,
            "context_negative_evidence_threshold": float(negative_evidence_threshold),
            "context_positive_evidence_threshold": float(positive_evidence_threshold),
            "context_feature_method": "cell_embedding_cosine_with_conservative_neutral_zone",
        }
    return out


# Function aliases only. Output field names use support terminology exclusively.
build_word_context_records = build_cell_context_records
compute_word_embedding_context_scores = compute_cell_embedding_context_scores
