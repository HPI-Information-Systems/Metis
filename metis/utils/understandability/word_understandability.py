from __future__ import annotations

"""Understandability assessment for non-empty textual table cells.

The implementation follows the final metric description:

* Original surface forms are preserved.
* Numeric columns are excluded.
* Lexical recognizability is determined by direct lookup in the configured
  WordNet resources. Princeton WordNet and OdeNet are queried without automatic
  language detection.
* Criteria that cannot be evaluated remain diagnostically unavailable, but
  contribute a numerical zero to the fixed four-criterion intrinsic score.
* The resource-based pipeline calculates four intrinsic criteria and applies a
  bounded cell-level context modifier.
* The LLM pipeline evaluates the exact word occurrences selected by the shared
  preprocessing step and independently evaluates the complete cell in its row
  and column context. The holistic cell score is not derived from word scores.
* The hybrid pipeline replaces the complete resource-based score only when
  globally fixed, dataset-independent uncertainty conditions are met.
* The table score is the direct mean over successfully scored assessable cell scores. Row and column
  aggregates are retained only as diagnostic data views.
"""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from pandas.api.types import is_numeric_dtype

from metis.utils.understandability.embedding_context import (
    build_cell_context_records,
    compute_cell_embedding_context_scores,
)

VALID_PIPELINE_MODES = {"resource_based", "hybrid", "llm"}
SUPPORTED_LEXICAL_LANGUAGES = {"en", "de"}

_DATE_LIKE_PATTERN = re.compile(
    r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})$"
)
_STRUCTURED_SEGMENTED_CODE_PATTERN = re.compile(
    r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]{1,16}(?:[-/.][A-Za-z0-9]{1,16})+$"
)
_COMPACT_ALPHANUMERIC_CODE_PATTERN = re.compile(r"^(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]{2,32}$")
_NATURAL_TOKEN_PATTERN = re.compile(r"^[^\W\d_]+(?:[-'][^\W\d_]+)*$", flags=re.UNICODE)
_REPEATED_SYMBOL_PATTERN = re.compile(r"([^\w\s])\1+")
_MOJIBAKE_MARKERS = (
    "Ã",
    "Â",
    "â€",
    "â€™",
    "â€œ",
    "â€˜",
    "ðŸ",
    "ï»¿",
    "�",
)


@dataclass(frozen=True)
class LanguageDecision:
    language: Optional[str]
    confidence: Optional[float]
    source: str


@dataclass(frozen=True)
class TokenOccurrence:
    occurrence_id: str
    token: str
    column: str
    row_index: Any
    row_position: int
    column_position: int
    token_position: int
    cell_value: str
    cell_key: str
    language: Optional[str] = None


@dataclass(frozen=True)
class NotationAssessment:
    score: Optional[float]
    notation_class: str
    encoding_artifact: bool
    structured_code: bool
    irregular_symbol_sequence: bool


@dataclass(frozen=True)
class ResourceWordFeature:
    occurrence_id: str
    token: str
    lookup_form: str
    language: Optional[str]
    column: str
    row_index: Any
    token_position: int
    lexical_recognizability: Optional[float]
    semantic_ambiguity_score: Optional[float]
    notational_clarity: Optional[float]
    lexical_processing_ease: Optional[float]
    lexical_applicable: bool
    lexical_resource_available: bool
    lexical_availability_reason: str
    synset_count: Optional[int]
    synonym_count_diagnostic: Optional[int]
    hypernym_count_diagnostic: Optional[int]
    syllables: Optional[int]
    notation_class: str
    encoding_artifact: bool
    structured_code: bool
    irregular_symbol_sequence: bool

    @property
    def unknown_lexical_expression(self) -> bool:
        return (
            self.lexical_applicable
            and self.lexical_resource_available
            and self.lexical_recognizability is not None
            and float(self.lexical_recognizability) <= 0.0
        )


@dataclass(frozen=True)
class CellScore:
    cell_key: str
    row_index: Any
    column: str
    cell_value: str
    language: Optional[str]
    language_confidence: Optional[float]
    language_detection_source: str
    tokens: List[str]
    pipeline: str
    score: float
    score_source: str
    llm_called: bool
    llm_confidence: Optional[float]
    llm_reason: Optional[str]
    llm_cell_understandability_score: Optional[float]
    llm_word_scores: List[Dict[str, Any]]
    llm_expected_word_count: int
    llm_returned_word_count: int
    llm_assessment_coverage: float
    fallback_triggered: bool
    fallback_reasons: List[str]
    unknown_tokens: List[str]
    features: Dict[str, Optional[float]]
    intrinsic_score: Optional[float]
    context_modifier: float
    context_effect: float
    context_lambda: float
    available_criterion_count: int
    possible_criterion_count: int
    assessment_coverage: float
    lexical_applicable_token_count: int
    lexical_evaluable_token_count: int
    resource_score_before_fallback: Optional[float]
    word_features: List[Dict[str, Any]]
    row_context: Dict[str, str]
    column_context_examples: List[str]
    warnings: List[str] = field(default_factory=list)


class LanguageDetector:
    """Deprecated compatibility wrapper. Automatic detection is not used by the active metric."""

    def __init__(
        self,
        language_hint: Optional[str] = None,
        minimum_probability: float = 0.60,
    ) -> None:
        hint = None if language_hint is None else str(language_hint).strip().lower()
        self.language_hint = hint if hint in SUPPORTED_LEXICAL_LANGUAGES else None
        self.minimum_probability = max(0.0, min(1.0, float(minimum_probability)))
        self.available = False
        self.error: Optional[str] = None
        self._detect_langs = None
        if self.language_hint:
            self.available = True
            return
        try:
            from langdetect import DetectorFactory, detect_langs  # type: ignore

            DetectorFactory.seed = 0
            self._detect_langs = detect_langs
            self.available = True
        except Exception as exc:  # pragma: no cover - optional dependency
            self.error = str(exc)

    @staticmethod
    def _alphabetic_count(text: str) -> int:
        return sum(character.isalpha() for character in text)

    def detect(self, context_record: Dict[str, Any]) -> LanguageDecision:
        if self.language_hint:
            return LanguageDecision(self.language_hint, 1.0, "configured_language_hint")
        if not self.available or self._detect_langs is None:
            return LanguageDecision(None, None, "language_detector_unavailable")

        cell_text = str(context_record.get("cell_value", "")).strip()
        row_values = [str(value) for value in (context_record.get("row_context") or {}).values()]
        column_values = [str(value) for value in (context_record.get("column_context_examples") or [])[:8]]

        # Prefer the cell itself when it contains enough linguistic evidence.
        if self._alphabetic_count(cell_text) >= 8:
            evidence = cell_text
            source = "cell_value"
        else:
            evidence = " ".join([cell_text, *row_values, *column_values]).strip()
            source = "cell_value_with_observable_context"

        if self._alphabetic_count(evidence) < 3:
            return LanguageDecision(None, None, "insufficient_language_evidence")
        try:
            candidates = list(self._detect_langs(evidence))
        except Exception:
            return LanguageDecision(None, None, "language_detection_failed")
        if not candidates:
            return LanguageDecision(None, None, "language_detection_failed")

        top = candidates[0]
        language = str(getattr(top, "lang", "")).lower()
        probability = float(getattr(top, "prob", 0.0))
        if probability < self.minimum_probability:
            return LanguageDecision(None, probability, "language_detection_below_threshold")
        if language not in SUPPORTED_LEXICAL_LANGUAGES:
            return LanguageDecision(language, probability, "unsupported_detected_language")
        return LanguageDecision(language, probability, source)


class LexicalResource:
    """Language-specific lexical resource router.

    English queries use NLTK's Princeton WordNet corpus. German queries use
    OdeNet 1.4 through the ``wn`` Python package when both the package and the
    lexicon are installed. No custom vocabulary, spelling correction, stemming,
    dataset-specific dictionary, or hyphen splitting is applied.
    """

    def __init__(self) -> None:
        self._english_wordnet = None
        self._german_wordnet = None
        self._synset_cache: Dict[Tuple[str, str], Optional[List[Any]]] = {}
        self.resource_errors: Dict[str, Optional[str]] = {"en": None, "de": None}
        self._load_english_wordnet()
        self._load_german_wordnet()

    def _load_english_wordnet(self) -> None:
        try:
            from nltk.corpus import wordnet as nltk_wordnet  # type: ignore

            # Force corpus availability without introducing a custom vocabulary.
            nltk_wordnet.synsets("hello")
            self._english_wordnet = nltk_wordnet
        except Exception as exc:
            self.resource_errors["en"] = str(exc)

    def _load_german_wordnet(self) -> None:
        try:
            import wn  # type: ignore

            german = wn.Wordnet("odenet:1.4")
            # Query once to ensure that the lexicon is actually installed.
            german.synsets("Haus")
            self._german_wordnet = german
        except Exception as exc:  # pragma: no cover - optional dependency/data
            self.resource_errors["de"] = str(exc)

    @property
    def wordnet_available(self) -> bool:
        """Backward-compatible English-resource availability flag."""

        return self._english_wordnet is not None

    def resource_available(self, language: Optional[str]) -> bool:
        if language == "en":
            return self._english_wordnet is not None
        if language == "de":
            return self._german_wordnet is not None
        return False

    def resource_name(self, language: Optional[str]) -> Optional[str]:
        if language == "en":
            return "Princeton WordNet via NLTK"
        if language == "de":
            return "OdeNet 1.4 via wn"
        return None

    def synsets(self, token: str, language: Optional[str] = "en") -> Optional[List[Any]]:
        """Return synsets, ``[]`` for no lexical entry, or ``None`` if unavailable."""

        if language not in SUPPORTED_LEXICAL_LANGUAGES or not self.resource_available(language):
            return None
        lookup = make_lookup_form(token, language)
        if not lookup:
            return []
        key = (language, lookup)
        if key in self._synset_cache:
            return self._synset_cache[key]

        try:
            if language == "en":
                synsets = list(self._english_wordnet.synsets(lookup))  # type: ignore[union-attr]
            else:
                synsets = list(self._german_wordnet.synsets(lookup))  # type: ignore[union-attr]
        except Exception:
            synsets = []
        self._synset_cache[key] = synsets
        return synsets

    def synset_count(self, token: str, language: Optional[str] = "en") -> Optional[int]:
        synsets = self.synsets(token, language)
        return None if synsets is None else len(distinct_synsets(synsets))

    def is_known_word(self, token: str, language: Optional[str] = "en") -> Optional[bool]:
        count = self.synset_count(token, language)
        return None if count is None else count > 0


def clamp01(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        numeric = float(value)
        if math.isnan(numeric):
            return 0.0
        return max(0.0, min(1.0, numeric))
    except Exception:
        return 0.0


def validate_pipeline_mode(pipeline_mode: str) -> str:
    mode = str(pipeline_mode).strip().lower()
    if mode not in VALID_PIPELINE_MODES:
        raise ValueError(f"Unknown pipeline_mode={pipeline_mode!r}. Use one of {sorted(VALID_PIPELINE_MODES)}.")
    return mode


def mean_available(values: Iterable[Optional[float]]) -> Optional[float]:
    cleaned: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            numeric = float(value)
        except Exception:
            continue
        if not math.isnan(numeric):
            cleaned.append(numeric)
    return float(sum(cleaned) / len(cleaned)) if cleaned else None


def select_content_columns(data: pd.DataFrame, ignore_numeric_columns: bool = True) -> List[str]:
    if not ignore_numeric_columns:
        return [str(column) for column in data.columns]
    return [str(column) for column in data.columns if not is_numeric_dtype(data[column])]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def tokenize_content_value(value: Any, min_token_length: int = 2) -> List[str]:
    """Whitespace tokenization preserving each exact visible token."""

    if _is_missing(value):
        return []
    tokens: List[str] = []
    for raw_token in str(value).split():
        if raw_token and (len(raw_token) >= min_token_length or any(character.isdigit() for character in raw_token)):
            tokens.append(raw_token)
    return tokens


def extract_content_tokens(
    data: pd.DataFrame,
    min_token_length: int = 2,
    columns: Optional[Iterable[str]] = None,
    ignore_numeric_columns: bool = True,
) -> List[TokenOccurrence]:
    """Compatibility token extraction without language routing."""

    selected_columns = list(columns) if columns is not None else select_content_columns(data, ignore_numeric_columns)
    occurrences: List[TokenOccurrence] = []
    for column_position, column in enumerate(selected_columns):
        if column not in data.columns:
            continue
        for row_position, (row_index, value) in enumerate(data[column].items()):
            if _is_missing(value) or not str(value).strip():
                continue
            cell_value = str(value)
            cell_key = f"{row_index}::{column}"
            for token_position, token in enumerate(tokenize_content_value(value, min_token_length)):
                occurrences.append(
                    TokenOccurrence(
                        occurrence_id=f"r{row_position}::c{column_position}::t{token_position}",
                        token=str(token),
                        column=str(column),
                        row_index=row_index,
                        row_position=row_position,
                        column_position=column_position,
                        token_position=token_position,
                        cell_value=cell_value,
                        cell_key=cell_key,
                    )
                )
    return occurrences


def make_lookup_form(token: str, language: Optional[str] = None) -> str:
    """Create a language-query form without changing the recorded surface value."""

    value = unicodedata.normalize("NFC", str(token)).strip()
    # Remove punctuation only at the boundaries. Internal hyphens, apostrophes,
    # digits, capitalization in the stored token, and all other surface evidence
    # remain untouched outside the lexical query.
    while value and unicodedata.category(value[0]).startswith("P"):
        value = value[1:]
    while value and unicodedata.category(value[-1]).startswith("P"):
        value = value[:-1]
    return value.casefold()


def normalize_token(token: Any) -> str:
    """Deprecated compatibility alias for an internal lexical lookup form."""

    return make_lookup_form(str(token))


def _contains_control_characters(token: str) -> bool:
    return any(unicodedata.category(character) in {"Cc", "Cf"} for character in token)


def analyze_notation(token: str) -> NotationAssessment:
    """Assess visible structure rather than penalizing character types per se."""

    value = str(token)
    if not value:
        return NotationAssessment(0.0, "empty", False, False, False)

    encoding_artifact = any(marker in value for marker in _MOJIBAKE_MARKERS)
    if encoding_artifact:
        return NotationAssessment(0.10, "encoding_artifact", True, False, True)
    if _contains_control_characters(value):
        return NotationAssessment(0.05, "control_character_corruption", False, False, True)

    if _DATE_LIKE_PATTERN.fullmatch(value):
        return NotationAssessment(0.90, "structured_date", False, True, False)
    if _STRUCTURED_SEGMENTED_CODE_PATTERN.fullmatch(value):
        return NotationAssessment(0.90, "structured_segmented_code", False, True, False)
    if _COMPACT_ALPHANUMERIC_CODE_PATTERN.fullmatch(value):
        return NotationAssessment(0.82, "structured_compact_code", False, True, False)
    if value.isdigit():
        return NotationAssessment(0.90, "structured_numeric_token", False, True, False)

    repeated_symbols = bool(_REPEATED_SYMBOL_PATTERN.search(value))
    non_alnum = [character for character in value if not character.isalnum()]
    symbol_ratio = len(non_alnum) / max(1, len(value))
    underscore_mixture = "_" in value and any(character in value for character in "@#$%^&*=+|\\")
    unbalanced_pairs = any(value.count(left) != value.count(right) for left, right in (("(", ")"), ("[", "]"), ("{", "}")))
    irregular = repeated_symbols or underscore_mixture or unbalanced_pairs or symbol_ratio > 0.40
    if irregular:
        score = 0.20
        if symbol_ratio <= 0.30 and not encoding_artifact:
            score = 0.35
        return NotationAssessment(score, "irregular_symbol_sequence", False, False, True)

    lookup = make_lookup_form(value)
    if _NATURAL_TOKEN_PATTERN.fullmatch(lookup):
        # Apostrophes, hyphens, and terminal punctuation may represent clear
        # segmentation and therefore receive no blanket penalty.
        score = 0.95
        letters = "".join(character for character in value if character.isalpha())
        if letters and not (letters.islower() or letters.isupper() or letters.istitle()):
            score -= 0.05
        if len(value) > 30:
            score -= 0.05
        return NotationAssessment(clamp01(score), "natural_or_segmented_text", False, False, False)

    if all(character.isalnum() or character in "-'./:()" for character in value):
        return NotationAssessment(0.80, "parseable_mixed_notation", False, False, False)

    return NotationAssessment(0.60, "unclassified_but_parseable", False, False, False)


def notational_clarity(token: str) -> float:
    return analyze_notation(token).score


def _is_strongly_code_like(token: str) -> bool:
    notation = analyze_notation(token)
    return notation.structured_code or notation.encoding_artifact or notation.irregular_symbol_sequence


def _alphabetic_components(token: str) -> List[str]:
    return re.findall(r"[^\W\d_]+", str(token), flags=re.UNICODE)


def estimate_syllables(token: str, language: Optional[str] = None) -> int:
    """Estimate syllables for English/German alphabetic components."""

    value = str(token)
    if not value:
        return 0
    vowel_pattern = r"[aeiouyäöü]+" if language == "de" else r"[aeiouy]+"
    syllables = len(re.findall(vowel_pattern, value.casefold(), flags=re.UNICODE))
    if language == "en" and value.casefold().endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables) if any(character.isalpha() for character in value) else 0


def _component_processing_ease(component: str, language: Optional[str]) -> Tuple[float, int]:
    length = len(component)
    syllables = estimate_syllables(component, language)

    # Moderate, saturating penalties. Length and syllables are evidence of
    # processing effort, not direct determinants of understandability.
    length_penalty = 0.30 * min(1.0, max(0.0, (length - 8.0) / 20.0))
    syllable_penalty = 0.20 * min(1.0, max(0.0, (syllables - 3.0) / 5.0))
    long_unsegmented_penalty = 0.05 if length > 24 else 0.0
    score = clamp01(1.0 - length_penalty - syllable_penalty - long_unsegmented_penalty)
    return score, syllables


def lexical_processing_ease(
    token: str,
    resource: Optional[LexicalResource] = None,
    language: Optional[str] = None,
) -> Optional[float]:
    """Formal processing ease for alphabetic components only.

    The ``resource`` argument is retained for backward compatibility but is not
    used. Lexical recognition and processing ease are intentionally independent.
    """

    if _is_strongly_code_like(token):
        return None
    components = _alphabetic_components(token)
    if not components:
        return None
    return mean_available([_component_processing_ease(component, language)[0] for component in components])


def _synset_identifier(synset: Any) -> str:
    for attribute in ("id", "name"):
        value = getattr(synset, attribute, None)
        if callable(value):
            try:
                return str(value())
            except Exception:
                continue
        if value is not None:
            return str(value)
    return repr(synset)


def distinct_synsets(synsets: Sequence[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for synset in synsets:
        identifier = _synset_identifier(synset)
        if identifier in seen:
            continue
        seen.add(identifier)
        out.append(synset)
    return out


def _diagnostic_lemma_count(synsets: Sequence[Any]) -> int:
    lemmas = set()
    for synset in synsets:
        try:
            values = synset.lemmas()
        except Exception:
            continue
        for lemma in values:
            if hasattr(lemma, "name"):
                try:
                    lemmas.add(str(lemma.name()))
                    continue
                except Exception:
                    pass
            lemmas.add(str(lemma))
    return len(lemmas)


def _diagnostic_hypernym_count(synsets: Sequence[Any]) -> int:
    hypernyms = set()
    for synset in synsets:
        try:
            values = synset.hypernyms()
        except Exception:
            continue
        for hypernym in values:
            hypernyms.add(_synset_identifier(hypernym))
    return len(hypernyms)


def synset_count_to_ambiguity_score(synset_count: int) -> Optional[float]:
    """Map distinct lexical senses to a score where higher means less ambiguity."""

    count = int(synset_count)
    if count <= 0:
        return None
    return clamp01(1.0 / (1.0 + math.log(float(count))))


def lexical_recognizability(
    token: str,
    resource: LexicalResource,
    language: Optional[str] = "en",
) -> Optional[float]:
    """Return 1/0 when a lexical lookup is applicable, otherwise ``None``."""

    notation = analyze_notation(token)
    lookup = make_lookup_form(token, language)
    lexical_applicable = bool(_alphabetic_components(lookup)) and not notation.structured_code
    if not lexical_applicable:
        return None
    known = resource.is_known_word(lookup, language)
    if known is None:
        return None
    return 1.0 if known else 0.0


def semantic_ambiguity(
    token: str,
    resource: LexicalResource,
    language: Optional[str] = "en",
) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    """Estimate ambiguity solely from the number of distinct synsets.

    Synonym and hypernym counts are returned for diagnostics only and do not
    contribute to the score.
    """

    synsets = resource.synsets(token, language)
    if synsets is None:
        return None, None, None, None
    distinct = distinct_synsets(synsets)
    count = len(distinct)
    if count <= 0:
        return None, 0, 0, 0
    score = synset_count_to_ambiguity_score(count)
    return score, count, _diagnostic_lemma_count(distinct), _diagnostic_hypernym_count(distinct)


# Deprecated function name retained only for old imports. No duplicate output
# field is produced by the final metric.
def cognate_ambiguity(
    token: str,
    resource: LexicalResource,
    language: Optional[str] = "en",
) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[int]]:
    return semantic_ambiguity(token, resource, language)


def _lookup_across_wordnets(
    token: str,
    resource: LexicalResource,
) -> Tuple[Dict[str, List[Any]], List[str]]:
    """Query every available configured WordNet without detecting language.

    The resource-based pipeline asks a direct operational question: is the
    unchanged word-like token represented in at least one configured lexical
    resource? Princeton WordNet and OdeNet are queried independently. The list
    of available resources is returned separately from the resources containing
    at least one lexical sense.
    """

    matches: Dict[str, List[Any]] = {}
    available: List[str] = []
    for lexical_language in ("en", "de"):
        if not resource.resource_available(lexical_language):
            continue
        available.append(lexical_language)
        synsets = resource.synsets(token, lexical_language)
        if synsets:
            matches[lexical_language] = distinct_synsets(synsets)
    return matches, available


def _select_ambiguity_evidence(
    matches: Dict[str, List[Any]],
) -> Tuple[Optional[str], List[Any]]:
    """Select one resource deterministically for ambiguity diagnostics.

    If the same surface form occurs in multiple WordNets, the resource with the
    greatest number of distinct senses is used. This is conservative and avoids
    summing unrelated senses across languages. Ties follow the fixed resource
    order English, then German.
    """

    if not matches:
        return None, []
    priority = {"en": 0, "de": 1}
    selected = sorted(
        matches,
        key=lambda language: (-len(matches[language]), priority.get(language, 99)),
    )[0]
    return selected, matches[selected]


def score_resource_word(token_occurrence: TokenOccurrence, resource: LexicalResource) -> ResourceWordFeature:
    token = token_occurrence.token
    lookup = make_lookup_form(token)
    notation = analyze_notation(token)
    lexical_applicable = bool(_alphabetic_components(lookup)) and not notation.structured_code

    matches, available_resources = _lookup_across_wordnets(lookup, resource)
    resource_available = bool(available_resources)
    matched_resource, selected_synsets = _select_ambiguity_evidence(matches)

    if not lexical_applicable:
        lexical_reason = "not_applicable_to_structured_or_nonlexical_token"
        recognizability: Optional[float] = None
        ambiguity: Optional[float] = None
        synset_count: Optional[int] = None
        synonym_count: Optional[int] = None
        hypernym_count: Optional[int] = None
    elif not resource_available:
        lexical_reason = "no_configured_wordnet_available"
        recognizability = None
        ambiguity = None
        synset_count = None
        synonym_count = None
        hypernym_count = None
    elif not matches:
        lexical_reason = "not_found_in_available_wordnets"
        recognizability = 0.0
        ambiguity = None
        synset_count = 0
        synonym_count = 0
        hypernym_count = 0
    else:
        resource_labels = "+".join(sorted(matches))
        lexical_reason = f"found_in_wordnet_{resource_labels}"
        recognizability = 1.0
        synset_count = len(selected_synsets)
        ambiguity = synset_count_to_ambiguity_score(synset_count)
        synonym_count = _diagnostic_lemma_count(selected_synsets)
        hypernym_count = _diagnostic_hypernym_count(selected_synsets)

    # A configured language hint may improve syllable estimation, but no
    # automatic language classifier is used for lexical-resource routing.
    processing_language = token_occurrence.language if token_occurrence.language in SUPPORTED_LEXICAL_LANGUAGES else None
    processing = lexical_processing_ease(token, language=processing_language)
    component_syllables = [
        estimate_syllables(component, processing_language)
        for component in _alphabetic_components(token)
    ]

    return ResourceWordFeature(
        occurrence_id=token_occurrence.occurrence_id,
        token=token,
        lookup_form=lookup,
        language=matched_resource or processing_language,
        column=token_occurrence.column,
        row_index=token_occurrence.row_index,
        token_position=token_occurrence.token_position,
        lexical_recognizability=recognizability,
        semantic_ambiguity_score=ambiguity,
        notational_clarity=notation.score,
        lexical_processing_ease=processing,
        lexical_applicable=lexical_applicable,
        lexical_resource_available=resource_available,
        lexical_availability_reason=lexical_reason,
        synset_count=synset_count,
        synonym_count_diagnostic=synonym_count,
        hypernym_count_diagnostic=hypernym_count,
        syllables=sum(component_syllables) if component_syllables else None,
        notation_class=notation.notation_class,
        encoding_artifact=notation.encoding_artifact,
        structured_code=notation.structured_code,
        irregular_symbol_sequence=notation.irregular_symbol_sequence,
    )


def _cell_feature_mean_with_zero(
    word_features: List[ResourceWordFeature],
    attr: str,
) -> Tuple[float, bool]:
    """Aggregate a word criterion with unavailable values represented by zero.

    The availability flag remains diagnostic. Numerically, every predefined
    word occurrence contributes to every intrinsic criterion. A criterion that
    cannot be evaluated for a word therefore lowers the resource-based score
    instead of disappearing from the denominator.
    """

    if not word_features:
        return 0.0, False
    raw_values = [getattr(item, attr) for item in word_features]
    available = any(value is not None for value in raw_values)
    numeric_values = [clamp01(value) if value is not None else 0.0 for value in raw_values]
    return float(sum(numeric_values) / len(numeric_values)), available


def _resource_cell_score(
    word_features: List[ResourceWordFeature],
    context_record: Dict[str, Any],
    cell_value: str,
    context_lambda: float,
) -> Tuple[
    float,
    Dict[str, Optional[float]],
    Optional[float],
    float,
    float,
    int,
    int,
    float,
    int,
    int,
]:
    lexical, lexical_available = _cell_feature_mean_with_zero(
        word_features, "lexical_recognizability"
    )
    ambiguity, ambiguity_available = _cell_feature_mean_with_zero(
        word_features, "semantic_ambiguity_score"
    )
    notation, notation_available = _cell_feature_mean_with_zero(
        word_features, "notational_clarity"
    )
    processing, processing_available = _cell_feature_mean_with_zero(
        word_features, "lexical_processing_ease"
    )

    # A non-empty cell with no token after the configured tokenization still has
    # assessable visible notation. The other three criteria remain zero.
    if not notation_available and cell_value.strip():
        notation = analyze_notation(cell_value).score
        notation_available = True

    intrinsic_features: Dict[str, Optional[float]] = {
        "lexical_recognizability": lexical,
        "semantic_ambiguity_score": ambiguity,
        "notational_clarity": notation,
        "lexical_processing_ease": processing,
    }
    availability = {
        "lexical_recognizability": lexical_available,
        "semantic_ambiguity_score": ambiguity_available,
        "notational_clarity": notation_available,
        "lexical_processing_ease": processing_available,
    }
    available_count = sum(availability.values())
    possible_count = len(intrinsic_features)
    coverage = available_count / possible_count

    # Fixed denominator: unavailable criteria enter the resource-based score as
    # zero rather than being excluded.
    intrinsic_score = float(sum(float(value or 0.0) for value in intrinsic_features.values()) / possible_count)

    row_support = context_record.get("row_context_support")
    column_support = context_record.get("column_context_support")
    context_values = [value for value in (row_support, column_support) if value is not None]
    context_modifier = mean_available([float(value) - 0.5 for value in context_values]) or 0.0
    context_lambda = max(0.0, min(1.0, float(context_lambda)))
    context_effect = context_lambda * context_modifier
    score = clamp01(intrinsic_score + context_effect)

    lexical_applicable_count = sum(feature.lexical_applicable for feature in word_features)
    lexical_evaluable_count = sum(
        feature.lexical_applicable and feature.lexical_resource_available for feature in word_features
    )

    features: Dict[str, Optional[float]] = {
        **intrinsic_features,
        "row_context_support": row_support,
        "column_context_support": column_support,
    }
    return (
        score,
        features,
        intrinsic_score,
        context_modifier,
        context_effect,
        available_count,
        possible_count,
        coverage,
        lexical_applicable_count,
        lexical_evaluable_count,
    )


def _context_support_mean(features: Dict[str, Optional[float]]) -> Optional[float]:
    return mean_available([features.get("row_context_support"), features.get("column_context_support")])


def _hybrid_fallback_decision(
    *,
    cell_value: str,
    tokens: List[str],
    word_features: List[ResourceWordFeature],
    features: Dict[str, Optional[float]],
    available_criterion_count: int,
    minimum_available_criteria: int,
    severe_notation_threshold: float,
    low_lexical_threshold: float,
    weak_context_threshold: float,
) -> Tuple[bool, List[str], List[str]]:
    """Dataset-independent uncertainty decision for the hybrid pipeline."""

    reasons: List[str] = []
    if cell_value.strip() and not tokens:
        reasons.append("non_interpretable_tokenization")

    notation = features.get("notational_clarity")
    if notation is not None and notation <= severe_notation_threshold:
        reasons.append("severe_notational_corruption")

    lexical_applicable = [feature for feature in word_features if feature.lexical_applicable]
    lexical_evaluable = [feature for feature in lexical_applicable if feature.lexical_resource_available]
    unknown_tokens = [feature.token for feature in lexical_evaluable if feature.unknown_lexical_expression]

    if lexical_applicable and not lexical_evaluable:
        reasons.append("insufficient_lexical_evidence")

    lexical = features.get("lexical_recognizability")
    context_mean = _context_support_mean(features)
    weak_context = context_mean is None or context_mean <= weak_context_threshold
    if lexical is not None and lexical <= low_lexical_threshold and weak_context:
        reasons.append("unknown_expressions_with_weak_context_support")

    if available_criterion_count < int(minimum_available_criteria):
        reasons.append("insufficient_intrinsic_assessment_coverage")

    # Deduplicate while preserving deterministic order.
    unique_reasons = list(dict.fromkeys(reasons))
    return bool(unique_reasons), unique_reasons, unknown_tokens


def _cell_llm_requests(
    cell_context_records: Dict[str, Dict[str, Any]],
    cell_to_occurrences: Dict[str, List[TokenOccurrence]],
    language_by_cell: Dict[str, LanguageDecision],
    cell_keys: Sequence[str],
    table_name: Optional[str],
    table_description: Optional[str] = None,
    target_user_group: str = "general adult data users",
) -> List[Dict[str, Any]]:
    """Create complete cell requests from the shared token occurrences.

    The LLM never determines its own units. ``target_words`` contains exactly
    the occurrences already used by the resource-based pipeline, including the
    original surface form and deterministic occurrence ID.
    """

    requests: List[Dict[str, Any]] = []
    for cell_key in cell_keys:
        context = cell_context_records.get(cell_key)
        occurrences = cell_to_occurrences.get(cell_key, [])
        if not context:
            continue
        requests.append(
            {
                "id": cell_key,
                "table_name": table_name or "",
                "table_description": table_description or "",
                "row_index": context.get("row_index"),
                "column": context.get("column"),
                "language": language_by_cell.get(cell_key).language if cell_key in language_by_cell else None,
                "target_user_group": str(target_user_group),
                "cell_value": context.get("cell_value", ""),
                "target_words": [
                    {
                        "word_id": occurrence.occurrence_id,
                        "surface": occurrence.token,
                        "token_position": occurrence.token_position,
                    }
                    for occurrence in occurrences
                ],
                "row_context": context.get("row_context", {}) or {},
                "column_context_examples": context.get("column_context_examples", []) or [],
            }
        )
    return requests


def _call_llm_cell_backend(requests: List[Dict[str, Any]], backend: Any) -> Tuple[Dict[str, Dict[str, Any]], bool]:
    if backend is None or not requests:
        return {}, False
    if not hasattr(backend, "score_understandability_cells"):
        return {}, False
    raw = backend.score_understandability_cells(requests)
    if not isinstance(raw, dict):
        return {}, True
    out: Dict[str, Dict[str, Any]] = {}
    requested_ids = {str(request["id"]) for request in requests}
    for cell_id, value in raw.items():
        visible_id = str(cell_id)
        if visible_id not in requested_ids or not isinstance(value, dict):
            continue
        if "understandability_score" not in value or not isinstance(value.get("word_scores"), list):
            continue
        word_scores = [dict(item) for item in value.get("word_scores", []) if isinstance(item, dict)]
        expected_count = int(value.get("expected_word_count", len(word_scores)))
        returned_count = int(value.get("returned_word_count", len(word_scores)))
        coverage = float(value.get("assessment_coverage", 0.0 if expected_count == 0 else returned_count / expected_count))
        out[visible_id] = {
            "understandability_score": clamp01(value.get("understandability_score")),
            "cell_score": clamp01(value.get("cell_score", value.get("understandability_score"))),
            "word_scores": word_scores,
            "expected_word_count": expected_count,
            "returned_word_count": returned_count,
            "assessment_coverage": clamp01(coverage),
            "confidence": None if value.get("confidence") is None else clamp01(value.get("confidence")),
            "reason": None if value.get("reason") is None else str(value.get("reason")),
            "aggregation": value.get("aggregation", "independent_holistic_cell_assessment"),
            "cell_score_source": value.get("cell_score_source", "llm_holistic_cell_assessment"),
        }
    return out, True


def _token_scores_from_resource_words(resource_word_features: Dict[str, ResourceWordFeature]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[ResourceWordFeature]] = defaultdict(list)
    for feature in resource_word_features.values():
        grouped[feature.token].append(feature)

    out: Dict[str, Dict[str, Any]] = {}
    for token, items in grouped.items():
        lexical = float(sum(clamp01(item.lexical_recognizability) if item.lexical_recognizability is not None else 0.0 for item in items) / len(items))
        ambiguity = float(sum(clamp01(item.semantic_ambiguity_score) if item.semantic_ambiguity_score is not None else 0.0 for item in items) / len(items))
        notation = float(sum(clamp01(item.notational_clarity) if item.notational_clarity is not None else 0.0 for item in items) / len(items))
        processing = float(sum(clamp01(item.lexical_processing_ease) if item.lexical_processing_ease is not None else 0.0 for item in items) / len(items))
        intrinsic = float((lexical + ambiguity + notation + processing) / 4.0)
        out[token] = {
            "token": token,
            "lexical_recognizability": lexical,
            "semantic_ambiguity_score": ambiguity,
            "notational_clarity": notation,
            "lexical_processing_ease": processing,
            "word_intrinsic_score": intrinsic,
            "explanation": {
                "aggregation": "fixed four-criterion mean; unavailable criterion values contribute zero; cell context is not included in token output",
                "occurrence_count": len(items),
                "languages": sorted({item.language for item in items if item.language}),
                "score_source": "resource_based_word_evidence",
            },
        }
    return out


def _empty_result(
    mode: str,
    selected_columns: List[str],
    ignore_numeric_columns: bool,
    reason: str,
    warnings: List[str],
    context_lambda: float,
    context_negative_evidence_threshold: float,
    context_positive_evidence_threshold: float,
) -> Dict[str, Any]:
    return {
        "score": None,
        "reason": reason,
        "warnings": warnings,
        "token_count": 0,
        "unique_token_count": 0,
        "cell_count": 0,
        "token_scores": {},
        "word_occurrence_scores": {},
        "llm_word_occurrence_scores": {},
        "token_frequencies": {},
        "cell_scores": {},
        "lexical_cell_scores": {},
        "cell_details": {},
        "column_scores": {},
        "row_scores": {},
        "table_components": {
            "direct_cell_mean": None,
            "assessable_cell_count": 0,
            "context_lambda": context_lambda,
            "context_negative_evidence_threshold": context_negative_evidence_threshold,
            "context_positive_evidence_threshold": context_positive_evidence_threshold,
        },
        "aggregation": "direct mean over successfully scored assessable cell scores; no assessable textual cells found",
        "pipeline_mode": mode,
        "tokenization": "whitespace tokenization; original visible spelling preserved",
        "original_spelling_preserved": True,
        "selected_columns": selected_columns,
        "ignored_numeric_columns": [],
        "ignore_numeric_columns": ignore_numeric_columns,
        "llm_requested_cell_count": 0,
        "llm_scored_cell_count": 0,
        "llm_failed_cell_count": 0,
        "hybrid_fallback_cell_count": 0,
    }


def content_word_understandability_score(
    data: pd.DataFrame,
    min_token_length: int = 2,
    columns: Optional[Iterable[str]] = None,
    pipeline_mode: str = "resource_based",
    backend: Any = None,
    alpha: float = 0.5,  # deprecated; retained for old callers and ignored
    ignore_numeric_columns: bool = True,
    backend_error: Optional[str] = None,
    embedding_enabled: bool = True,
    embedding_backend: Any = None,
    embedding_model_id: Optional[str] = None,
    embedding_error: Optional[str] = None,
    embedding_max_column_context_tokens: int = 50,
    embedding_max_column_context_values: Optional[int] = None,
    include_embedding_context_text: bool = False,
    feature_weights: Optional[Dict[str, float]] = None,  # deprecated and ignored
    table_name: Optional[str] = None,
    table_description: Optional[str] = None,
    target_user_group: str = "general adult data users",
    context_lambda: float = 0.10,
    context_negative_evidence_threshold: float = -0.40,
    context_positive_evidence_threshold: float = 0.60,
    language_hint: Optional[str] = None,
    hybrid_minimum_available_criteria: int = 2,
    hybrid_severe_notation_threshold: float = 0.35,
    hybrid_low_lexical_threshold: float = 0.40,
    hybrid_weak_context_threshold: float = 0.55,
    lexical_resource: Optional[LexicalResource] = None,
    language_detector: Optional[LanguageDetector] = None,
) -> Dict[str, Any]:
    mode = validate_pipeline_mode(pipeline_mode)
    selected_columns = list(columns) if columns is not None else select_content_columns(data, ignore_numeric_columns)
    warnings: List[str] = []
    if backend_error:
        warnings.append(f"LLM backend could not be loaded: {backend_error}")
    if embedding_error:
        warnings.append(f"Embedding backend could not be loaded: {embedding_error}")
    if feature_weights:
        warnings.append("Legacy feature_weights were ignored because the final resource-based score uses the specified intrinsic mean and bounded context modifier.")
    if alpha != 0.5:
        warnings.append("Legacy alpha was ignored because the table score is the direct mean over successfully scored assessable cell scores.")

    max_column_context_values = int(embedding_max_column_context_values or embedding_max_column_context_tokens or 50)
    all_context_records = build_cell_context_records(
        data,
        selected_columns,
        max_column_context_values=max_column_context_values,
    )
    cell_context_records = {
        key: record
        for key, record in all_context_records.items()
        if str(record.get("cell_value", "")).strip()
    }
    if not cell_context_records:
        return _empty_result(
            mode,
            selected_columns,
            ignore_numeric_columns,
            "no non-empty textual cells found",
            warnings,
            context_lambda,
            context_negative_evidence_threshold,
            context_positive_evidence_threshold,
        )

    context_by_cell = {key: dict(record) for key, record in cell_context_records.items()}
    if mode in {"resource_based", "hybrid"} and embedding_enabled:
        if embedding_backend is None:
            warnings.append(
                "Cell-level contextual support is unavailable because no embedding backend is loaded. Intrinsic criteria are still assessed."
            )
        else:
            try:
                context_by_cell = compute_cell_embedding_context_scores(
                    cell_context_records,
                    embedding_backend,
                    negative_evidence_threshold=context_negative_evidence_threshold,
                    positive_evidence_threshold=context_positive_evidence_threshold,
                )
            except Exception as exc:
                embedding_error = str(exc)
                warnings.append(
                    f"Cell-level contextual support failed: {embedding_error}. Intrinsic criteria are still assessed."
                )
    else:
        context_by_cell = {
            key: {**record, "context_feature_method": "prompt_context" if mode == "llm" else "not_requested"}
            for key, record in cell_context_records.items()
        }

    # No automatic language detection is used. Every word-like token is queried
    # directly against all configured WordNet resources. ``language_hint`` is
    # retained only as optional metadata for the LLM and syllable estimation.
    if language_detector is not None:
        warnings.append(
            "A supplied language detector was ignored. Lexical recognizability now queries the configured WordNet resources directly."
        )
    configured_language = (
        str(language_hint).strip().lower()
        if language_hint is not None and str(language_hint).strip().lower() in SUPPORTED_LEXICAL_LANGUAGES
        else None
    )
    resource = lexical_resource or (LexicalResource() if mode in {"resource_based", "hybrid"} else None)
    if resource is not None:
        if not resource.resource_available("en"):
            warnings.append("Princeton WordNet is unavailable; its missing evidence contributes zero to the resource-based score.")
        if not resource.resource_available("de"):
            warnings.append("OdeNet 1.4 is unavailable; its missing evidence contributes zero to the resource-based score.")

    language_by_cell: Dict[str, LanguageDecision] = {
        cell_key: LanguageDecision(
            configured_language,
            1.0 if configured_language else None,
            "configured_language_metadata" if configured_language else "automatic_language_detection_not_used",
        )
        for cell_key in cell_context_records
    }

    cell_to_occurrences: Dict[str, List[TokenOccurrence]] = {key: [] for key in cell_context_records}
    occurrences: List[TokenOccurrence] = []
    for cell_key, record in cell_context_records.items():
        tokens = tokenize_content_value(record.get("cell_value"), min_token_length)
        language = language_by_cell[cell_key].language
        for token_position, token in enumerate(tokens):
            occurrence = TokenOccurrence(
                occurrence_id=f"r{record.get('row_position')}::c{record.get('column_position')}::t{token_position}",
                token=token,
                column=str(record.get("column", "")),
                row_index=record.get("row_index"),
                row_position=int(record.get("row_position", 0)),
                column_position=int(record.get("column_position", 0)),
                token_position=token_position,
                cell_value=str(record.get("cell_value", "")),
                cell_key=cell_key,
                language=language,
            )
            occurrences.append(occurrence)
            cell_to_occurrences[cell_key].append(occurrence)

    resource_word_features: Dict[str, ResourceWordFeature] = {}
    if resource is not None:
        for occurrence in occurrences:
            resource_word_features[occurrence.occurrence_id] = score_resource_word(occurrence, resource)

    resource_cell_data: Dict[str, Dict[str, Any]] = {}
    if mode in {"resource_based", "hybrid"}:
        for cell_key, occurrence_list in cell_to_occurrences.items():
            word_features = [resource_word_features[item.occurrence_id] for item in occurrence_list]
            result = _resource_cell_score(
                word_features,
                context_by_cell.get(cell_key, {}),
                str(cell_context_records[cell_key].get("cell_value", "")),
                context_lambda,
            )
            (
                resource_score,
                features,
                intrinsic_score,
                context_modifier,
                context_effect,
                available_count,
                possible_count,
                coverage,
                lexical_applicable_count,
                lexical_evaluable_count,
            ) = result
            fallback, fallback_reasons, unknown_tokens = _hybrid_fallback_decision(
                cell_value=str(cell_context_records[cell_key].get("cell_value", "")),
                tokens=[item.token for item in occurrence_list],
                word_features=word_features,
                features=features,
                available_criterion_count=available_count,
                minimum_available_criteria=hybrid_minimum_available_criteria,
                severe_notation_threshold=hybrid_severe_notation_threshold,
                low_lexical_threshold=hybrid_low_lexical_threshold,
                weak_context_threshold=hybrid_weak_context_threshold,
            )
            resource_cell_data[cell_key] = {
                "resource_score": resource_score,
                "features": features,
                "intrinsic_score": intrinsic_score,
                "context_modifier": context_modifier,
                "context_effect": context_effect,
                "available_criterion_count": available_count,
                "possible_criterion_count": possible_count,
                "assessment_coverage": coverage,
                "lexical_applicable_token_count": lexical_applicable_count,
                "lexical_evaluable_token_count": lexical_evaluable_count,
                "word_features": word_features,
                "fallback": fallback,
                "fallback_reasons": fallback_reasons,
                "unknown_tokens": unknown_tokens,
            }

    if mode == "llm":
        llm_requested_cell_keys = list(cell_context_records.keys())
    elif mode == "hybrid":
        llm_requested_cell_keys = [
            cell_key for cell_key, detail in resource_cell_data.items() if detail["fallback"]
        ]
    else:
        llm_requested_cell_keys = []

    llm_scores_by_cell: Dict[str, Dict[str, Any]] = {}
    llm_context_aware = False
    if llm_requested_cell_keys:
        if backend is None:
            warnings.append(
                "LLM scoring was requested but no LLM backend is loaded. Hybrid fallback cells and LLM-only cells receive no model result."
            )
        else:
            requests = _cell_llm_requests(
                cell_context_records,
                cell_to_occurrences,
                language_by_cell,
                llm_requested_cell_keys,
                table_name,
                table_description,
                target_user_group,
            )
            llm_scores_by_cell, llm_context_aware = _call_llm_cell_backend(requests, backend)
            missing = [cell_key for cell_key in llm_requested_cell_keys if cell_key not in llm_scores_by_cell]
            if missing:
                warnings.append(f"The {mode} LLM produced no result for {len(missing)} cell(s).")

    cell_results: Dict[str, CellScore] = {}
    for cell_key, record in cell_context_records.items():
        context = context_by_cell.get(cell_key, record)
        decision = language_by_cell[cell_key]
        tokens = [occurrence.token for occurrence in cell_to_occurrences[cell_key]]
        cell_warnings: List[str] = []

        if mode == "llm":
            llm_result = llm_scores_by_cell.get(cell_key)
            score = clamp01(llm_result.get("understandability_score")) if llm_result else None
            source = "llm_holistic_cell_assessment" if llm_result else "llm_holistic_cell_assessment_missing"
            features: Dict[str, Optional[float]] = {"llm_understandability_score": score if llm_result else None}
            intrinsic_score = None
            context_modifier = 0.0
            context_effect = 0.0
            available_count = 0
            possible_count = 0
            coverage = 0.0
            lexical_applicable_count = 0
            lexical_evaluable_count = 0
            word_features: List[ResourceWordFeature] = []
            fallback = False
            fallback_reasons: List[str] = []
            unknown_tokens: List[str] = []
            llm_called = True
            llm_confidence = llm_result.get("confidence") if llm_result else None
            llm_reason = llm_result.get("reason") if llm_result else None
            llm_cell_understandability_score = score if llm_result else None
            llm_word_scores = list(llm_result.get("word_scores", [])) if llm_result else []
            llm_expected_word_count = int(llm_result.get("expected_word_count", len(tokens))) if llm_result else len(tokens)
            llm_returned_word_count = int(llm_result.get("returned_word_count", 0)) if llm_result else 0
            llm_assessment_coverage = float(llm_result.get("assessment_coverage", 0.0)) if llm_result else 0.0
            resource_before = None
        else:
            resource_detail = resource_cell_data[cell_key]
            features = dict(resource_detail["features"])
            features["llm_understandability_score"] = None
            intrinsic_score = resource_detail["intrinsic_score"]
            context_modifier = resource_detail["context_modifier"]
            context_effect = resource_detail["context_effect"]
            available_count = resource_detail["available_criterion_count"]
            possible_count = resource_detail["possible_criterion_count"]
            coverage = resource_detail["assessment_coverage"]
            lexical_applicable_count = resource_detail["lexical_applicable_token_count"]
            lexical_evaluable_count = resource_detail["lexical_evaluable_token_count"]
            word_features = resource_detail["word_features"]
            fallback_reasons = resource_detail["fallback_reasons"]
            unknown_tokens = resource_detail["unknown_tokens"]

            if mode == "hybrid" and resource_detail["fallback"]:
                fallback = True
                llm_called = True
                llm_result = llm_scores_by_cell.get(cell_key)
                resource_before = resource_detail["resource_score"]
                if llm_result:
                    score = clamp01(llm_result.get("understandability_score"))
                    source = "hybrid_llm_replacement_global_uncertainty"
                    features["llm_understandability_score"] = score
                    llm_confidence = llm_result.get("confidence")
                    llm_reason = llm_result.get("reason")
                    llm_cell_understandability_score = score
                    llm_word_scores = list(llm_result.get("word_scores", []))
                    llm_expected_word_count = int(llm_result.get("expected_word_count", len(tokens)))
                    llm_returned_word_count = int(llm_result.get("returned_word_count", len(llm_word_scores)))
                    llm_assessment_coverage = float(llm_result.get("assessment_coverage", 1.0))
                else:
                    score = None
                    source = "hybrid_llm_replacement_missing"
                    llm_confidence = None
                    llm_reason = None
                    llm_cell_understandability_score = None
                    llm_word_scores = []
                    llm_expected_word_count = len(tokens)
                    llm_returned_word_count = 0
                    llm_assessment_coverage = 0.0
                    cell_warnings.append("LLM replacement was requested but no complete validated word-and-cell LLM result was returned.")
            else:
                fallback = False
                llm_called = False
                score = resource_detail["resource_score"]
                source = "resource_based_cell" if mode == "resource_based" else "hybrid_resource_based_no_fallback"
                resource_before = None
                llm_confidence = None
                llm_reason = None
                llm_cell_understandability_score = None
                llm_word_scores = []
                llm_expected_word_count = 0
                llm_returned_word_count = 0
                llm_assessment_coverage = 0.0

        cell_results[cell_key] = CellScore(
            cell_key=cell_key,
            row_index=record.get("row_index"),
            column=str(record.get("column", "")),
            cell_value=str(record.get("cell_value", "")),
            language=decision.language,
            language_confidence=decision.confidence,
            language_detection_source=decision.source,
            tokens=tokens,
            pipeline=mode,
            score=None if score is None else float(score),
            score_source=source,
            llm_called=llm_called,
            llm_confidence=None if llm_confidence is None else float(llm_confidence),
            llm_reason=llm_reason,
            llm_cell_understandability_score=(
                None
                if llm_cell_understandability_score is None
                else float(llm_cell_understandability_score)
            ),
            llm_word_scores=llm_word_scores,
            llm_expected_word_count=int(llm_expected_word_count),
            llm_returned_word_count=int(llm_returned_word_count),
            llm_assessment_coverage=float(llm_assessment_coverage),
            fallback_triggered=fallback,
            fallback_reasons=fallback_reasons,
            unknown_tokens=unknown_tokens,
            features=features,
            intrinsic_score=intrinsic_score,
            context_modifier=float(context_modifier),
            context_effect=float(context_effect),
            context_lambda=float(context_lambda),
            available_criterion_count=int(available_count),
            possible_criterion_count=int(possible_count),
            assessment_coverage=float(coverage),
            lexical_applicable_token_count=int(lexical_applicable_count),
            lexical_evaluable_token_count=int(lexical_evaluable_count),
            resource_score_before_fallback=resource_before,
            word_features=[asdict(feature) for feature in word_features],
            row_context=record.get("row_context", {}) or {},
            column_context_examples=record.get("column_context_examples", []) or [],
            warnings=cell_warnings,
        )

    cell_scores = {cell_key: result.score for cell_key, result in cell_results.items()}
    table_score = mean_available(cell_scores.values())

    column_values: Dict[str, List[float]] = defaultdict(list)
    row_values: Dict[str, List[float]] = defaultdict(list)
    for result in cell_results.values():
        if result.score is not None:
            column_values[result.column].append(float(result.score))
            row_values[str(result.row_index)].append(float(result.score))
    column_scores = {key: float(sum(values) / len(values)) for key, values in column_values.items() if values}
    row_scores = {key: float(sum(values) / len(values)) for key, values in row_values.items() if values}

    cell_details: Dict[str, Dict[str, Any]] = {}
    for cell_key, result in cell_results.items():
        detail = asdict(result)
        detail["context_feature_method"] = context_by_cell.get(cell_key, {}).get("context_feature_method")
        detail["row_context_cosine"] = context_by_cell.get(cell_key, {}).get("row_context_cosine")
        detail["column_context_cosine"] = context_by_cell.get(cell_key, {}).get("column_context_cosine")
        if include_embedding_context_text:
            detail["row_context_text"] = context_by_cell.get(cell_key, {}).get("row_context_text")
            detail["column_context_text"] = context_by_cell.get(cell_key, {}).get("column_context_text")
        cell_details[cell_key] = detail

    llm_word_occurrence_scores: Dict[str, Dict[str, Any]] = {}
    for cell_key, detail in cell_details.items():
        for item in detail.get("llm_word_scores", []) or []:
            if not isinstance(item, dict) or not item.get("word_id"):
                continue
            llm_word_occurrence_scores[str(item["word_id"])] = {
                **item,
                "cell_key": cell_key,
                "cell_value": detail.get("cell_value"),
                "column": detail.get("column"),
                "row_index": detail.get("row_index"),
                "pipeline": mode,
            }

    token_frequencies = Counter(occurrence.token for occurrence in occurrences)
    token_scores = _token_scores_from_resource_words(resource_word_features) if resource_word_features else {}
    fallback_cells = [key for key, result in cell_results.items() if result.fallback_triggered]
    llm_called_cells = [key for key, result in cell_results.items() if result.llm_called]

    resource_status = None
    if resource is not None:
        resource_status = {
            "en": {
                "resource": resource.resource_name("en"),
                "available": resource.resource_available("en"),
                "error": resource.resource_errors.get("en"),
            },
            "de": {
                "resource": resource.resource_name("de"),
                "available": resource.resource_available("de"),
                "error": resource.resource_errors.get("de"),
            },
        }

    return {
        "score": None if table_score is None else float(table_score),
        "reason": None,
        "warnings": list(dict.fromkeys(warnings)),
        "token_count": int(len(occurrences)),
        "unique_token_count": int(len(token_frequencies)),
        "cell_count": int(len(cell_results)),
        "token_scores": token_scores,
        "word_occurrence_scores": {key: asdict(value) for key, value in resource_word_features.items()},
        "llm_word_occurrence_scores": llm_word_occurrence_scores,
        "token_frequencies": dict(token_frequencies),
        "cell_scores": cell_scores,
        "lexical_cell_scores": {
            key: detail["features"].get("lexical_recognizability")
            for key, detail in cell_details.items()
            if isinstance(detail.get("features"), dict)
        },
        "cell_details": cell_details,
        "column_scores": column_scores,
        "row_scores": row_scores,
        "table_components": {
            "direct_cell_mean": table_score,
            "assessable_cell_count": len(cell_results),
            "successfully_scored_cell_count": sum(result.score is not None for result in cell_results.values()),
            "cell_assessment_coverage": (
                sum(result.score is not None for result in cell_results.values()) / len(cell_results)
                if cell_results else 0.0
            ),
            "context_lambda": float(context_lambda),
            "context_negative_evidence_threshold": float(context_negative_evidence_threshold),
            "context_positive_evidence_threshold": float(context_positive_evidence_threshold),
        },
        "aggregation": {
            "intrinsic_cell_score": "fixed-denominator mean over lexical_recognizability, semantic_ambiguity_score, notational_clarity, and lexical_processing_ease",
            "unavailable_criteria": "diagnostically unavailable but numerically represented by zero in the intrinsic score",
            "context_modifier": "mean of available row_context_support and column_context_support after centering each at 0.5",
            "resource_based_cell_score": "clip(fixed_denominator_intrinsic_score + context_lambda * context_modifier, 0, 1)",
            "llm_word_score": "one holistic contextual understandability score for each exact pre-tokenized word occurrence; no separate feature scores",
            "llm_cell_score": "independent holistic LLM assessment of the complete unchanged cell in its row and column context; not derived from word scores",
            "hybrid_cell_score": "resource-based score unless global uncertainty triggers complete replacement by the independent holistic LLM cell score",
            "table_score": "direct arithmetic mean over all assessable non-empty textual cell scores; each cell contributes once",
            "row_and_column_scores": "diagnostic data views only and not used to calculate the table score",
        },
        "pipeline_mode": mode,
        "pipeline_feature_parity": {
            "same_overarching_construct": True,
            "identical_information_sources": False,
            "resource_based_estimator": "direct lookup in configured WordNet resources, deterministic surface-form rules, and fixed multilingual cell embeddings",
            "llm_estimator": "contextual word-occurrence assessment over the shared pre-tokenized units plus an independent holistic assessment of the complete cell",
            "hybrid_estimator": "resource-based first; complete score replacement under dataset-independent uncertainty conditions",
        },
        "tokenization": "shared preprocessing before all pipelines; whitespace-delimited exact visible word occurrences with deterministic IDs",
        "original_spelling_preserved": True,
        "selected_columns": selected_columns,
        "ignored_numeric_columns": [str(column) for column in data.columns if str(column) not in selected_columns],
        "ignore_numeric_columns": ignore_numeric_columns,
        "features": {
            "lexical_recognizability": "direct evidence from Princeton WordNet and OdeNet queried without automatic language detection; any lexical match is recognized",
            "lexical_familiarity": "not calculated separately by the resource-based pipeline; considered holistically by the LLM",
            "semantic_ambiguity_score": "inverse-log transformation of the number of distinct synsets; synonym and hypernym counts are diagnostic only",
            "notational_clarity": "global structural rules distinguishing clear codes and segmentation from corruption and irregular symbol sequences",
            "lexical_processing_ease": "moderate length and syllable evidence for alphabetic components; unavailable for strongly code-like values",
            "row_context_support": "conservative cell-level embedding evidence; cosine values from -0.40 through 0.60 are neutral",
            "column_context_support": "conservative cell-level embedding evidence from a deterministic column sample; cosine values from -0.40 through 0.60 are neutral",
            "llm_understandability_score": "independent holistic cell-level score produced from the complete cell value and supplied row and column context",
        },
        "global_parameters": {
            "ambiguity_transform": "1 / (1 + ln(distinct_synset_count))",
            "context_lambda": float(context_lambda),
            "context_negative_evidence_threshold": float(context_negative_evidence_threshold),
            "context_positive_evidence_threshold": float(context_positive_evidence_threshold),
            "hybrid_minimum_available_criteria": int(hybrid_minimum_available_criteria),
            "hybrid_severe_notation_threshold": float(hybrid_severe_notation_threshold),
            "hybrid_low_lexical_threshold": float(hybrid_low_lexical_threshold),
            "hybrid_weak_context_threshold": float(hybrid_weak_context_threshold),
            "automatic_language_detection": False,
            "embedding_max_column_context_values": int(max_column_context_values),
            "target_user_group": str(target_user_group),
            "lexical_lookup_strategy": "query Princeton WordNet and OdeNet directly for each word-like token",
            "unavailable_intrinsic_value": 0.0,
        },
        "schema_excluded": True,
        "context_excluded": False,
        "embedding_enabled": embedding_enabled and mode in {"resource_based", "hybrid"},
        "embedding_model_id": embedding_model_id if mode in {"resource_based", "hybrid"} and embedding_enabled else None,
        "embedding_backend_loaded": embedding_backend is not None if mode in {"resource_based", "hybrid"} else False,
        "embedding_error": embedding_error,
        "embedding_cells_with_context": sum(
            1
            for detail in cell_details.values()
            if (detail.get("features") or {}).get("row_context_support") is not None
            or (detail.get("features") or {}).get("column_context_support") is not None
        ),
        "embedding_word_occurrences_with_context": 0,
        "embedding_max_column_context_values": int(max_column_context_values),
        "llm_context_aware_backend": llm_context_aware,
        "llm_requested_cell_count": len(set(llm_called_cells)),
        "llm_scored_cell_count": len([key for key in llm_called_cells if key in llm_scores_by_cell]),
        "llm_failed_cell_count": len([key for key in llm_called_cells if key not in llm_scores_by_cell]),
        "hybrid_fallback_cell_count": len(set(fallback_cells)),
        "language_hint": language_hint,
        "language_detector_available": False,
        "language_detector_error": None,
        "lexical_lookup_strategy": "direct lookup in all configured WordNet resources without automatic language detection",
        "lexical_resource_status": resource_status,
    }


def score_word_understandability(
    token: str,
    resource: Optional[LexicalResource] = None,
    pipeline_mode: str = "resource_based",
    backend: Any = None,
    language: Optional[str] = "en",
) -> Dict[str, Any]:
    """Context-free compatibility helper for one visible token."""

    lexical_resource = resource or LexicalResource()
    occurrence = TokenOccurrence(
        occurrence_id="single",
        token=str(token),
        column="",
        row_index=None,
        row_position=0,
        column_position=0,
        token_position=0,
        cell_value=str(token),
        cell_key="single::",
        language=language,
    )
    feature = score_resource_word(occurrence, lexical_resource)
    intrinsic = mean_available(
        [
            feature.lexical_recognizability,
            feature.semantic_ambiguity_score,
            feature.notational_clarity,
            feature.lexical_processing_ease,
        ]
    )
    return {
        **asdict(feature),
        "word_intrinsic_score": intrinsic,
        "pipeline_mode": validate_pipeline_mode(pipeline_mode),
        "final_metric_note": "The final metric assesses cells. This helper exposes context-free intrinsic evidence only.",
    }


# Deprecated alias used by an old diagnostic script. The final field name is
# lexical_processing_ease and the implementation no longer depends on WordNet.
def lexical_difficulty(token: str, resource: Optional[LexicalResource] = None) -> Optional[float]:
    return lexical_processing_ease(token, resource=resource, language="en")
