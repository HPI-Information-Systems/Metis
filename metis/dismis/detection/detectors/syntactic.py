import time
from collections.abc import Sequence
from functools import lru_cache
from itertools import chain, combinations
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

from metis.dismis.detection.detectors.detector import DMVDetector
from metis.dismis.utils.types import COLUMN_TYPES

# The algorithm for this detector is taken from https://github.com/HPI-Information-Systems/SURAGH

# ---------Abstractions for single characters---------#


class SingleCharAbstraction:
    """Base class for abstractions that replace single characters with a symbol."""

    name = "SingleCharAbstraction"
    symbol: str
    characters: set[str]

    def __call__(self, data: Sequence[str]) -> List[str]:
        output: List[str] = []
        for token in data:
            if isinstance(token, str) and len(token) == 1 and self.matches(token):
                output.append(self.symbol)
            else:
                output.append(token)
        return output

    def matches(self, char: str) -> bool:
        return char in self.characters


class DelimiterAbstraction(SingleCharAbstraction):
    name = "DelimiterAbstraction"
    symbol = "<DEL>"
    characters = {",", ";", ":", "|", "\t"}


class UpperCaseLetterAbstraction(SingleCharAbstraction):
    name = "UpperCaseLetterAbstraction"
    symbol = "<UL>"
    characters = {chr(code) for code in range(ord("A"), ord("Z") + 1)}


class LowerCaseLetterAbstraction(SingleCharAbstraction):
    name = "LowerCaseLetterAbstraction"
    symbol = "<LL>"
    characters = {chr(code) for code in range(ord("a"), ord("z") + 1)}


class DigitAbstraction(SingleCharAbstraction):
    name = "DigitAbstraction"
    symbol = "<D>"
    characters = {str(d) for d in range(10)}


class SpaceAbstraction(SingleCharAbstraction):
    name = "SpaceAbstraction"
    symbol = "<S>"
    characters = {" "}


class QuotationAbstraction(SingleCharAbstraction):
    name = "QuotationAbstraction"
    symbol = "<QUOTE>"
    characters = {'"', "'", "\u2019", "\u2018"}


class ArithmeticAbstraction(SingleCharAbstraction):
    name = "ArithmeticAbstraction"
    symbol = "<ARITH>"
    characters = {"*", "+", "-", "/", "%", "=", "<", ">"}


class BracketAbstraction(SingleCharAbstraction):
    name = "BracketAbstraction"
    symbol = "<BRKT>"
    characters = {"[", "]", "{", "}", "(", ")"}


class SymbolAbstraction(SingleCharAbstraction):
    name = "SymbolAbstraction"
    symbol = "<SYM>"
    characters = {"$", "#", ".", "?", "@", "\\", "^", "~", "_", ",", "&", "!"}


class LineBreakAbstraction(SingleCharAbstraction):
    name = "LineBreakAbstraction"
    symbol = "<LB>"
    characters = {"\n", "\r"}


class EmptyValueAbstraction:
    name = "EmptyValueAbstraction"
    symbol = "<EV>"

    def __call__(self, data: Sequence[str]) -> List[str]:
        output: List[str] = []
        for token in data:
            output.append(self.symbol if token is None else token)
        return output


class FulltextAbstraction:
    name = "FulltextAbstraction"
    symbol = "<FTXT>"

    def __call__(self, data: Sequence[str]) -> List[str]:
        return [self.symbol]


# ---------Abstractions for multiple characters---------#


class TokenRunAbstraction:
    """Aggregate runs of a specific token into a higher-level abstraction."""

    name = "TokenRunAbstraction"
    symbol: str
    target_token: str
    min_run_length: int = 1

    def __init__(self, min_run_length: int | None = None) -> None:
        if min_run_length is not None:
            if min_run_length < 1:
                raise ValueError("min_run_length must be at least 1.")
            self.min_run_length = min_run_length

    def __call__(self, data: Sequence[str]) -> List[str]:
        output: List[str] = []
        i = 0
        while i < len(data):
            token = data[i]
            if token == self.target_token:
                run_end = i
                while run_end < len(data) and data[run_end] == self.target_token:
                    run_end += 1
                run_length = run_end - i
                if run_length >= self.min_run_length:
                    output.append(self.symbol)
                else:
                    output.extend(data[i:run_end])
                i = run_end
            else:
                output.append(token)
                i += 1
        return output


class UpperCaseSequenceAbstraction(TokenRunAbstraction):
    name = "UpperCaseSequenceAbstraction"
    symbol = "<SEQU>"
    target_token = "<UL>"
    min_run_length = 1


class LowerCaseSequenceAbstraction(TokenRunAbstraction):
    name = "LowerCaseSequenceAbstraction"
    symbol = "<SEQLL>"
    target_token = "<LL>"
    min_run_length = 1


class DigitSequenceAbstraction(TokenRunAbstraction):
    name = "DigitSequenceAbstraction"
    symbol = "<SEQD>"
    target_token = "<D>"
    min_run_length = 1


class WhitespaceSequenceAbstraction(TokenRunAbstraction):
    name = "WhitespaceSequenceAbstraction"
    symbol = "<WS>"
    target_token = "<S>"
    min_run_length = 1


def _expand_token_pattern(spec: str) -> List[str]:
    tokens: List[str] = []
    for raw_part in spec.split("|"):
        part = raw_part.strip()
        if not part:
            continue
        if "*<" in part:
            count_text, token = part.split("*", 1)
            tokens.extend([token.strip()] * int(count_text))
        else:
            tokens.append(part)
    return tokens


class DateAbstraction:
    name = "DateAbstraction"
    symbol = "<DT>"
    _pattern_specs = [
        "2*<D>| <ARITH>| 2*<D>| <ARITH>| 4*<D>",
        "2*<D>| <ARITH>| 2*<D>| <ARITH>| 2*<D>",
        "4*<D>| <ARITH>| 2*<D>| <ARITH>| 2*<D>",
        "<D>| <ARITH>| <D>| <ARITH>| 4*<D>",
        "<D>| <ARITH>| <D>| <ARITH>| 2*<D>",
        "<D>| <ARITH>| 2*<D>| <ARITH>| 2*<D>",
        "<D>| <ARITH>| 2*<D>| <ARITH>| 4*<D>",
        "4*<D>| <ARITH>| <D>| <ARITH>| <D>",
        "4*<D>| <ARITH>| 2*<D>| <ARITH>| <D>",
        "4*<D>| <ARITH>| <D>| <ARITH>| 2*<D>",
    ]
    _patterns = tuple(
        sorted(
            (_expand_token_pattern(spec) for spec in _pattern_specs),
            key=len,
            reverse=True,
        )
    )

    def __call__(self, data: Sequence[str]) -> List[str]:
        output: List[str] = []
        i = 0
        while i < len(data):
            match_length = 0
            for pattern in self._patterns:
                length = len(pattern)
                if data[i : i + length] == pattern:
                    match_length = length
                    break
            if match_length:
                output.append(self.symbol)
                i += match_length
            else:
                output.append(data[i])
                i += 1
        return output


# ---------Third Level Abstractions---------#

_LETTER_TOKENS = {"<UL>", "<SEQU>", "<LL>", "<SEQLL>"}

_TEXT_BODY_SYMBOLS = _LETTER_TOKENS | {
    "<S>",
    "<WS>",
    "-",
    "_",
    "'",
    '"',
    "/",
    ".",
    "|",
    "&",
}

_WHITESPACE_TOKENS = {"<S>", "<WS>"}

_MISSING_WORDS = {"null", "na", "n/a", "nan", "none"}

_MISSING_PREFIXES = {word[:i] for word in _MISSING_WORDS for i in range(1, len(word))}


def _is_letter_token(token: str) -> bool:
    if not isinstance(token, str):
        return False
    if token in _LETTER_TOKENS:
        return True
    if token.isalpha():
        return True
    return False


def _is_text_body_token(token: str) -> bool:
    if not isinstance(token, str):
        return False
    if token in _TEXT_BODY_SYMBOLS:
        return True
    if _is_whitespace_token(token):
        return True
    if len(token) == 1 and token.isalpha():
        return True
    if token.isalpha():
        return True
    return False


def _is_whitespace_token(token: str) -> bool:
    if not isinstance(token, str):
        return False
    if token in _WHITESPACE_TOKENS:
        return True
    if token.strip() == "":
        return True
    return token in {"\t", "\n", "\r"}


class NumberAbstraction:
    name = "NumberAbstraction"
    symbol = "<NUM>"
    _pattern_specs = [
        "+",
        "-",
        "<D>",
        "<SEQD>",
        "+| <D>",
        "-| <D>",
        "<D>| ,| <D>",
        "<SEQD>| ,| <D>",
        "<D>| ,| <SEQD>",
        "<SEQD>| ,| <SEQD>",
        "+| <D>| ,| <D>",
        "+| <SEQD>| ,| <D>",
        "+| <D>| ,| <SEQD>",
        "+| <SEQD>| ,| <SEQD>",
        "-| <D>| ,| <D>",
        "-| <SEQD>| ,| <D>",
        "-| <D>| ,| <SEQD>",
        "-| <SEQD>| ,| <SEQD>",
        "<D>| .| <D>",
        "<SEQD>| .| <D>",
        "<D>| .| <SEQD>",
        "<SEQD>| .| <SEQD>",
        "+| <D>| .| <D>",
        "+| <SEQD>| .| <D>",
        "+| <D>| .| <SEQD>",
        "+| <SEQD>| .| <SEQD>",
        "-| <D>| .| <D>",
        "-| <SEQD>| .| <D>",
        "-| <D>| .| <SEQD>",
        "-| <SEQD>| .| <SEQD>",
    ]
    _patterns = tuple(
        sorted(
            (_expand_token_pattern(spec) for spec in _pattern_specs),
            key=len,
            reverse=True,
        )
    )

    def __call__(self, data: Sequence[str]) -> List[str]:
        output: List[str] = []
        i = 0
        while i < len(data):
            match_length = 0
            for pattern in self._patterns:
                length = len(pattern)
                if data[i : i + length] == pattern:
                    match_length = length
                    break
            if match_length:
                output.append(self.symbol)
                i += match_length
            else:
                output.append(data[i])
                i += 1
        return output


class TextAbstraction:
    name = "TextAbstraction"
    symbol = "<TXT>"

    def __call__(self, data: Sequence[str]) -> List[str]:
        output: List[str] = []
        i = 0
        while i < len(data):
            match_length = self._match_text(data, i)
            if match_length:
                output.append(self.symbol)
                i += match_length
            else:
                output.append(data[i])
                i += 1
        return output

    def _match_text(self, data: Sequence[str], start: int) -> int:
        i = start
        end = len(data)
        if i >= end:
            return 0
        if not _is_letter_token(data[i]):
            return 0
        i += 1
        while i < end and _is_text_body_token(data[i]):
            i += 1
        return i - start


class MissingValueAbstraction:
    name = "MissingValueAbstraction"
    symbol = "<MV>"

    def __call__(self, data: Sequence[str]) -> List[str]:
        output: List[str] = []
        i = 0
        while i < len(data):
            match_length = self._match_missing(data, i)
            if match_length:
                output.append(self.symbol)
                i += match_length
            else:
                output.append(data[i])
                i += 1
        return output

    def _match_missing(self, data: Sequence[str], start: int) -> int:
        token = data[start]
        if token == "<EV>" or token is None:
            return 1
        if isinstance(token, str):
            trimmed = token.strip()
            if trimmed == "":
                return 1
            if trimmed.casefold() in _MISSING_WORDS:
                return 1
            if token == "\t":
                return 1
        builder: List[str] = []
        i = start
        end = len(data)
        while i < end:
            current = data[i]
            if not isinstance(current, str):
                break
            if len(current) == 1 and (current.isalpha() or current == "/"):
                builder.append(current)
                candidate = "".join(builder).casefold()
                if candidate in _MISSING_WORDS:
                    return len(builder)
                if candidate not in _MISSING_PREFIXES:
                    break
                i += 1
                continue
            break
        return 0


# ---------Manage Abstraction Dependencies---------#

_ABSTRACTION_DEPENDENCIES = {
    DateAbstraction.name: {DigitAbstraction.name, ArithmeticAbstraction.name},
    DigitSequenceAbstraction.name: {DigitAbstraction.name},
    UpperCaseSequenceAbstraction.name: {UpperCaseLetterAbstraction.name},
    LowerCaseSequenceAbstraction.name: {LowerCaseLetterAbstraction.name},
    WhitespaceSequenceAbstraction.name: {SpaceAbstraction.name},
    NumberAbstraction.name: {DigitSequenceAbstraction.name},
    TextAbstraction.name: {
        UpperCaseSequenceAbstraction.name,
        LowerCaseSequenceAbstraction.name,
    },
    MissingValueAbstraction.name: {
        EmptyValueAbstraction.name,
        WhitespaceSequenceAbstraction.name,
    },
}


@lru_cache(maxsize=None)
def _all_dependencies(name: str) -> frozenset[str]:
    deps = set(_ABSTRACTION_DEPENDENCIES.get(name, ()))
    for dep in list(deps):
        deps.update(_all_dependencies(dep))
    return frozenset(deps)


def is_valid_abstraction_sequence(sequence: Sequence[type]) -> bool:
    """
    Return True when dependency constraints are satisfied for the given abstraction order.

    A sequence is valid if for every abstraction in it, all of its direct
    dependencies are also present in the sequence *before* it.
    """
    if not sequence:
        return True

    present_abstractions = {abstraction.name for abstraction in sequence}

    for i, abstraction in enumerate(sequence):
        # Check for direct dependencies
        dependencies = _ABSTRACTION_DEPENDENCIES.get(abstraction.name, set())
        if not dependencies.issubset(present_abstractions):
            return False

        # Check that dependencies appear before the current abstraction
        # We only need to check against the part of the sequence we've already seen.
        abstractions_before = {sequence[j].name for j in range(i)}
        if not dependencies.issubset(abstractions_before):
            return False

    return True


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


_STAGE_ONE_CLASSES = [
    EmptyValueAbstraction,
    UpperCaseLetterAbstraction,
    LowerCaseLetterAbstraction,
    DigitAbstraction,
    DelimiterAbstraction,
    SpaceAbstraction,
    QuotationAbstraction,
    ArithmeticAbstraction,
    BracketAbstraction,
    SymbolAbstraction,
    LineBreakAbstraction,
]

_STAGE_TWO_CLASSES = [
    DateAbstraction,
    UpperCaseSequenceAbstraction,
    LowerCaseSequenceAbstraction,
    DigitSequenceAbstraction,
    WhitespaceSequenceAbstraction,
]

_STAGE_THREE_CLASSES = [
    NumberAbstraction,
    TextAbstraction,
    MissingValueAbstraction,
]

_STAGE_FOUR_CLASSES = [FulltextAbstraction]

_ALL_TRANSFORMS = {
    cls.name: cls()
    for cls in _STAGE_ONE_CLASSES
    + _STAGE_TWO_CLASSES
    + _STAGE_THREE_CLASSES
    + _STAGE_FOUR_CLASSES
}


@lru_cache(maxsize=1024)
def applicable_abstractions(value: str) -> list[object]:
    """Return abstraction instances that meaningfully apply to ``value``."""
    if not isinstance(value, str):
        raise TypeError("value must be a string")

    if len(value) > 50:
        return [FulltextAbstraction]

    applicable: list[type] = []
    seen: set[type] = set()

    def record(cls: type) -> None:
        if cls not in seen:
            seen.add(cls)
            applicable.append(cls)

    tokens = list(value)
    for cls in _STAGE_ONE_CLASSES:
        transform = _ALL_TRANSFORMS[cls.name]
        new_tokens = transform(tokens)
        if new_tokens != tokens:
            record(cls)
        tokens = new_tokens

    stage_one_tokens = tokens
    stage_two_tokens = stage_one_tokens
    for cls in _STAGE_TWO_CLASSES:
        transform = _ALL_TRANSFORMS[cls.name]
        detection_output = transform(stage_one_tokens)
        if detection_output != stage_one_tokens:
            record(cls)
        stage_two_tokens = transform(stage_two_tokens)

    possible_transform_lists = [
        combo for combo in powerset(applicable) if is_valid_abstraction_sequence(combo)
    ]

    # Cache results of applying transform lists
    memo = {(): list(value)}
    # Ensure we process prefixes first, sort by length
    possible_transform_lists.sort(key=len)

    for transform_list in possible_transform_lists:
        # Build upon the result of the prefix
        prefix = transform_list[:-1]
        last_cls = transform_list[-1]

        # Prefixes are guaranteed to be in memo because of the sort
        base_tokens = memo[prefix]

        transform = _ALL_TRANSFORMS[last_cls.name]
        memo[transform_list] = transform(base_tokens)

    for cls in _STAGE_THREE_CLASSES:
        transform = _ALL_TRANSFORMS[cls.name]
        for transform_list in possible_transform_lists:
            tokens = memo[transform_list]
            new_tokens = transform(tokens)
            if new_tokens != tokens:
                record(cls)
                break  # Found it's applicable, no need to check other lists for this class

    return [cls for cls in applicable]


# ---------Pattern generation---------#

weight_1 = ["<FTXT>"]
weight_2 = ["<NUM>", "<TXT>", "<MV>"]
weight_3 = ["<DT>", "<SEQD>", "<SEQU>", "<SEQLL>", "<WS>"]
weight_4 = [
    "<ARITH>",
    "<D>",
    "<SYM>",
    "<DEL>",
    "<QUOTE>",
    "<BRKT>",
    "<S>",
    "<LB>",
    "<EV>",
    "<UL>",
    "<LL>",
]

_WEIGHT_GROUPS = [
    (1, set(weight_1)),
    (2, set(weight_2)),
    (3, set(weight_3)),
    (4, set(weight_4)),
]


def _average_weight(tokens: Tuple[str, ...]) -> float:
    if not tokens:
        return 0.0
    symbol_weights: list[int] = []
    for symbol in tokens:
        assigned = False
        for weight, group in _WEIGHT_GROUPS:
            if symbol in group:
                symbol_weights.append(weight)
                assigned = True
                break
        if not assigned:
            symbol_weights.append(5)
    return sum(symbol_weights) / len(symbol_weights)


def _record_pattern(
    tokens: Iterable[str],
    transform_sequence: Tuple[type, ...],
    patterns_seen: set[Tuple[str, ...]],
    occurences: dict[Tuple[str, ...], int],
    weights: dict[Tuple[str, ...], float],
    pattern_sequences: dict[Tuple[str, ...], set[Tuple[type, ...]]],
    pattern_values: dict[Tuple[str, ...], set[str]],
    value_pattern_sets: dict[str, set[Tuple[str, ...]]],
    value: str,
    value_count: int,
) -> None:
    tokens_tuple = tuple(tokens)
    sequence_tuple = tuple(transform_sequence)
    if tokens_tuple not in patterns_seen:
        patterns_seen.add(tokens_tuple)
        weights[tokens_tuple] = _average_weight(tokens_tuple)
        pattern_sequences[tokens_tuple] = {sequence_tuple}
    else:
        pattern_sequences.setdefault(tokens_tuple, set()).add(sequence_tuple)

    values_for_pattern = pattern_values.setdefault(tokens_tuple, set())
    if value not in values_for_pattern:
        values_for_pattern.add(value)
        occurences[tokens_tuple] = occurences.get(tokens_tuple, 0) + value_count

    value_pattern_sets.setdefault(value, set()).add(tokens_tuple)


def generate_syntactic_value_patterns(column: List[str]) -> Tuple[
    Set[Tuple[str, ...]],
    Dict[Tuple[str, ...], int],
    Dict[Tuple[str, ...], float],
    Dict[Tuple[str, ...], Set[Tuple[type, ...]]],
    Dict[str, Set[Tuple[str, ...]]],
]:
    """Generate syntactic patterns for all values in a column.

    Returns
    -------
    patterns:
        Set of all unique token patterns observed.
    occurences:
        Mapping from pattern to count of distinct column values that produced it.
    weights:
        Average symbol weight per pattern.
    pattern_sequences:
        For each pattern, set of abstraction sequences that yielded it.
    value_patterns:
        For each distinct input value, set of patterns that the value can yield.
    """
    patterns_seen: Set[Tuple[str, ...]] = set()
    weights: Dict[Tuple[str, ...], float] = {}
    occurences: Dict[Tuple[str, ...], int] = {}
    pattern_sequences: Dict[Tuple[str, ...], Set[Tuple[type, ...]]] = {}
    pattern_values: Dict[Tuple[str, ...], Set[str]] = {}
    value_pattern_sets: Dict[str, Set[Tuple[str, ...]]] = {}

    values, counts = np.unique(column, return_counts=True)

    for value, value_count in zip(values, counts):
        raw_tokens = list(value)
        _record_pattern(
            raw_tokens,
            tuple(),
            patterns_seen,
            occurences,
            weights,
            pattern_sequences,
            pattern_values,
            value_pattern_sets,
            value,
            value_count,
        )

        abstractions = applicable_abstractions(value)

        abstraction_lists = [
            combo
            for combo in powerset(abstractions)
            if is_valid_abstraction_sequence(combo)
        ]
        abstraction_lists.sort(key=len)

        memo = {(): list(value)}
        for abstraction_sequence in abstraction_lists:
            prefix = abstraction_sequence[:-1]
            last_cls = abstraction_sequence[-1]

            base_tokens = memo[prefix]

            transform = _ALL_TRANSFORMS[last_cls.name]
            new_tokens = transform(base_tokens)
            memo[abstraction_sequence] = new_tokens

            _record_pattern(
                new_tokens,
                abstraction_sequence,
                patterns_seen,
                occurences,
                weights,
                pattern_sequences,
                pattern_values,
                value_pattern_sets,
                value,
                value_count,
            )

    return (
        patterns_seen,
        occurences,
        weights,
        pattern_sequences,
        value_pattern_sets,
    )


def _apply_transform_sequence(
    tokens: Tuple[str, ...], sequence: Tuple[type, ...]
) -> Tuple[str, ...]:
    result = list(tokens)
    for abstraction_cls in sequence:
        transform = _ALL_TRANSFORMS[abstraction_cls.name]
        result = transform(result)
    return tuple(result)


def _is_generalized_version(
    base_pattern: Tuple[str, ...],
    candidate_pattern: Tuple[str, ...],
    pattern_sequences: dict[Tuple[str, ...], set[Tuple[type, ...]]],
) -> bool:
    if base_pattern == candidate_pattern:
        return False
    candidate_sequences = pattern_sequences.get(candidate_pattern)
    if not candidate_sequences:
        return False
    for sequence in candidate_sequences:
        if not sequence:
            continue
        transformed = _apply_transform_sequence(base_pattern, sequence)
        if transformed == candidate_pattern:
            return True
    return False


def prune_generalized_patterns(
    sorted_patterns: List[Tuple[str, ...]],
    pattern_sequences: dict[Tuple[str, ...], set[Tuple[type, ...]]],
) -> List[Tuple[str, ...]]:
    """Remove generalized patterns that appear after more specific ones in the sorted sequence."""
    kept: list[Tuple[str, ...]] = []
    removed: set[Tuple[str, ...]] = set()
    for index, current in enumerate(sorted_patterns):
        if current in removed:
            continue
        kept.append(current)
        for subsequent in sorted_patterns[index + 1 :]:
            if subsequent in removed:
                continue
            if _is_generalized_version(current, subsequent, pattern_sequences):
                removed.add(subsequent)
    return kept


# ---------DisMis Detector Class---------#


class SyntacticDetector(DMVDetector):
    def __init__(
        self,
        target_types: List[str] = ["numeric", "categorical", "date", "text"],
        coverage_threshold: float = 0.2,
    ):
        self.target_types = target_types
        self.coverage_threshold = coverage_threshold

    def __call__(
        self,
        dataset: pd.DataFrame,
        column_types: Dict[str, COLUMN_TYPES],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:

        times: Dict[str, float] = {
            "pattern_generation": 0,
            "scoring": 0,
        }

        total_starttime = time.time()

        df_predict = pd.DataFrame(0, index=dataset.index, columns=dataset.columns)
        assessed = []

        for idx, column in enumerate(dataset.columns):
            if target_columns is not None and column not in target_columns:
                continue
            if column_types.get(column) not in self.target_types:
                continue
            assessed.append(column)

            pattern_generation_starttime = time.time()
            values = dataset[column].astype(str).tolist()
            (
                patterns,
                occurences,
                weighs,
                pattern_sequences,
                value_patterns,
            ) = generate_syntactic_value_patterns(values)
            times["pattern_generation"] += time.time() - pattern_generation_starttime

            scoring_starttime = time.time()
            sorted_patterns = sorted(
                patterns, key=lambda p: weighs[p] * occurences[p], reverse=True
            )
            selected_patterns = [
                p for p in sorted_patterns if occurences[p] / len(values) > 0.2
            ]
            pruned_patterns = prune_generalized_patterns(
                selected_patterns, pattern_sequences
            )
            not_represented = [
                not any(
                    pattern in value_patterns.get(value, [])
                    for pattern in pruned_patterns
                )
                for value in values
            ]
            df_predict.iloc[:, idx] = np.array(not_represented).astype(int)
            times["scoring"] += time.time() - scoring_starttime

        times["total"] = time.time() - total_starttime

        return df_predict.copy().astype(float), df_predict.astype(int), times, assessed
