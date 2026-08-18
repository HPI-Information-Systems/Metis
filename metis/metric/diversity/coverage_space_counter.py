"""Exact finite-domain DNF counting for MUP-induced pattern regions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import prod
from typing import Iterable

Literal = tuple[int, int]
Term = tuple[Literal, ...]
Terms = tuple[Term, ...]


@dataclass(frozen=True)
class CoverageSpaceScore:
    """Exact size and normalized score of an uncovered pattern space."""

    uncovered_patterns: int
    total_patterns: int
    effective_term_count: int

    @property
    def coverage_gap(self) -> float:
        return self.uncovered_patterns / self.total_patterns

    @property
    def coverage_space(self) -> float:
        return 1.0 - self.coverage_gap


class CoverageSpaceCounter:
    """Count the union of MUP specialization regions without materializing it.

    Each term is a conjunction of ``(attribute_index, value_index)`` literals.
    Terms are combined disjunctively. Every attribute has its concrete domain
    values plus one wildcard state, matching the pattern lattice definition in
    FLAPS.
    """

    def __init__(self, domain_sizes: Iterable[int]) -> None:
        self.domain_sizes = tuple(int(size) for size in domain_sizes)
        if not self.domain_sizes:
            raise ValueError("At least one diversity attribute is required.")
        if any(size <= 0 for size in self.domain_sizes):
            raise ValueError("Every diversity attribute must have a non-empty domain.")

        remaining = [1] * (len(self.domain_sizes) + 1)
        for index in range(len(self.domain_sizes) - 1, -1, -1):
            remaining[index] = remaining[index + 1] * (self.domain_sizes[index] + 1)
        self._remaining_products = tuple(remaining)

    def calculate(self, raw_terms: Iterable[Iterable[Literal]]) -> CoverageSpaceScore:
        """Return the exact uncovered and total pattern-space sizes."""
        terms = self._normalize_terms(raw_terms)

        @lru_cache(maxsize=None)
        def count(active_terms: Terms, next_attribute: int) -> int:
            if not active_terms:
                return 0
            if not active_terms[0]:
                return self._remaining_products[next_attribute]
            if next_attribute >= len(self.domain_sizes):
                return 0

            branching_attribute = min(
                attribute
                for term in active_terms
                for attribute, _ in term
                if attribute >= next_attribute
            )
            branch_count = count_with_attribute(active_terms, branching_attribute)
            if branching_attribute == next_attribute:
                return branch_count

            skipped_assignments = (
                self._remaining_products[next_attribute]
                // self._remaining_products[branching_attribute]
            )
            return skipped_assignments * branch_count

        def count_with_attribute(active_terms: Terms, attribute: int) -> int:
            unconstrained: list[Term] = []
            terms_by_value: dict[int, list[Term]] = {}

            for term in active_terms:
                literal_index = next(
                    (i for i, literal in enumerate(term) if literal[0] == attribute),
                    None,
                )
                if literal_index is None:
                    unconstrained.append(term)
                    continue

                _, value = term[literal_index]
                reduced = term[:literal_index] + term[literal_index + 1 :]
                terms_by_value.setdefault(value, []).append(reduced)

            base_terms = self._canonicalize(unconstrained)
            skipped_concrete_values = self.domain_sizes[attribute] - len(terms_by_value)
            result = (skipped_concrete_values + 1) * count(base_terms, attribute + 1)

            for value_terms in terms_by_value.values():
                combined = self._canonicalize([*unconstrained, *value_terms])
                result += count(combined, attribute + 1)
            return result

        uncovered = count(terms, 0)
        return CoverageSpaceScore(
            uncovered_patterns=uncovered,
            total_patterns=prod(size + 1 for size in self.domain_sizes),
            effective_term_count=len(terms),
        )

    def _normalize_terms(self, raw_terms: Iterable[Iterable[Literal]]) -> Terms:
        terms: list[Term] = []
        for raw_term in raw_terms:
            term = tuple(sorted((int(attribute), int(value)) for attribute, value in raw_term))
            previous_attribute = -1
            for attribute, value in term:
                if attribute < 0 or attribute >= len(self.domain_sizes):
                    raise ValueError(f"Invalid attribute index in MUP: {attribute}.")
                if value < 0 or value >= self.domain_sizes[attribute]:
                    raise ValueError(
                        f"Invalid value index {value} for attribute {attribute}."
                    )
                if attribute == previous_attribute:
                    raise ValueError(
                        f"A MUP constrains attribute {attribute} more than once."
                    )
                previous_attribute = attribute
            terms.append(term)
        return self._canonicalize(terms)

    @staticmethod
    def _canonicalize(raw_terms: Iterable[Term]) -> Terms:
        """Remove duplicate terms and terms subsumed by a general term."""
        terms: list[Term] = []
        for raw_term in raw_terms:
            term = tuple(raw_term)
            term_set = frozenset(term)
            if any(frozenset(existing).issubset(term_set) for existing in terms):
                continue
            terms = [
                existing
                for existing in terms
                if not term_set.issubset(frozenset(existing))
            ]
            terms.append(term)
        return tuple(sorted(terms))
