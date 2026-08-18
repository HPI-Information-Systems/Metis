"""MUP-based diversity assessment using exact finite-domain DNF counting."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

from metis.metric.config import MetricConfig
from metis.metric.diversity.coverage_space_counter import CoverageSpaceCounter, Literal
from metis.metric.diversity.diversity_coverageGap_config import (
    diversity_coverageGap_config,
)
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.result import DQResult


class diversity_coverageGap(Metric):
    """Calculate coverage gap from a supplied MUP frontier.

    The exact paper-defined coverage gap is included in ``DQexplanation``.
    ``DQvalue`` is its complement (coverage space) so the metric follows the
    existing METIS convention that larger DQ scores indicate better quality.
    """

    _gui_requires_reference: bool = False
    _gui_config_required: bool = True
    _gui_callable_config: bool = False
    _gui_recommended_granularities: frozenset = frozenset({DQGranularity.TABLE})
    _gui_description: str = (
        "Exact coverage-based diversity from Maximal Uncovered Patterns (MUPs). "
        "Counts the MUP-induced finite-domain DNF union. The METIS score is "
        "coverage space (1 − coverage gap); exact gap and counts are shown in details."
    )

    def assess(
        self,
        data: pd.DataFrame,
        reference: pd.DataFrame | None = None,
        metric_config: str | MetricConfig | None = None,
    ) -> list[DQResult]:
        config = self.load_config(metric_config or "", diversity_coverageGap_config)
        attributes = list(config.attributes or [])
        missing = [attribute for attribute in attributes if attribute not in data.columns]
        if missing:
            raise ValueError(f"Configured diversity attributes are missing: {missing}.")

        domains = [self._domain(data[attribute]) for attribute in attributes]
        domain_indexes = [
            {value: index for index, value in enumerate(domain)} for domain in domains
        ]
        text = self._read_mups(config)
        terms, raw_mup_count, mup_coverages, intermediate_field_count = self._parse_mups(
            text=text,
            attributes=attributes,
            domain_indexes=domain_indexes,
            wildcard=config.wildcard,
            delimiter=config.delimiter,
            mincov=config.mincov,
        )

        score = CoverageSpaceCounter(len(domain) for domain in domains).calculate(terms)
        source_name = config.mups_filename
        if source_name is None and config.mups_path:
            source_name = Path(config.mups_path).name

        explanation = {
            "coverage_gap": score.coverage_gap,
            "coverage_space": score.coverage_space,
            "uncovered_patterns": str(score.uncovered_patterns),
            "total_patterns": str(score.total_patterns),
            "mup_count": raw_mup_count,
            "mup_coverage_min": min(mup_coverages) if mup_coverages else None,
            "mup_coverage_max": max(mup_coverages) if mup_coverages else None,
            "mup_coverages_validated_against_mincov": config.mincov is not None,
            "effective_dnf_terms": score.effective_term_count,
            "dnf_method": "exact finite-domain DNF counting",
            "attributes": attributes,
            "domain_sizes": {
                attribute: len(domain)
                for attribute, domain in zip(attributes, domains)
            },
            "mincov": config.mincov,
            "mups_file": source_name,
            "ignored_intermediate_fields_per_mup": intermediate_field_count,
        }

        return [DQResult(
            timestamp=pd.Timestamp.now(),
            DQdimension=DQDimension.DIVERSITY,
            DQmetric=self.__class__.__name__,
            DQgranularity=DQGranularity.TABLE,
            DQvalue=score.coverage_space,
            DQexplanation=explanation,
            columnNames=attributes,
            configJson=config.to_json(),
        )]

    @staticmethod
    def _domain(series: pd.Series) -> tuple[str, ...]:
        values = tuple(dict.fromkeys(diversity_coverageGap._normalize_value(v) for v in series))
        if not values:
            raise ValueError(
                f"Diversity attribute '{series.name}' has no observed domain values."
            )
        return values

    @staticmethod
    def _normalize_value(value) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    @staticmethod
    def _read_mups(config: diversity_coverageGap_config) -> str:
        if config.mups_content is not None:
            return config.mups_content

        path = Path(config.mups_path or "")
        try:
            return path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1")
        except OSError as exc:
            raise ValueError(f"Could not read MUP file '{path}': {exc}") from exc

    @staticmethod
    def _parse_mups(
        text: str,
        attributes: list[str],
        domain_indexes: list[dict[str, int]],
        wildcard: str,
        delimiter: str,
        mincov: int | None,
    ) -> tuple[list[tuple[Literal, ...]], int, list[int], int | list[int]]:
        terms: list[tuple[Literal, ...]] = []
        mup_coverages: list[int] = []
        expected_fields = len(attributes)
        intermediate_counts: set[int] = set()

        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        for line_number, raw_fields in enumerate(reader, start=1):
            if not raw_fields or all(not field.strip() for field in raw_fields):
                continue
            if raw_fields[0].lstrip().startswith("#"):
                continue
            if len(raw_fields) < expected_fields + 1:
                raise ValueError(
                    f"MUP line {line_number} has {len(raw_fields)} fields; "
                    f"{expected_fields} pattern fields plus the final MUP coverage "
                    "field are required."
                )

            intermediate_counts.add(len(raw_fields) - expected_fields - 1)
            term: list[Literal] = []
            for attribute_index, attribute in enumerate(attributes):
                value = raw_fields[attribute_index].strip()
                if value == wildcard:
                    continue
                normalized = diversity_coverageGap._normalize_value(value)
                value_index = domain_indexes[attribute_index].get(normalized)
                if value_index is None:
                    raise ValueError(
                        f"Unknown value '{value}' for diversity attribute "
                        f"'{attribute}' on MUP line {line_number}."
                    )
                term.append((attribute_index, value_index))
            terms.append(tuple(term))

            coverage_raw = raw_fields[-1].strip()
            try:
                coverage = int(coverage_raw)
            except ValueError as exc:
                raise ValueError(
                    f"MUP coverage '{coverage_raw}' on line {line_number} is not an integer."
                ) from exc
            if coverage < 0:
                raise ValueError(
                    f"MUP coverage on line {line_number} must be non-negative."
                )
            if mincov is not None and coverage >= mincov:
                raise ValueError(
                    f"MUP coverage {coverage} on line {line_number} is not below "
                    f"mincov {mincov}."
                )
            mup_coverages.append(coverage)

        intermediate_field_count: int | list[int]
        if not intermediate_counts:
            intermediate_field_count = 0
        elif len(intermediate_counts) == 1:
            intermediate_field_count = next(iter(intermediate_counts))
        else:
            intermediate_field_count = sorted(intermediate_counts)
        return terms, len(terms), mup_coverages, intermediate_field_count
