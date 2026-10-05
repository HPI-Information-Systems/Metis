"""Declarative metadata attached to every :class:`~metis.metric.metric.Metric`."""
from __future__ import annotations

from dataclasses import dataclass, field

from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity


@dataclass(frozen=True)
class MetricMeta:
    """Describes what a metric measures, how it reports, and what it needs.

    Every registered metric declares one of these as a ``meta`` class
    attribute. It is the single source of truth for the metric catalog, the
    GUI, and generated documentation.

    :param label: Human-readable display name without the dimension prefix
        (e.g. ``"Null Ratio"`` for ``completeness_nullRatio``). Callers that
        want a qualified name compose it as ``f"{dimension}: {label}"``.
    :param description: What the metric measures and how it is computed.
    :param dimension: The data quality dimension this metric belongs to.
    :param granularities: Granularities the metric produces meaningful
        results at.
    :param requires_reference: The metric cannot run without reference data
        supplied in its config.
    :param config_required: The metric refuses to run without a config.
    :param callable_config: The config carries Python callables and needs the
        callable editor rather than the generic one.
    :param standard: Citation of the external standard this metric implements,
        e.g. ``"ISO/IEC 25024:2015 Acc-I-4"``. ``None`` when the metric is not
        derived from a published standard.
    :param value_range: Inclusive bounds of ``DQResult.DQvalue``. Metis
        normalises every metric to ``(0.0, 1.0)``.
    :param higher_is_better: Whether a larger value means better quality.
        Metrics whose natural formula runs the other way invert before
        reporting, so this is ``True`` throughout Metis today.
    """

    label: str
    description: str
    dimension: DQDimension
    granularities: frozenset[DQGranularity] = field(default_factory=frozenset)
    requires_reference: bool = False
    config_required: bool = False
    callable_config: bool = False
    standard: str | None = None
    value_range: tuple[float, float] = (0.0, 1.0)
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if not self.label or not self.label.strip():
            raise ValueError("MetricMeta.label must be a non-empty string.")
        if not self.description or not self.description.strip():
            raise ValueError("MetricMeta.description must be a non-empty string.")
        low, high = self.value_range
        if low >= high:
            raise ValueError(
                f"MetricMeta.value_range must be increasing, got {self.value_range}."
            )
        # Accept a plain set for ergonomics, but stay hashable.
        object.__setattr__(self, "granularities", frozenset(self.granularities))

    @property
    def cell_granularity(self) -> bool:
        """Whether this metric can emit one result per cell."""
        return DQGranularity.CELL in self.granularities
