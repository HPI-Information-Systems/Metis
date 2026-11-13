import dataclasses

import pytest

from metis.metric.metric_meta import MetricMeta
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity


def _meta(**overrides):
    base = dict(
        label="Null Ratio",
        description="Share of non-null values.",
        dimension=DQDimension.COMPLETENESS,
        granularities=frozenset({DQGranularity.COLUMN}),
    )
    base.update(overrides)
    return MetricMeta(**base)


def test_defaults_are_the_zero_to_one_convention():
    meta = _meta()
    assert meta.value_range == (0.0, 1.0)
    assert meta.higher_is_better is True
    assert meta.standard is None
    assert meta.requires_reference is False
    assert meta.config_required is False
    assert meta.callable_config is False


def test_is_frozen():
    meta = _meta()
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.label = "Something else"


def test_cell_granularity_is_derived_from_granularities():
    assert _meta(granularities=frozenset({DQGranularity.CELL})).cell_granularity is True
    assert _meta(granularities=frozenset({DQGranularity.COLUMN})).cell_granularity is False


def test_empty_label_is_rejected():
    with pytest.raises(ValueError, match="label"):
        _meta(label="   ")


def test_empty_description_is_rejected():
    with pytest.raises(ValueError, match="description"):
        _meta(description="")


def test_inverted_value_range_is_rejected():
    with pytest.raises(ValueError, match="value_range"):
        _meta(value_range=(1.0, 0.0))


def test_granularities_accepts_a_plain_set_and_freezes_it():
    meta = _meta(granularities={DQGranularity.COLUMN, DQGranularity.TABLE})
    assert isinstance(meta.granularities, frozenset)
