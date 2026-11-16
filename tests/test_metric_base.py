import pytest

from metis.metric.metric import Metric
from metis.metric.metric_meta import MetricMeta
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity


class _Described(Metric):
    meta = MetricMeta(
        label="Described",
        description="A metric that declares metadata.",
        dimension=DQDimension.COMPLETENESS,
        granularities=frozenset({DQGranularity.TABLE}),
    )

    def assess(self, data, metric_config=None):
        return []


class _Undescribed(Metric):
    def assess(self, data, metric_config=None):
        return []


def test_describe_returns_the_declared_meta():
    assert _Described.describe().label == "Described"


def test_describe_raises_when_meta_is_missing():
    with pytest.raises(NotImplementedError, match="_Undescribed"):
        _Undescribed.describe()


def test_base_class_meta_defaults_to_none():
    assert Metric.meta is None


def test_subclasses_are_still_registered():
    assert Metric.registry["_Described"] is _Described
