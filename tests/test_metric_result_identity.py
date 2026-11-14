"""Every metric must identify its results by its registry name.

The GUI catalog, the metric palette and every per-metric filter key on the
registry name, so a metric that writes a different string into DQmetric
produces results that can never be traced back to it.
"""
import pytest

import metis.metric  # noqa: F401 - populates Metric.registry
from benchmarks import datagen, fixtures
from metis.metric.metric import Metric

_METRICS = sorted(
    n for n in Metric.registry
    if not n.startswith("_") and n not in fixtures.SKIPPED
)


@pytest.mark.parametrize("name", _METRICS)
def test_every_result_carries_the_registry_name(name, tmp_path):
    frame = datagen.make_frame(rows=20, cols=6)
    config = fixtures.config_for(name, frame, tmp_path)

    results = Metric.registry[name]().assess(frame, metric_config=config)

    assert results
    assert {r.DQmetric for r in results} == {name}
