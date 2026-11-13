"""Unit tests for completeness_nullRatio.

This file is the template for testing a metric. Copy it to
``tests/metric/<dimension>/test_<metric_name>.py`` and adapt it. Each test
builds a tiny table whose score can be worked out by hand, runs the metric on
it and compares the result.
"""
import pandas as pd
import pytest

from metis.metric.completeness.completeness_nullRatio import completeness_nullRatio
from metis.metric.completeness.completeness_nullRatio_config import (
    completeness_nullRatio_config,
)
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity


@pytest.fixture
def data():
    """Four rows. ``name`` misses one value, ``age`` misses two."""
    return pd.DataFrame({
        "name": ["Ada", None, "Grace", "Linus"],
        "age": [36, 41, None, None],
    })


def test_default_reports_the_filled_share_of_each_column(data):
    results = completeness_nullRatio().assess(
        data, metric_config=completeness_nullRatio_config()
    )
    assert {r.columnNames[0]: r.DQvalue for r in results} == {
        "name": pytest.approx(0.75),
        "age": pytest.approx(0.5),
    }
    assert all(r.DQgranularity == DQGranularity.COLUMN for r in results)


def test_results_identify_the_metric_and_its_dimension(data):
    results = completeness_nullRatio().assess(
        data, metric_config=completeness_nullRatio_config()
    )
    assert all(r.DQmetric == "completeness_nullRatio" for r in results)
    assert all(r.DQdimension == DQDimension.COMPLETENESS for r in results)


def test_row_aggregation_reports_the_filled_share_of_each_row(data):
    results = completeness_nullRatio().assess(
        data, metric_config=completeness_nullRatio_config(aggregation_axis="columns")
    )
    assert {r.rowIndex: r.DQvalue for r in results} == {
        0: pytest.approx(1.0),
        1: pytest.approx(0.5),
        2: pytest.approx(0.5),
        3: pytest.approx(0.5),
    }


def test_without_aggregation_every_cell_scores_zero_or_one(data):
    results = completeness_nullRatio().assess(
        data, metric_config=completeness_nullRatio_config(aggregation_axis=None)
    )
    by_cell = {(r.columnNames[0], r.rowIndex): r.DQvalue for r in results}
    assert len(by_cell) == data.size
    assert by_cell[("name", 0)] == 1.0
    assert by_cell[("name", 1)] == 0.0


def test_aggregate_all_reports_one_score_for_the_table(data):
    results = completeness_nullRatio().assess(
        data, metric_config=completeness_nullRatio_config(aggregate_all=True)
    )
    assert len(results) == 1
    assert results[0].DQvalue == pytest.approx(5 / 8)
    assert results[0].DQgranularity == DQGranularity.TABLE


def test_a_table_without_missing_values_scores_one():
    full = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    results = completeness_nullRatio().assess(
        full, metric_config=completeness_nullRatio_config()
    )
    assert all(r.DQvalue == pytest.approx(1.0) for r in results)


def test_an_invalid_config_is_rejected(data):
    with pytest.raises(ValueError, match="aggregation_axis"):
        completeness_nullRatio().assess(
            data, metric_config=completeness_nullRatio_config(aggregation_axis="rows")
        )


def test_the_input_data_is_not_modified(data):
    before = data.copy()
    completeness_nullRatio().assess(data, metric_config=completeness_nullRatio_config())
    pd.testing.assert_frame_equal(data, before)
