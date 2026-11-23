"""Bad input to the orchestrator and the CSV loader is declined with a clear message."""
import json
import logging

import pandas as pd
import pytest

import metis.globals
import metis.metric  # noqa: F401 - populates Metric.registry
from metis.dq_orchestrator import DQOrchestrator
from metis.metric.metric import Metric
from metis.utils.dataset_checks import DatasetLoadError


@pytest.fixture
def orchestrator():
    orchestrator = DQOrchestrator()
    orchestrator.dataframes["Sample"] = pd.DataFrame({"a": [1, None, 3], "b": [1, 1, 2]})
    orchestrator.data_paths["Sample"] = "sample.json"
    return orchestrator


@pytest.fixture
def assessed(monkeypatch):
    calls = []
    monkeypatch.setattr(
        Metric.registry["completeness_nullRatio"], "assess",
        lambda self, data, *, metric_config=None: calls.append(data) or [],
    )
    return calls


def test_fewer_configs_than_metrics_is_declined(orchestrator, assessed):
    with pytest.raises(ValueError, match=r"Got 2 metric\(s\) but 1 config\(s\)") as e:
        orchestrator.assess(
            metrics=["completeness_nullRatio", "minimality_duplicateCount"],
            metric_configs=[None],
        )
    assert "one config per metric, in the same order" in str(e.value)
    assert "None" in str(e.value)
    assert not assessed


def test_more_configs_than_metrics_is_declined(orchestrator):
    with pytest.raises(ValueError, match=r"Got 1 metric\(s\) but 2 config\(s\)"):
        orchestrator.assess(metrics=["completeness_nullRatio"], metric_configs=[None, None])


def test_every_unknown_metric_is_named_before_any_metric_runs(orchestrator, assessed):
    with pytest.raises(ValueError, match="not registered: nope_one, nope_two"):
        orchestrator.assess(
            metrics=["completeness_nullRatio", "nope_one", "nope_two"],
            metric_configs=[None, None, None],
        )
    assert not assessed


def test_matching_lengths_with_known_metrics_run(orchestrator):
    results = orchestrator.assess(
        metrics=["completeness_nullRatio", "minimality_duplicateCount"],
        metric_configs=[None, None],
    )
    assert results


def test_a_none_config_means_the_defaults(orchestrator):
    assert orchestrator.assess(metrics=["completeness_nullRatio"], metric_configs=[None])


def test_load_config_builds_the_defaults_for_none():
    from metis.metric.completeness.completeness_nullAndDMVRatio import (
        completeness_nullAndDMVRatio,
    )
    from metis.metric.completeness.completeness_nullAndDMVRatio_config import (
        completeness_nullAndDMVRatio_config,
    )

    config = completeness_nullAndDMVRatio().load_config(
        None, completeness_nullAndDMVRatio_config
    )
    assert config == completeness_nullAndDMVRatio_config()


def _write_config(tmp_path, text):
    path = tmp_path / "dataset.json"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_a_missing_data_config_is_declined(tmp_path):
    path = str(tmp_path / "missing.json")
    with pytest.raises(DatasetLoadError, match="missing.json' does not exist") as e:
        DQOrchestrator().load([path])
    assert isinstance(e.value.__cause__, FileNotFoundError)


def test_a_data_config_that_is_not_json_is_declined(tmp_path):
    path = _write_config(tmp_path, "{not json")
    with pytest.raises(DatasetLoadError, match="dataset.json' is not valid JSON") as e:
        DQOrchestrator().load([path])
    assert isinstance(e.value.__cause__, json.JSONDecodeError)


def test_a_data_config_that_is_not_an_object_is_declined(tmp_path):
    path = _write_config(tmp_path, "[1, 2]")
    with pytest.raises(DatasetLoadError, match="dataset.json' must hold a JSON object, not list"):
        DQOrchestrator().load([path])


@pytest.fixture
def load(tmp_path, monkeypatch):
    monkeypatch.setattr(metis.globals, "data_root", str(tmp_path))

    def _load(content: bytes, **config):
        (tmp_path / "data.csv").write_bytes(content)
        path = _write_config(tmp_path, json.dumps({
            "loader": "CSV", "name": "Sample", "file_name": "data.csv", **config,
        }))
        orchestrator = DQOrchestrator()
        orchestrator.load([path])
        return orchestrator.dataframes["Sample"]

    return _load


def test_repeated_column_names_are_warned_about_and_the_file_still_loads(load, caplog):
    with caplog.at_level(logging.WARNING, logger="metis"):
        frame = load(b"a,b,a,b,c\n1,2,3,4,5\n")

    assert list(frame.columns) == ["a", "b", "a.1", "b.1", "c"]
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(messages) == 1
    assert "['a', 'b']" in messages[0]
    assert "renamed" in messages[0]


def test_a_real_column_named_with_a_suffix_is_not_flagged(load, caplog):
    with caplog.at_level(logging.WARNING, logger="metis"):
        frame = load(b"a,a.1\n1,2\n")

    assert list(frame.columns) == ["a", "a.1"]
    assert not caplog.records


def test_repeats_are_found_in_the_configured_header_row_and_delimiter(load, caplog):
    with caplog.at_level(logging.WARNING, logger="metis"):
        load(b"junk;row\nx;x\n1;2\n", delimiter=";", header=1)

    assert "['x']" in caplog.records[0].getMessage()


def test_a_title_line_above_the_header_does_not_break_the_load(load, caplog):
    with caplog.at_level(logging.WARNING, logger="metis"):
        frame = load(b"Exported on Monday\na,b,c\n1,2,3\n", header=1)

    assert list(frame.columns) == ["a", "b", "c"]
    assert not caplog.records


def test_the_duplicate_check_is_skipped_without_a_header_row_index(load, caplog):
    with caplog.at_level(logging.WARNING, logger="metis"):
        frame = load(b"a,a\n1,2\n", header=None)

    assert frame.shape == (2, 2)
    assert not caplog.records
