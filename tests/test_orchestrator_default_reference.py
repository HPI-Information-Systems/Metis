"""A dataset-level reference is the default for metrics that require one."""
import json

import pandas as pd
import pytest

import metis.globals
from metis.dq_orchestrator import DQOrchestrator
from metis.metric.correctness.correctness_heinrich_config import (
    correctness_heinrich_config,
)
from metis.metric.validity.validity_outOfVocabulary_config import (
    validity_outOfVocabulary_config,
)


@pytest.fixture
def orchestrator(tmp_path, monkeypatch):
    """An orchestrator with a dataset whose config declares a reference file."""
    monkeypatch.setattr(metis.globals, "data_root", str(tmp_path))
    (tmp_path / "dirty.csv").write_text("n,word\n10,apple\n99,zzzz\n", encoding="utf-8")
    (tmp_path / "clean.csv").write_text("n,word\n10,apple\n20,pear\n", encoding="utf-8")
    config_path = tmp_path / "dataset.json"
    config_path.write_text(json.dumps({
        "loader": "CSV",
        "name": "Dirty",
        "file_name": "dirty.csv",
        "reference_file_name": "clean.csv",
    }), encoding="utf-8")

    orchestrator = DQOrchestrator()
    orchestrator.load(data_loader_configs=[str(config_path)])
    return orchestrator


def test_a_metric_without_its_own_reference_uses_the_dataset_reference(orchestrator):
    results = orchestrator.assess(
        metrics=["correctness_heinrich"],
        metric_configs=[correctness_heinrich_config()],
    )
    by_cell = {(r.columnNames[0], r.rowIndex): r.DQvalue for r in results}
    assert by_cell[("n", 0)] == pytest.approx(1.0)
    assert by_cell[("n", 1)] < 1.0


def test_a_json_config_without_a_reference_uses_the_dataset_reference(orchestrator):
    results = orchestrator.assess(
        metrics=["accuracy_semanticReference"],
        metric_configs=["{}"],
    )
    assert {r.columnNames[0]: r.DQvalue for r in results} == {
        "n": pytest.approx(0.5),
        "word": pytest.approx(0.5),
    }


def test_a_metric_config_reference_overrides_the_dataset_reference(orchestrator):
    own = pd.DataFrame({"n": [10, 99], "word": ["apple", "zzzz"]})
    results = orchestrator.assess(
        metrics=["correctness_heinrich"],
        metric_configs=[correctness_heinrich_config(reference=own)],
    )
    assert all(r.DQvalue == pytest.approx(1.0) for r in results)


def test_the_dataset_reference_is_not_forced_on_optional_reference_metrics(orchestrator):
    # validity_outOfVocabulary falls back to its built-in word list. The
    # dataset reference has two columns and would be rejected as a vocabulary.
    results = orchestrator.assess(
        metrics=["validity_outOfVocabulary"],
        metric_configs=[validity_outOfVocabulary_config()],
    )
    assert results


def test_without_a_dataset_reference_a_required_reference_still_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(metis.globals, "data_root", str(tmp_path))
    (tmp_path / "dirty.csv").write_text("n\n10\n", encoding="utf-8")
    config_path = tmp_path / "dataset.json"
    config_path.write_text(json.dumps({
        "loader": "CSV", "name": "Dirty", "file_name": "dirty.csv",
    }), encoding="utf-8")
    orchestrator = DQOrchestrator()
    orchestrator.load(data_loader_configs=[str(config_path)])

    with pytest.raises(ValueError, match="requires a reference"):
        orchestrator.assess(
            metrics=["correctness_heinrich"],
            metric_configs=[correctness_heinrich_config()],
        )
