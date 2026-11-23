"""An empty dataset is declined centrally, the same way for every metric.

The check sits in the orchestrator and in the GUI runner rather than in the
metrics, so a newly added metric is covered without its author doing
anything. These tests run once per registered metric to prove that.
"""
import json
import pathlib
import sys

import pandas as pd
import pytest

import metis.globals
import metis.metric  # noqa: F401 - populates Metric.registry
from metis.dq_orchestrator import DQOrchestrator
from metis.metric.metric import Metric
from metis.utils.dataset_checks import (
    DatasetLoadError,
    EmptyDatasetError,
    ensure_not_empty,
)
from metis.utils.reference_loader import load_reference_frame

_GUI = pathlib.Path(__file__).parent.parent / "gui"
if str(_GUI) not in sys.path:
    sys.path.insert(0, str(_GUI))

from core.metric_runner import run_all  # noqa: E402

_METRICS = sorted(n for n in Metric.registry if not n.startswith("_"))

_EMPTY = {
    "no_rows": pd.DataFrame({"name": pd.Series(dtype="object"), "age": pd.Series(dtype="int64")}),
    "no_columns": pd.DataFrame(),
}


def test_a_dataset_with_data_passes():
    ensure_not_empty(pd.DataFrame({"a": [1]}))


def test_no_rows_is_reported_as_such():
    with pytest.raises(EmptyDatasetError, match="2 column\\(s\\) but no rows"):
        ensure_not_empty(_EMPTY["no_rows"], "Dataset 'Adult'")


def test_no_columns_is_reported_as_such():
    with pytest.raises(EmptyDatasetError, match="Dataset 'Adult' is empty: it has no columns"):
        ensure_not_empty(_EMPTY["no_columns"], "Dataset 'Adult'")


@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("case", _EMPTY)
def test_the_orchestrator_declines_an_empty_dataset_for_every_metric(metric, case):
    orchestrator = DQOrchestrator()
    orchestrator.dataframes["Empty"] = _EMPTY[case]
    orchestrator.data_paths["Empty"] = "empty.json"

    with pytest.raises(EmptyDatasetError, match="Dataset 'Empty' is empty"):
        orchestrator.assess(metrics=[metric], metric_configs=[None])


@pytest.mark.parametrize("metric", _METRICS)
@pytest.mark.parametrize("case", _EMPTY)
def test_the_gui_runner_declines_an_empty_dataset_for_every_metric(metric, case):
    results, errors = run_all([metric], _EMPTY[case], {}, lambda *_: None)

    assert not results
    assert [e.metric for e in errors] == [metric]
    assert errors[0].error.startswith("The dataset is empty")


def test_an_empty_dataset_stops_the_run_before_any_metric_is_assessed(monkeypatch):
    assessed = []
    monkeypatch.setattr(
        Metric.registry["completeness_nullRatio"], "assess",
        lambda self, data, *, metric_config=None: assessed.append(data) or [],
    )
    orchestrator = DQOrchestrator()
    orchestrator.dataframes["Full"] = pd.DataFrame({"a": [1]})
    orchestrator.dataframes["Empty"] = _EMPTY["no_rows"]
    orchestrator.data_paths.update({"Full": "full.json", "Empty": "empty.json"})

    with pytest.raises(EmptyDatasetError):
        orchestrator.assess(metrics=["completeness_nullRatio"], metric_configs=[""])
    assert not assessed


@pytest.fixture
def load(tmp_path, monkeypatch):
    """Load a dataset whose CSV file has the given content."""
    monkeypatch.setattr(metis.globals, "data_root", str(tmp_path))

    def _load(content: bytes | None, **config):
        if content is not None:
            (tmp_path / "data.csv").write_bytes(content)
        config_path = tmp_path / "dataset.json"
        config_path.write_text(json.dumps({
            "loader": "CSV", "name": "Sample", "file_name": "data.csv", **config,
        }), encoding="utf-8")
        orchestrator = DQOrchestrator()
        orchestrator.load(data_loader_configs=[str(config_path)])
        return orchestrator

    return _load


def test_loading_a_valid_file_still_works(load):
    assert load(b"a,b\n1,2\n").dataframes["Sample"].shape == (1, 2)


def test_loading_a_file_with_only_a_header_is_declined(load):
    with pytest.raises(EmptyDatasetError, match="Dataset 'Sample' is empty"):
        load(b"a,b\n")


def test_loading_an_empty_file_is_declined(load):
    with pytest.raises(DatasetLoadError, match="data.csv' is empty"):
        load(b"")


def test_loading_a_missing_file_is_declined(load):
    with pytest.raises(DatasetLoadError, match="does not exist. Check 'file_name'"):
        load(None)


def test_loading_a_file_in_another_encoding_is_declined(load):
    with pytest.raises(DatasetLoadError, match="is not valid utf-8. Set 'encoding'"):
        load("name\nJosé\n".encode("latin-1"))


def test_loading_a_file_with_inconsistent_rows_is_declined(load):
    with pytest.raises(DatasetLoadError, match="is not a valid CSV file"):
        load(b"a,b\n1,2\n3,4,5\n")


def test_loading_with_a_column_the_file_lacks_is_declined(load):
    with pytest.raises(DatasetLoadError, match="Dataset 'Sample' could not be loaded"):
        load(b"a,b\n1,2\n", usecols=["a", "missing"])


def test_a_header_only_reference_file_does_not_block_loading(load, tmp_path):
    (tmp_path / "ref.csv").write_text("a,b\n", encoding="utf-8")
    orchestrator = load(b"a,b\n1,2\n", reference_file_name="ref.csv")
    assert orchestrator.dataframes["Sample"].shape == (1, 2)


def test_a_missing_reference_file_names_the_reference_field(load):
    with pytest.raises(
        DatasetLoadError, match="ref.csv' does not exist. Check 'reference_file_name'"
    ):
        load(b"a,b\n1,2\n", reference_file_name="ref.csv")


@pytest.mark.parametrize("source", ["ref.csv", {"file_name": "ref.csv"}], ids=["path", "mapping"])
def test_a_missing_metric_reference_names_the_metric_config_field(source, tmp_path, monkeypatch):
    monkeypatch.setattr(metis.globals, "data_root", str(tmp_path))
    with pytest.raises(
        DatasetLoadError, match="does not exist. Check 'reference' in the metric config"
    ):
        load_reference_frame(source)
