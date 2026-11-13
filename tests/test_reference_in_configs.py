import inspect

import pandas as pd
import pytest

import metis.metric  # noqa: F401
from metis.metric.accuracy.accuracy_semanticReference import accuracy_semanticReference
from metis.metric.accuracy.accuracy_semanticReference_config import (
    accuracy_semanticReference_config,
)
from metis.metric.correctness.correctness_heinrich import correctness_heinrich
from metis.metric.correctness.correctness_heinrich_config import (
    correctness_heinrich_config,
)
from metis.metric.metric import Metric
from metis.utils.reference_loader import MissingReferenceError
from metis.metric.validity.validity_outOfVocabulary import validity_outOfVocabulary
from metis.metric.validity.validity_outOfVocabulary_config import (
    validity_outOfVocabulary_config,
)

# Metric.__init_subclass__ registers EVERY subclass globally and forever, so
# test doubles defined in tests/test_metric_base.py (_Described, _Undescribed)
# are in the registry too once that module has been collected. This test is
# about the project's real metrics, so private names are excluded (see
# tests/test_metric_meta_conformance.py for the same fix and rationale).
_METRICS = sorted((n, c) for n, c in Metric.registry.items() if not n.startswith("_"))


@pytest.mark.parametrize("name,cls", _METRICS, ids=[n for n, _ in _METRICS])
def test_assess_no_longer_takes_a_reference_parameter(name, cls):
    params = list(inspect.signature(cls.assess).parameters)
    assert "reference" not in params
    assert params == ["self", "data", "metric_config"]


def test_semantic_reference_reads_its_reference_from_config():
    data = pd.DataFrame({"city": ["Berlin", "Paris"]})
    ref = pd.DataFrame({"city": ["Berlin", "Lyon"]})
    results = accuracy_semanticReference().assess(
        data, metric_config=accuracy_semanticReference_config(reference=ref)
    )
    assert results[0].DQvalue == pytest.approx(0.5)


def test_semantic_reference_without_a_reference_raises():
    data = pd.DataFrame({"city": ["Berlin"]})
    with pytest.raises(ValueError, match="reference"):
        accuracy_semanticReference().assess(
            data, metric_config=accuracy_semanticReference_config()
        )


def test_correctness_reads_its_reference_from_config():
    data = pd.DataFrame({"n": [10, 20]})
    ref = pd.DataFrame({"n": [10, 20]})
    results = correctness_heinrich().assess(
        data, metric_config=correctness_heinrich_config(reference=ref)
    )
    assert all(r.DQvalue == pytest.approx(1.0) for r in results)


def test_out_of_vocabulary_takes_an_inline_vocabulary():
    data = pd.DataFrame({"word": ["apple", "zzzz"]})
    results = validity_outOfVocabulary().assess(
        data, metric_config=validity_outOfVocabulary_config(reference={"apple"})
    )
    assert results[0].DQvalue == pytest.approx(0.5)


def test_reference_survives_config_serialisation():
    cfg = correctness_heinrich_config(reference="data/clean.csv")
    assert cfg.to_json()["reference"] == "data/clean.csv"


@pytest.mark.parametrize(
    "config_class", [accuracy_semanticReference_config, correctness_heinrich_config]
)
def test_a_missing_required_reference_has_its_own_error_type(config_class):
    # The GUI catches this type to tell the user where to upload a reference.
    with pytest.raises(MissingReferenceError):
        config_class().validate()
