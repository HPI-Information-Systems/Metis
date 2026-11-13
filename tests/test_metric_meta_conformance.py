"""Every registered metric must declare complete, self-consistent metadata."""
import pathlib

import pytest

import metis.metric  # noqa: F401 - populates Metric.registry
from metis.metric.metric import Metric
from metis.metric.metric_meta import MetricMeta
from metis.utils.dq_dimension import DQDimension

# Metric.__init_subclass__ registers EVERY subclass globally and forever, so
# test doubles defined in other test modules (tests/test_metric_base.py defines
# _Described and _Undescribed) are in the registry too by the time pytest has
# collected everything. Conformance is about the project's real metrics, so
# private names are excluded. A cleanup fixture cannot fix this: pytest imports
# every test module during collection, before any fixture runs.
_METRICS = sorted((n, c) for n, c in Metric.registry.items() if not n.startswith("_"))


def _metric_modules() -> list[str]:
    """Names of the metric modules under ``metis/metric/<dimension>/``."""
    root = pathlib.Path(metis.metric.__file__).parent
    return sorted(
        path.stem
        for path in root.glob("*/*.py")
        if not path.stem.startswith("_") and not path.stem.endswith("_config")
    )


def test_every_metric_module_is_registered():
    # A metric only reaches the registry once it is imported in
    # metis/metric/__init__.py. Forgetting that import leaves the metric
    # invisible to the orchestrator and the GUI without any error.
    missing = [name for name in _metric_modules() if name not in Metric.registry]
    assert not missing, (
        f"{missing} are not registered. Import them in metis/metric/__init__.py."
    )


@pytest.mark.parametrize("name,cls", _METRICS, ids=[n for n, _ in _METRICS])
def test_metric_declares_meta(name, cls):
    assert isinstance(cls.describe(), MetricMeta)


@pytest.mark.parametrize("name,cls", _METRICS, ids=[n for n, _ in _METRICS])
def test_dimension_matches_the_name_prefix(name, cls):
    prefix = name.split("_")[0]
    assert cls.describe().dimension == DQDimension(prefix.capitalize())


@pytest.mark.parametrize("name,cls", _METRICS, ids=[n for n, _ in _METRICS])
def test_label_does_not_repeat_the_dimension(name, cls):
    label = cls.describe().label
    prefix = name.split("_")[0].lower()
    assert not label.lower().startswith(prefix), (
        f"{name} label {label!r} repeats the dimension. Callers add the "
        f"dimension prefix themselves."
    )


@pytest.mark.parametrize("name,cls", _METRICS, ids=[n for n, _ in _METRICS])
def test_declares_at_least_one_granularity(name, cls):
    assert cls.describe().granularities


@pytest.mark.parametrize("name,cls", _METRICS, ids=[n for n, _ in _METRICS])
def test_no_legacy_gui_attributes_remain(name, cls):
    legacy = [a for a in dir(cls) if a.startswith("_gui_")]
    assert not legacy, f"{name} still declares {legacy}. Move them into meta."


@pytest.mark.parametrize("name,cls", _METRICS, ids=[n for n, _ in _METRICS])
def test_scores_use_the_zero_to_one_convention(name, cls):
    meta = cls.describe()
    assert meta.value_range == (0.0, 1.0)
    assert meta.higher_is_better is True


def test_only_accuracy_metrics_cite_a_standard():
    for name, cls in _METRICS:
        standard = cls.describe().standard
        if standard is not None:
            assert name.startswith("accuracy_"), (
                f"{name} cites {standard!r}. Only the accuracy metrics are "
                f"mapped to a standard in this release."
            )
            assert standard.startswith("ISO/IEC 25024:2015 Acc-I-")


def test_all_four_accuracy_metrics_cite_their_iso_measure():
    cited = {
        name: cls.describe().standard
        for name, cls in _METRICS
        if name.startswith("accuracy_")
    }
    assert cited == {
        "accuracy_syntacticDomain": "ISO/IEC 25024:2015 Acc-I-1",
        "accuracy_semanticReference": "ISO/IEC 25024:2015 Acc-I-2",
        "accuracy_outlierRisk": "ISO/IEC 25024:2015 Acc-I-4",
        "accuracy_dataRange": "ISO/IEC 25024:2015 Acc-I-7",
    }
