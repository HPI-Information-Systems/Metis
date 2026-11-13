import sys
import pathlib

_GUI = pathlib.Path(__file__).parent.parent / "gui"
if str(_GUI) not in sys.path:
    sys.path.insert(0, str(_GUI))

from core.metric_catalog import get_catalog  # noqa: E402
from metis.metric.metric import Metric  # noqa: E402


def test_catalog_covers_every_registered_metric():
    registered = {n for n in Metric.registry if not n.startswith("_")}
    assert set(get_catalog()) == registered


def test_info_carries_the_label_from_meta():
    assert get_catalog()["completeness_nullRatio"].label == "Null Ratio"


def test_info_carries_the_iso_standard():
    assert get_catalog()["accuracy_outlierRisk"].standard == "ISO/IEC 25024:2015 Acc-I-4"
    assert get_catalog()["completeness_nullRatio"].standard is None


def test_dimension_comes_from_meta_not_string_splitting():
    assert get_catalog()["correctness_heinrich"].dimension == "Correctness"


def test_cell_granularity_is_derived():
    assert get_catalog()["correctness_heinrich"].cell_granularity is True
    assert get_catalog()["minimality_duplicateCount"].cell_granularity is False


def test_every_metric_has_a_description():
    missing = [n for n, i in get_catalog().items() if not i.description]
    assert not missing


def test_reference_flags_match_meta():
    catalog = get_catalog()
    needing = {n for n, i in catalog.items() if i.requires_reference}
    assert needing == {"accuracy_semanticReference", "correctness_heinrich"}
