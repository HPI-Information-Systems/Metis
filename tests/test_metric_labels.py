import sys
import pathlib

_GUI = pathlib.Path(__file__).parent.parent / "gui"
if str(_GUI) not in sys.path:
    sys.path.insert(0, str(_GUI))

from visualization.metric_palette import short_name  # noqa: E402
from ui.pages.metrics_page import _format_name  # noqa: E402


def test_short_name_uses_the_declared_label():
    assert short_name("completeness_nullRatio") == "Null Ratio"
    assert short_name("consistency_ruleBasedPipino") == "Rule Based (Pipino)"
    assert short_name("validity_outOfVocabulary") == "Out of Vocabulary"


def test_short_name_falls_back_for_unregistered_metrics():
    assert short_name("legacy_someOldMetric") == "someOldMetric"
    assert short_name("noUnderscore") == "noUnderscore"


def test_format_name_qualifies_with_the_dimension():
    assert _format_name("completeness_nullRatio") == "Completeness: Null Ratio"
    assert _format_name("correctness_heinrich") == "Correctness: Heinrich"


def test_format_name_falls_back_for_unregistered_metrics():
    assert _format_name("legacy_someOldMetric") == "Legacy: Some Old Metric"
