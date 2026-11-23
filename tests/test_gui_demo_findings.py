"""The Dataset step describes the shown dataset, and FD violations are counted."""
import pathlib
import sys

import pandas as pd

_GUI = pathlib.Path(__file__).parent.parent / "gui"
if str(_GUI) not in sys.path:
    sys.path.insert(0, str(_GUI))

from ui.pages import dataset_page  # noqa: E402
from visualization.renderers.fd_violations import extract_rules  # noqa: E402
from metis.metric.consistency.consistency_countFDViolations import (  # noqa: E402
    consistency_countFDViolations,
)


def test_the_statistics_belong_to_the_dataframe_that_is_described():
    first = dataset_page._describe_cached(pd.DataFrame({"n": range(288)}))
    second = dataset_page._describe_cached(pd.DataFrame({"n": range(864)}))

    assert first.loc["count", "n"] == "288.0"
    assert second.loc["count", "n"] == "864.0"


def test_the_violating_rows_of_an_fd_rule_are_counted(tmp_path):
    data = pd.DataFrame({
        "address": ["a", "a", "b", "c"],
        "city": ["x", "y", "z", "z"],
    })
    config = tmp_path / "fds.json"
    config.write_text('{"address": ["city"]}')
    results = consistency_countFDViolations().assess(data, metric_config=str(config))
    assert results[0].DQvalue == 0.5

    rules = extract_rules([{"DQexplanation": r.DQexplanation} for r in results])

    assert rules["address:city"] == [
        {"address": "a", "city": "x"}, {"address": "a", "city": "y"},
    ]


def test_a_list_of_determinant_values_is_still_accepted():
    assert extract_rules([{"DQexplanation": {"a:b": ["x", "y"]}}]) == {"a:b": ["x", "y"]}
