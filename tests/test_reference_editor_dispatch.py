import inspect
import sys
import pathlib

import pandas as pd

_GUI = pathlib.Path(__file__).parent.parent / "gui"
if str(_GUI) not in sys.path:
    sys.path.insert(0, str(_GUI))

from core.metric_runner import run_all  # noqa: E402
from ui.components.config_editors import simple_editor  # noqa: E402
from metis.utils.reference_loader import ReferenceSource  # noqa: E402


def test_run_all_takes_the_global_reference_not_a_per_run_reference():
    params = inspect.signature(run_all).parameters
    assert "reference" not in params
    assert "global_reference" in params


def test_reference_annotation_is_recognised():
    assert simple_editor.is_reference_field(ReferenceSource) is True


def test_ordinary_annotations_are_not_reference_fields():
    assert simple_editor.is_reference_field(str) is False
    assert simple_editor.is_reference_field(int) is False
    assert simple_editor.is_reference_field(pd.DataFrame) is False


def test_app_state_holds_the_global_reference():
    import streamlit as st

    from ui.state import AppState

    st.session_state.pop("reference_df", None)
    assert AppState.get_reference_df() is None

    reference = pd.DataFrame({"a": [1]})
    AppState.set_reference_df(reference)
    assert AppState.get_reference_df() is reference


def test_reset_clears_the_global_reference():
    import streamlit as st

    from ui.state import AppState

    AppState.set_reference_df(pd.DataFrame({"a": [1]}))
    st.session_state["_ref_cache__global__stamp"] = "ref.csv::123"

    AppState.reset()

    assert AppState.get_reference_df() is None
    assert "_ref_cache__global__stamp" not in st.session_state


def test_run_all_feeds_the_global_reference_to_a_metric_that_requires_one():
    data = pd.DataFrame({"n": [10, 99]})
    results, errors = run_all(
        ["correctness_heinrich"], data, {}, lambda *_: None,
        global_reference=pd.DataFrame({"n": [10, 20]}),
    )
    assert not errors
    assert [r.DQvalue for r in results][0] == 1.0
    assert [r.DQvalue for r in results][1] < 1.0


def test_run_all_prefers_a_metrics_own_reference_over_the_global_one():
    from metis.metric.correctness.correctness_heinrich_config import (
        correctness_heinrich_config,
    )

    data = pd.DataFrame({"n": [10, 99]})
    own = correctness_heinrich_config(reference=data.copy())
    results, errors = run_all(
        ["correctness_heinrich"], data, {"correctness_heinrich": own},
        lambda *_: None,
        global_reference=pd.DataFrame({"n": [10, 20]}),
    )
    assert not errors
    assert all(r.DQvalue == 1.0 for r in results)


def test_run_all_reports_a_missing_reference_as_that_metrics_error():
    data = pd.DataFrame({"n": [10, 99]})
    results, errors = run_all(["correctness_heinrich"], data, {}, lambda *_: None)
    assert not results
    assert "requires a reference" in errors[0].error


def test_reset_clears_per_metric_reference_state():
    import streamlit as st

    from ui.state import AppState

    st.session_state["df"] = pd.DataFrame({"a": [1]})
    st.session_state["dataset_name"] = "dataset-a"
    st.session_state["_ref_cache__correctness_heinrich__reference"] = pd.DataFrame({"b": [2]})
    st.session_state["_ref_cache__correctness_heinrich__reference__stamp"] = "ref.csv::123"
    st.session_state["correctness_heinrich__reference"] = "uploaded-file-stub"

    AppState.reset()

    assert "df" not in st.session_state
    assert "dataset_name" not in st.session_state
    assert "_ref_cache__correctness_heinrich__reference" not in st.session_state
    assert "_ref_cache__correctness_heinrich__reference__stamp" not in st.session_state
    assert "correctness_heinrich__reference" not in st.session_state


class _Upload:
    """Stands in for Streamlit's ``UploadedFile``."""

    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self.size = len(content)
        self._content = content

    def read(self) -> bytes:
        return self._content


def _render_heinrich_config(monkeypatch, upload):
    import streamlit as st

    from core.metric_catalog import get_catalog
    from ui.pages import metrics_page

    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    metrics_page._render_inline_config(
        get_catalog()["correctness_heinrich"], pd.DataFrame({"n": [10, 99]})
    )


def test_a_declined_reference_upload_drops_the_reference_stored_before(monkeypatch):
    import streamlit as st

    from ui.state import AppState

    st.session_state.clear()
    _render_heinrich_config(monkeypatch, _Upload("ref.csv", b"n\n10\n20\n"))
    assert len(AppState.get_metric_configs()["correctness_heinrich"].reference) == 2

    _render_heinrich_config(monkeypatch, _Upload("empty.csv", b""))

    assert "correctness_heinrich" not in AppState.get_metric_configs()
    st.session_state.clear()


def test_a_declined_reference_upload_falls_back_to_the_global_reference(monkeypatch):
    import streamlit as st

    from ui.state import AppState

    st.session_state.clear()
    AppState.set_reference_df(pd.DataFrame({"n": [10, 20, 30]}))
    _render_heinrich_config(monkeypatch, _Upload("ref.csv", b"n\n10\n20\n"))
    assert len(AppState.get_metric_configs()["correctness_heinrich"].reference) == 2

    _render_heinrich_config(monkeypatch, _Upload("empty.csv", b""))

    assert AppState.get_metric_configs()["correctness_heinrich"].reference is None
    st.session_state.clear()
