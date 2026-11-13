"""The Dataset step loads a valid upload and declines an invalid one."""
import pathlib
import sys

import pandas as pd
import pytest
import streamlit as st

_GUI = pathlib.Path(__file__).parent.parent / "gui"
if str(_GUI) not in sys.path:
    sys.path.insert(0, str(_GUI))

from ui.pages import dataset_page  # noqa: E402
from ui.state import AppState  # noqa: E402


class _Upload:
    """Stands in for Streamlit's ``UploadedFile``."""

    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self.size = len(content)
        self._content = content

    def read(self) -> bytes:
        return self._content


@pytest.fixture(autouse=True)
def clean_session():
    st.session_state.clear()
    yield
    st.session_state.clear()


def test_a_valid_upload_becomes_the_dataset():
    dataset_page._load_csv(_Upload("people.csv", b"name,age\nAda,36\n"))

    assert AppState.get_df().shape == (1, 2)
    assert AppState.get_dataset_name() == "people"
    assert "_upload_error" not in st.session_state


@pytest.mark.parametrize(
    "content",
    [b"", b"name,age\n", b"name;age\nAda;36\n", b"\x00\x01\x02"],
    ids=["empty", "header_only", "semicolons", "binary"],
)
def test_an_invalid_upload_is_declined_and_loads_nothing(content):
    dataset_page._load_csv(_Upload("bad.csv", content))

    assert AppState.get_df() is None
    assert "bad.csv" in st.session_state["_upload_error"]


def test_a_declined_upload_replaces_the_dataset_loaded_before():
    AppState.set_df(pd.DataFrame({"a": [1]}))

    dataset_page._load_csv(_Upload("bad.csv", b""))

    assert AppState.get_df() is None


def test_a_valid_upload_clears_the_previous_decline():
    dataset_page._load_csv(_Upload("bad.csv", b""))
    dataset_page._load_csv(_Upload("good.csv", b"a\n1\n"))

    assert "_upload_error" not in st.session_state
    assert AppState.get_df() is not None


def test_skipped_rows_are_kept_as_a_warning():
    dataset_page._load_csv(_Upload("ragged.csv", b"a,b\n1,2\n3,4,5\n"))

    assert AppState.get_df().shape == (1, 2)
    assert "1 row(s)" in st.session_state["_upload_warnings"][0]
