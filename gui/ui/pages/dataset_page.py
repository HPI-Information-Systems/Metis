"""Step 1: Upload a CSV dataset and set run metadata."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from core.csv_upload import CSV_FORMAT_EXPECTATION, read_csv_upload
from core.metric_catalog import get_catalog
from core.result_store import get_active_store
from metis.utils.dataset_checks import DatasetError
from ui.state import AppState

LARGE_DATASET_ROW_THRESHOLD: int = 100_000
MAX_RECENT_RUNS_VISIBLE: int = 4
RECENT_RUN_CARD_BORDER: str = "#e0e0e0"
RECENT_RUN_CARD_BG: str = "#fafafa"
RECENT_RUN_CARD_HEIGHT_PX: int = 90

# Session keys of the global reference upload. The "_ref_cache__" prefix puts
# them in the sweep AppState.reset() runs over all reference upload state.
_REFERENCE_STAMP_KEY: str = "_ref_cache__global__stamp"
_REFERENCE_VERSION_KEY: str = "_ref_cache__global__version"
_REFERENCE_ERROR_KEY: str = "_ref_cache__global__error"

# A file is parsed once, so its messages are kept to survive later reruns.
_UPLOAD_ERROR_KEY: str = "_upload_error"
_UPLOAD_WARNINGS_KEY: str = "_upload_warnings"

# A tag still equal to this was not edited by the user.
_AUTO_TAG_KEY: str = "_auto_experiment_tag"


def render(demo_mode: bool = False, demo_df: pd.DataFrame | None = None) -> None:
    """
    Render the dataset upload step (or a read-only demo info card).

    :param demo_mode: When True, render the read-only demo info card instead of the upload UI.
    :param demo_df: Pre-loaded demo dataframe (only used when ``demo_mode`` is True).
    :return: None.
    """
    if demo_mode:
        _render_demo_dataset_info(demo_df)
        return
    _render_recent_runs_fragment()
    _render_upload_section()


def _render_upload_section() -> None:
    st.header("Dataset")
    st.write("Upload a CSV file to assess its data quality.")
    st.markdown(CSV_FORMAT_EXPECTATION)

    uploaded = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Drag and drop or click to browse. Files >100 MB may be slow.",
    )

    if uploaded is not None:
        file_id = f"{uploaded.name}::{uploaded.size}"
        if st.session_state.get("_loaded_file_id") != file_id:
            _load_csv(uploaded)
            st.session_state["_loaded_file_id"] = file_id
    else:
        # Forget the declined file so it can be uploaded again.
        if st.session_state.pop(_UPLOAD_ERROR_KEY, None) is not None:
            st.session_state.pop("_loaded_file_id", None)

    error = st.session_state.get(_UPLOAD_ERROR_KEY)
    if error:
        st.error(error)

    df = AppState.get_df()
    if df is not None:
        st.success(
            f"Dataset loaded: **{AppState.get_dataset_name()}** — "
            f"{len(df):,} rows × {len(df.columns)} columns"
        )
        for warning in st.session_state.get(_UPLOAD_WARNINGS_KEY, []):
            st.warning(warning)
        # Metadata form runs in main scope so its values affect session_state
        # immediately. Nav-button availability reads df from session_state on the same rerun.
        _render_metadata_form()
        _render_preview_fragment(df)
        _render_reference_upload()
    elif not error:
        st.info("Upload a CSV file to get started.")


def _load_csv(uploaded) -> None:
    """
    Parse an uploaded CSV file and seed app state, or decline the file.

    A file that is empty or cannot be read as CSV is declined: the reason is
    kept for display and no dataset is loaded, so the wizard cannot continue
    with it.

    :param uploaded: A Streamlit ``UploadedFile``.
    :return: None.
    """
    st.session_state.pop(_UPLOAD_ERROR_KEY, None)
    st.session_state.pop(_UPLOAD_WARNINGS_KEY, None)

    try:
        upload = read_csv_upload(uploaded.read(), uploaded.name)
    except DatasetError as e:
        # Also drop an earlier dataset, so it is not shown next to the error.
        AppState.clear_df()
        st.session_state[_UPLOAD_ERROR_KEY] = str(e)
        return

    df = upload.frame
    notes = list(upload.warnings)
    if len(df) > LARGE_DATASET_ROW_THRESHOLD:
        notes.append(
            f"Large file: {len(df):,} rows. Metrics may be slow. "
            "Consider using the max-rows cap in the Compute step."
        )
    st.session_state[_UPLOAD_WARNINGS_KEY] = notes

    dataset_name = uploaded.name.removesuffix(".csv")
    AppState.set_df(df)
    AppState.set_dataset_name(dataset_name)
    _sync_metadata_input("input_dataset_name", dataset_name)

    tag = AppState.get_experiment_tag()
    if not tag or tag == st.session_state.get(_AUTO_TAG_KEY):
        tag = f"{dataset_name}_{datetime.now():%Y%m%d_%H%M}"
        AppState.set_experiment_tag(tag)
        st.session_state[_AUTO_TAG_KEY] = tag
        _sync_metadata_input("input_experiment_tag", tag)


def _sync_metadata_input(widget_key: str, value: str) -> None:
    """
    Point a run-metadata text input at a newly loaded dataset.

    The inputs keep their own widget state. Left alone, an input that is
    already on screen writes the previous file's name back over the new one.

    :param widget_key: Session key of the text input.
    :param value: The value the input should show.
    :return: None.
    """
    if widget_key in st.session_state:
        st.session_state[widget_key] = value


def _render_reference_upload() -> None:
    """
    Render the optional upload of a global reference dataset.

    The reference is the default for every metric that requires one. A metric
    can override it with its own upload on the Metrics step.

    :return: None.
    """
    with st.expander(
        "Reference dataset (optional)",
        expanded=AppState.get_reference_df() is not None,
    ):
        users = ", ".join(
            f"**{info.dimension}: {info.label}**"
            for info in get_catalog().values()
            if info.requires_reference
        )
        st.write(
            f"Some metrics ({users}) compare your data against a clean "
            "reference. Upload it here once and each of them uses it. You can "
            "still give a single metric its own reference on the Metrics step."
        )

        # A file uploader cannot be emptied from code, so removing the
        # reference moves the uploader to a fresh key instead.
        version = st.session_state.get(_REFERENCE_VERSION_KEY, 0)
        uploaded = st.file_uploader(
            "Choose a reference CSV file",
            type=["csv"],
            key=f"reference_uploader_{version}",
        )
        if uploaded is not None:
            stamp = f"{uploaded.name}::{uploaded.size}"
            if st.session_state.get(_REFERENCE_STAMP_KEY) != stamp:
                st.session_state[_REFERENCE_STAMP_KEY] = stamp
                try:
                    upload = read_csv_upload(uploaded.read(), uploaded.name)
                except DatasetError as e:
                    AppState.set_reference_df(None)
                    st.session_state[_REFERENCE_ERROR_KEY] = str(e)
                else:
                    AppState.set_reference_df(upload.frame)
                    st.session_state.pop(_REFERENCE_ERROR_KEY, None)
        elif st.session_state.pop(_REFERENCE_ERROR_KEY, None) is not None:
            st.session_state.pop(_REFERENCE_STAMP_KEY, None)

        if st.session_state.get(_REFERENCE_ERROR_KEY):
            st.error(st.session_state[_REFERENCE_ERROR_KEY])

        reference = AppState.get_reference_df()
        if reference is not None:
            st.success(
                f"Reference loaded: {len(reference):,} rows × "
                f"{len(reference.columns)} columns"
            )
            st.button(
                "Remove reference",
                key="reference_remove",
                on_click=_remove_reference,
            )


def _remove_reference() -> None:
    """
    Drop the global reference dataset and empty its uploader.

    :return: None.
    """
    AppState.set_reference_df(None)
    st.session_state.pop(_REFERENCE_STAMP_KEY, None)
    st.session_state.pop(_REFERENCE_ERROR_KEY, None)
    st.session_state[_REFERENCE_VERSION_KEY] = (
        st.session_state.get(_REFERENCE_VERSION_KEY, 0) + 1
    )


def _render_metadata_form() -> None:
    st.subheader("Run metadata")
    col1, col2, col3 = st.columns(3)

    with col1:
        name = st.text_input(
            "Dataset name",
            value=AppState.get_dataset_name(),
            key="input_dataset_name",
        )
        AppState.set_dataset_name(name)

    with col2:
        table = st.text_input(
            "Table name (optional)",
            value=AppState.get_table_name(),
            key="input_table_name",
        )
        AppState.set_table_name(table)

    with col3:
        tag = st.text_input(
            "Experiment tag",
            value=AppState.get_experiment_tag(),
            placeholder="e.g. baseline-2026-03",
            key="input_experiment_tag",
        )
        AppState.set_experiment_tag(tag)


@st.fragment
def _render_preview_fragment(df: pd.DataFrame) -> None:
    """
    Render the dataset preview in an isolated fragment.

    Wrapped in a fragment so typing in the metadata form above does not retrigger
    the (expensive) ``df.describe`` call.

    :param df: The currently loaded dataframe.
    :return: None.
    """
    st.subheader("Preview")
    st.dataframe(df.head(50), width='stretch')

    with st.expander("Column types & statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Dtypes**")
            st.dataframe(df.dtypes.rename("dtype").to_frame().astype(str), width='stretch')
        with col2:
            st.write("**Descriptive statistics**")
            st.dataframe(_describe_cached(df), width='stretch')


@st.cache_data(show_spinner=False)
def _describe_cached(_df: pd.DataFrame) -> pd.DataFrame:
    """Cache ``df.describe(include='all')`` — expensive on wide datasets."""
    return _df.describe(include="all").astype(str)


@st.fragment
def _render_recent_runs_fragment() -> None:
    try:
        store = get_active_store()
        runs = store.list_runs()
    except Exception:
        return

    if not runs:
        return

    hcol, vcol = st.columns([5, 1])
    with hcol:
        st.subheader("Recent Results")
    with vcol:
        if st.button("View all →", width="stretch", key="recent_view_all"):
            AppState.set_last_experiment_tag(runs[0].experiment_tag)
            AppState.set_step(3)
            st.rerun()

    visible = runs[:MAX_RECENT_RUNS_VISIBLE]
    cols = st.columns(len(visible))

    for col, run in zip(cols, visible):
        try:
            dt = datetime.fromisoformat(run.timestamp)
            date_str = dt.strftime("%-d %b %Y, %H:%M")
        except Exception:
            date_str = run.timestamp[:16] if run.timestamp else "—"

        n_metrics = len(run.metrics) if run.metrics else 0
        metric_str = f"{n_metrics} metric{'s' if n_metrics != 1 else ''}"

        with col:
            st.markdown(
                f"<div style='"
                f"border:1px solid {RECENT_RUN_CARD_BORDER};border-radius:8px;padding:12px 14px 8px;"
                f"background:{RECENT_RUN_CARD_BG};height:{RECENT_RUN_CARD_HEIGHT_PX}px;overflow:hidden;"
                f"'>"
                f"<div style='font-weight:600;font-size:14px;margin-bottom:3px;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
                f"{run.experiment_tag or '(no tag)'}</div>"
                f"<div style='color:#555;font-size:12px;margin-bottom:4px;"
                f"white-space:nowrap;overflow:hidden;text-overflow:ellipsis;'>"
                f"{run.dataset_name or '—'}</div>"
                f"<div style='color:#888;font-size:11px;'>"
                f"🕐 {date_str}<br>📊 {metric_str}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button("Open →", key=f"open_run_{run.experiment_tag}", width="stretch"):
                AppState.set_last_experiment_tag(run.experiment_tag)
                AppState.set_step(3)
                st.rerun()

    st.divider()


def _render_demo_dataset_info(df: pd.DataFrame | None) -> None:
    """
    Render a read-only dataset card shown in demo mode.

    :param df: The pre-loaded demo dataframe, or ``None`` if it failed to load.
    :return: None.
    """
    st.header("Dataset")
    st.info(
        "**Demo mode**. The restaurant dataset is pre-loaded. "
        "Switch to the **Own Files** tab to upload your own CSV."
    )
    if df is None:
        st.warning("Demo dataset could not be loaded.")
        return

    st.success(
        f"**restaurant_sample.csv** — {len(df):,} rows × {len(df.columns)} columns  "
        f"(dirty-restaurants duplicate-detection benchmark)"
    )
    _render_preview_fragment(df)
