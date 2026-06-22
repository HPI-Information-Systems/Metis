"""Step 1: Upload a CSV dataset and set run metadata."""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import streamlit as st

from core.result_store import get_active_store
from ui.state import AppState

LARGE_DATASET_ROW_THRESHOLD: int = 100_000
MAX_RECENT_RUNS_VISIBLE: int = 4
RECENT_RUN_CARD_BORDER: str = "#e0e0e0"
RECENT_RUN_CARD_BG: str = "#fafafa"
RECENT_RUN_CARD_HEIGHT_PX: int = 90


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

    df = AppState.get_df()
    if df is not None:
        st.success(
            f"Dataset loaded: **{AppState.get_dataset_name()}** — "
            f"{len(df):,} rows × {len(df.columns)} columns"
        )
        # Metadata form runs in main scope so its values affect session_state
        # immediately. Nav-button availability reads df from session_state on the same rerun.
        _render_metadata_form()
        _render_preview_fragment(df)
        _render_reference_upload()
    else:
        st.info("Upload a CSV file to get started.")


def _load_csv(uploaded) -> None:
    """
    Parse an uploaded CSV file with UTF-8 then latin-1 fallback, and seed app state.

    :param uploaded: A Streamlit ``UploadedFile``.
    :return: None.
    """
    raw = uploaded.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8", on_bad_lines="skip", low_memory=False)
    except UnicodeDecodeError:
        st.warning("UTF-8 decoding failed. Retrying with latin-1.")
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip", low_memory=False)

    if len(df) > LARGE_DATASET_ROW_THRESHOLD:
        st.warning(
            f"Large file: {len(df):,} rows. Metrics may be slow. "
            "Consider using the max-rows cap in the Compute step."
        )

    dataset_name = uploaded.name.removesuffix(".csv")
    AppState.set_df(df)
    AppState.set_dataset_name(dataset_name)

    if not AppState.get_experiment_tag():
        AppState.set_experiment_tag(f"{dataset_name}_{datetime.now():%Y%m%d_%H%M}")


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


def _render_reference_upload() -> None:
    with st.expander("Reference dataset (optional)"):
        st.write(
            "Some metrics (e.g. **correctness_heinrich**) compare your data against "
            "a clean reference. Upload it here if needed."
        )
        ref_file = st.file_uploader(
            "Reference CSV",
            type=["csv"],
            key="reference_uploader",
        )
        if ref_file is not None:
            ref_id = f"{ref_file.name}::{ref_file.size}"
            if st.session_state.get("_loaded_ref_id") != ref_id:
                raw = ref_file.read()
                try:
                    ref_df = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
                except UnicodeDecodeError:
                    ref_df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
                AppState.set_reference_df(ref_df)
                st.session_state["_loaded_ref_id"] = ref_id
            ref = AppState.get_reference_df()
            if ref is not None:
                st.success(f"Reference loaded: {len(ref):,} rows × {len(ref.columns)} columns")
        elif AppState.get_reference_df() is not None:
            ref = AppState.get_reference_df()
            st.info(f"Reference already loaded: {len(ref):,} rows × {len(ref.columns)} columns")
