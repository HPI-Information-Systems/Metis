from __future__ import annotations

import glob
import json
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import streamlit as st

try:
    import pyodide  # noqa: F401 — only present in stlite/Pyodide
    BROWSER_MODE = True
except ImportError:
    BROWSER_MODE = False

from core.result_store import (
    JSONResultStore,
    RunMetadata,
    SQLiteResultStore,
    set_active_store,
)
from core.metric_catalog import get_compute_blockers
from core.serialization import dict_to_result
from theme import HPI_RED
from ui.pages import compute_page, dataset_page, metrics_page, results_page
from ui.state import AppState


DEMO_ONLY: bool = os.environ.get("METIS_DEMO_ONLY", "").lower() in ("1", "true", "yes")

st.set_page_config(
    page_title="Metis · Data Quality Assessment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_STEPS: list[tuple[str, str]] = [
    ("📁", "Dataset"),
    ("📋", "Metrics"),
    ("⚙️", "Compute"),
    ("📈", "Results"),
]

STEP_CIRCLE_SIZE_PX: int = 38
STEP_FONT_SIZE_PX: int = 15
STEP_LABEL_FONT_SIZE_PX: int = 13

STEP_COLOR_CURRENT: str = HPI_RED
STEP_COLOR_DONE: str = "#2ca02c"
STEP_COLOR_PENDING_BG: str = "#f0f0f0"
STEP_COLOR_PENDING_FG: str = "#ccc"
STEP_COLOR_LINE_DONE: str = "#2ca02c"
STEP_COLOR_LINE_PENDING: str = "#e0e0e0"

LANDING_CARD_BORDER: str = "#e0e0e0"
LANDING_CARD_HEIGHT_PX: int = 200


@st.cache_resource
def _get_store():
    """
    Build (and cache) the active result store for this Streamlit process.

    :return: A ``JSONResultStore`` in browser mode, otherwise a ``SQLiteResultStore``.
    """
    if BROWSER_MODE:
        store = JSONResultStore()
    else:
        store = SQLiteResultStore()
    _seed_demo_results(store)
    return store


def _seed_demo_results(store) -> None:
    """
    Pre-load every bundled ``restaurant_results*.json`` snapshot into the store.

    The canonical "current" snapshot lives in ``restaurant_results.json`` and uses
    the tag ``demo``. Additional historical snapshots produced by
    ``build_temporal_demo.py`` (``restaurant_results_t*.json``) carry their own
    tags and predated timestamps, so the Comparison-over-time tab has multiple
    runs to chart.

    :param store: Either a ``JSONResultStore`` or a ``SQLiteResultStore``.
    :return: None.
    """
    demo_dir = os.path.join(os.path.dirname(__file__), "demo", "precomputed")
    if not os.path.isdir(demo_dir):
        return

    files = sorted(glob.glob(os.path.join(demo_dir, "restaurant_results*.json")))
    if not files:
        return

    existing_tags = {r.experiment_tag for r in store.list_runs()}

    for demo_file in files:
        with open(demo_file) as f:
            payload = json.load(f)
        results_raw = payload.get("results", payload) if isinstance(payload, dict) else payload
        if not results_raw:
            continue

        tag = results_raw[0].get("experimentTag") or "demo"
        dataset_name = results_raw[0].get("dataset") or "restaurant_sample"

        if tag in existing_tags:
            continue

        if isinstance(store, JSONResultStore):
            store._save_file(tag, results_raw)  # noqa: SLF001
        else:
            dq_results = [dict_to_result(r) for r in results_raw]
            store.save_run(
                dq_results,
                RunMetadata(experiment_tag=tag, dataset_name=dataset_name),
            )
        existing_tags.add(tag)


@st.cache_data
def _load_demo_df() -> pd.DataFrame:
    """
    Load the bundled restaurant sample CSV.

    :return: The cached ``pd.DataFrame`` for the demo dataset.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "demo", "restaurant_sample.csv")
    return pd.read_csv(csv_path, low_memory=False)


def main() -> None:
    """
    Entry point: route to the demo wizard, the own-files wizard, or both.

    :return: None.
    """
    st.title("Metis  ·  Data Quality Assessment")
    store = _get_store()
    set_active_store(store)

    if DEMO_ONLY:
        _render_demo_wizard(store)
        return

    if BROWSER_MODE:
        if not AppState.get_demo_mode_chosen():
            _render_landing(store)
            return
        path = AppState.get_browser_path()
        if path == "demo":
            _render_demo_wizard(store)
        else:
            _render_own_wizard(store)
        return

    tab_demo, tab_own = st.tabs(["Demo", "Own Files"])
    with tab_demo:
        _render_demo_wizard(store)
    with tab_own:
        _render_own_wizard(store)


def _render_own_wizard(store) -> None:
    """
    Render the standard upload-your-own-data wizard.

    :param store: The active result store.
    :return: None.
    """
    step = AppState.get_step()
    _render_step_indicator(step)
    st.divider()

    if step == 0:
        dataset_page.render()
    elif step == 1:
        metrics_page.render()
    elif step == 2:
        compute_page.render(store)
    elif step == 3:
        results_page.render(store, key_prefix="own_")

    df = AppState.get_df()
    selected = AppState.get_selected_metrics()
    accessible = _accessible_steps(df, selected)
    st.divider()
    _render_nav_buttons(step, accessible, key_prefix="own")


def _render_demo_wizard(store) -> None:
    """
    Render the demo wizard: pre-loaded dataset, read-only configs, instant compute.

    :param store: The active result store.
    :return: None.
    """
    demo_step = AppState.get_demo_step()
    _render_step_indicator(demo_step)
    st.divider()

    demo_df = _load_demo_df()

    if demo_step == 0:
        dataset_page.render(demo_mode=True, demo_df=demo_df)
    elif demo_step == 1:
        metrics_page.render(demo_mode=True)
    elif demo_step == 2:
        compute_page.render(store, demo_mode=True)
    elif demo_step == 3:
        results_page.render(store, key_prefix="demo_")

    demo_selected = AppState.get_demo_selected_metrics()
    accessible = {0, 1}
    if demo_selected:
        accessible.add(2)
    if AppState.get_last_experiment_tag():
        accessible.add(3)
    st.divider()
    _render_nav_buttons(
        demo_step,
        accessible,
        key_prefix="demo",
        set_step_fn=AppState.set_demo_step,
    )


def _accessible_steps(df, selected: list) -> set[int]:
    """
    Compute which step indices the user is allowed to navigate to.

    :param df: The currently uploaded dataset, or ``None``.
    :param selected: Currently selected metric names.
    :return: A set of accessible step indices in ``{0, 1, 2, 3}``.
    """
    accessible = {0}
    if df is not None:
        accessible.add(1)
    if df is not None and selected and not get_compute_blockers(selected, AppState.get_metric_configs()):
        accessible.add(2)
    if AppState.get_last_experiment_tag():
        accessible.add(3)
    return accessible


def _render_step_indicator(current: int) -> None:
    """
    Render the four-step progress indicator at the top of each wizard.

    :param current: The active step index ``[0, 3]``.
    :return: None.
    """
    parts: list[str] = []
    for i, (icon, label) in enumerate(_STEPS):
        is_current = i == current
        is_done = i < current

        if is_current:
            circle_bg = STEP_COLOR_CURRENT
            circle_fg = "white"
            text_color = STEP_COLOR_CURRENT
            text_weight = "700"
            circle_text = str(i + 1)
        elif is_done:
            circle_bg = STEP_COLOR_DONE
            circle_fg = "white"
            text_color = STEP_COLOR_DONE
            text_weight = "600"
            circle_text = "✓"
        else:
            circle_bg = STEP_COLOR_PENDING_BG
            circle_fg = STEP_COLOR_PENDING_FG
            text_color = STEP_COLOR_PENDING_FG
            text_weight = "400"
            circle_text = str(i + 1)

        parts.append(
            f'<div style="display:flex;flex-direction:column;align-items:center;flex:1;min-width:0;">'
            f'  <div style="width:{STEP_CIRCLE_SIZE_PX}px;height:{STEP_CIRCLE_SIZE_PX}px;border-radius:50%;background:{circle_bg};'
            f'              color:{circle_fg};display:flex;align-items:center;justify-content:center;'
            f'              font-weight:700;font-size:{STEP_FONT_SIZE_PX}px;flex-shrink:0;">'
            f'    {circle_text}'
            f'  </div>'
            f'  <div style="margin-top:6px;color:{text_color};font-weight:{text_weight};'
            f'              font-size:{STEP_LABEL_FONT_SIZE_PX}px;text-align:center;white-space:nowrap;">'
            f'    {icon}&nbsp;{label}'
            f'  </div>'
            f'</div>'
        )

        if i < len(_STEPS) - 1:
            line_color = STEP_COLOR_LINE_DONE if i < current else STEP_COLOR_LINE_PENDING
            parts.append(
                f'<div style="flex:2;height:2px;background:{line_color};'
                f'            margin-top:19px;margin-bottom:auto;"></div>'
            )

    st.markdown(
        '<div style="display:flex;align-items:flex-start;padding:12px 0 4px;">'
        + "".join(parts)
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_nav_buttons(
    step: int,
    accessible: set[int],
    key_prefix: str = "own",
    set_step_fn=None,
) -> None:
    """
    Render the Back / Next navigation buttons under each wizard step.

    :param step: The current step index.
    :param accessible: Step indices the user is allowed to navigate to.
    :param key_prefix: Streamlit widget key prefix to isolate own/demo wizards.
    :param set_step_fn: Optional setter for the active step (defaults to ``AppState.set_step``).
    :return: None.
    """
    if set_step_fn is None:
        set_step_fn = AppState.set_step

    col_back, _, col_next = st.columns([1, 6, 1])

    with col_back:
        if step > 0:
            if st.button("← Back", width='stretch', key=f"{key_prefix}_nav_back"):
                set_step_fn(step - 1)
                st.rerun()

    with col_next:
        next_step = step + 1
        if next_step < len(_STEPS):
            can_next = next_step in accessible
            _, next_label = _STEPS[next_step]
            if st.button(
                f"{next_label} →",
                disabled=not can_next,
                type="primary",
                width='stretch',
                key=f"{key_prefix}_nav_next",
            ):
                if key_prefix == "own" and step == 1:
                    AppState.set_run_requested(True)
                set_step_fn(next_step)
                st.rerun()
        elif step == len(_STEPS) - 1:
            if st.button(
                "New Dataset",
                type="primary",
                width="stretch",
                key=f"{key_prefix}_nav_new",
            ):
                if key_prefix == "demo":
                    AppState.set_demo_step(0)
                else:
                    AppState.reset()
                st.rerun()


def _render_landing(store) -> None:
    """
    Render the browser-mode landing page that lets the user pick demo or upload.

    :param store: The active result store.
    :return: None.
    """
    st.markdown(
        """
        Metis is a **data quality assessment framework** for tabular datasets.
        Choose how you'd like to explore it:
        """
    )

    col_demo, col_upload = st.columns(2)

    with col_demo:
        st.markdown(
            f"<div style='border:1px solid {LANDING_CARD_BORDER}; border-radius:8px; padding:24px; height:{LANDING_CARD_HEIGHT_PX}px;'>"
            "<h3 style='margin-top:0'>🍽️ Load Demo</h3>"
            "<p>Explore pre-computed DQ results on a restaurant dataset "
            "(1,000 rows, multiple metrics). No upload needed.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Load restaurant demo", type="primary", width='stretch'):
            AppState.set_demo_mode_chosen(True)
            AppState.set_browser_path("demo")
            AppState.set_demo_step(0)
            st.rerun()

    with col_upload:
        st.markdown(
            f"<div style='border:1px solid {LANDING_CARD_BORDER}; border-radius:8px; padding:24px; height:{LANDING_CARD_HEIGHT_PX}px;'>"
            "<h3 style='margin-top:0'>📁 Upload Your Data</h3>"
            "<p>Upload your own CSV and run DQ metrics interactively. "
            "Works in the browser — no installation required.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("Upload a CSV", width='stretch'):
            AppState.set_demo_mode_chosen(True)
            AppState.set_browser_path("own")
            AppState.set_step(0)
            st.rerun()


if __name__ == "__main__":
    main()
