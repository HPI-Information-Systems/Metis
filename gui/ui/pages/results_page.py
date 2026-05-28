"""Step 4: Browse, visualise and export results."""
from __future__ import annotations

import dataclasses
import json

import pandas as pd
import streamlit as st

from core.metric_catalog import get_catalog
from core.result_store import RunMetadata, SQLiteResultStore, get_active_store
from metis.utils.result import DQResult
from ui.icons import icon_for
from ui.state import AppState
from visualization import dispatch
from visualization.metric_palette import short_name
from visualization.renderers import (
    dimension_header,
    heatmap,
    metric_breakdown,
    multi_metric_worst_table,
    temporal,
)


# Tiebreak order when the active metrics have a mix of primary granularities.
# Column first because grouped bar charts compare cleanest; then row/cell
# (distribution-shaped); table last because KPI cards convey no comparison.
_GRANULARITY_PREFERENCE: tuple[str, ...] = ("column", "row", "cell", "table")

_EXPORT_WARN_THRESHOLD: int = 100_000

# Export state machine values stored in session_state under f"_export_state_{tag}":
#   "idle"      – default, show button
#   "queued"    – confirmed by user; next fragment run will show spinner then rerun
#   "computing" – spinner visible, heavy work running
#   "done"      – cache hit only, show download button


@st.cache_data(show_spinner=False)
def _cached_list_runs(_store) -> list[dict]:
    return [dataclasses.asdict(r) for r in _store.list_runs()]


@st.cache_data(show_spinner=False)
def _cached_get_metrics(_store, tag: str) -> list[str]:
    return _store.list_metrics_for_run(tag)


@st.cache_data(show_spinner=False)
def _cached_get_columns(_store, tag: str) -> list[str]:
    return _store.list_columns_for_run(tag)


@st.cache_data(show_spinner=False)
def _cached_get_heatmap(_store, tag: str) -> list[dict]:
    return _store.get_heatmap_data(tag)


@st.cache_data(show_spinner=False)
def _cached_load_temporal(_store, metric_name: str, dataset_name: str) -> list[dict]:
    """Load SQL-aggregated ``(run, column, mean_DQvalue)`` rows — not raw results."""
    return _store.load_temporal_data(metric_name, dataset_name)


@st.cache_data(show_spinner=False)
def _cached_export_json(_store, tag: str) -> bytes:
    return _store.export_json(tag)


def _dimension_for(metric_name: str) -> str:
    """
    Resolve the dimension for a metric, falling back to the registry name prefix.

    :param metric_name: Full metric name like ``completeness_nullRatio``.
    :return: The dimension string (e.g. ``"Completeness"``).
    """
    info = get_catalog().get(metric_name)
    if info:
        return info.dimension
    return metric_name.split("_", 1)[0].capitalize() or "Other"


def render(store: SQLiteResultStore, key_prefix: str = "") -> None:
    """
    Render the Results page: run selector, dimension breakdowns, exports.

    :param store: The active result store.
    :param key_prefix: Streamlit widget key prefix to isolate the own/demo wizards.
    :return: None.
    """
    st.header("Results")

    runs = _cached_list_runs(store)

    if not runs and not AppState.get_last_experiment_tag():
        st.info("No results yet. Run an assessment in the Compute step first.")
        return

    selected_tag = _render_run_header(runs, store, key_prefix)
    if selected_tag is None:
        return

    metrics = _cached_get_metrics(store, selected_tag)

    if not metrics:
        st.warning("No results found for this run.")
        return

    dataset_cols = _cached_get_columns(store, selected_tag)
    _render_run_subheader(runs, selected_tag)

    by_dim: dict[str, list[str]] = {}
    for name in metrics:
        dim = _dimension_for(name)
        by_dim.setdefault(dim, []).append(name)
    ordered_dims = sorted(by_dim.keys())

    if not ordered_dims:
        st.caption("No metrics to display.")
        return

    _render_results_tabs(store, selected_tag, metrics, dataset_cols, by_dim, ordered_dims, runs, key_prefix)


def _render_run_header(runs: list[dict], store: SQLiteResultStore, key_prefix: str) -> str | None:
    """
    Render the run selector + export + import row at the top of the page.

    :param runs: All known runs.
    :param store: The active result store.
    :param key_prefix: Widget key prefix.
    :return: The selected experiment tag, or ``None`` if no run is selectable.
    """
    col_sel, col_exp, col_imp = st.columns([4, 1, 1])

    with col_sel:
        selected_tag = _render_run_selector(runs, key_prefix=key_prefix)

    with col_exp:
        if selected_tag:
            run_info = next((r for r in runs if r["experiment_tag"] == selected_tag), None)
            result_count = run_info["result_count"] if run_info else 0
            _render_export_fragment(store, selected_tag, result_count, key_prefix=key_prefix)

    with col_imp:
        if st.button("Import JSON", width='stretch', key=f"{key_prefix}import_btn"):
            st.session_state[f"{key_prefix}show_import"] = True

    if st.session_state.get(f"{key_prefix}show_import"):
        _render_import(key_prefix)

    return selected_tag


def _render_run_subheader(runs: list[dict], selected_tag: str) -> None:
    """
    Render the run-level subheader showing tag, count, and dataset name.

    :param runs: All known runs.
    :param selected_tag: The selected experiment tag.
    :return: None.
    """
    run_summary = next((r for r in runs if r["experiment_tag"] == selected_tag), None)
    if run_summary:
        st.subheader(
            f"Run: `{selected_tag}` — {run_summary['result_count']:,} DQ Measurement Results"
            f"  ·  {run_summary['dataset_name']}"
        )
    else:
        st.subheader(f"Run: `{selected_tag}`")


def _render_results_tabs(
    store: SQLiteResultStore,
    selected_tag: str,
    metrics: list[str],
    dataset_cols: list[str],
    by_dim: dict[str, list[str]],
    ordered_dims: list[str],
    runs: list[dict],
    key_prefix: str,
) -> None:
    """
    Render the main tab strip: Overview + per-dimension + Comparison-over-time.

    :param store: The active result store.
    :param selected_tag: The selected experiment tag.
    :param metrics: All metrics in the run.
    :param dataset_cols: Column names of the original dataset.
    :param by_dim: Metrics grouped by dimension.
    :param ordered_dims: Dimensions in their tab order.
    :param runs: All known runs.
    :param key_prefix: Widget key prefix.
    :return: None.
    """
    tab_labels = (
        ["Overview"]
        + [f"{icon_for(dim)} {dim} ({len(by_dim[dim])})" for dim in ordered_dims]
        + [":material/timeline: Comparison over time"]
    )

    # Push the last tab ("Comparison over time") to the far right of the tab
    # bar via flex margin-left:auto. The tab list contains decorative siblings
    # (tab-highlight, tab-border), so we must target buttons by data-testid
    # rather than :last-child. Scoped to this container only.
    st.markdown(
        f"""
        <style>
        .st-key-{key_prefix}results_tabs [data-baseweb="tab-list"] button[data-testid="stTab"]:last-of-type {{
            margin-left: auto !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key=f"{key_prefix}results_tabs"):
        tabs = st.tabs(tab_labels)

        with tabs[0]:
            _render_overview_tab(selected_tag, metrics, dataset_cols, by_dim, key_prefix)

        for tab, dim in zip(tabs[1:1 + len(ordered_dims)], ordered_dims):
            with tab:
                _render_dimension_tab(selected_tag, dim, sorted(by_dim[dim]), dataset_cols, key_prefix)

        with tabs[-1]:
            _render_temporal_tab(selected_tag, metrics, runs, key_prefix)


@st.fragment
def _render_overview_tab(
    tag: str,
    metrics: list[str],
    dataset_cols: list[str],
    by_dim: dict[str, list[str]],
    key_prefix: str = "",
) -> None:
    """
    Render the Overview tab: dimension KPI strip + cross-dimension heatmap.

    :param tag: The selected experiment tag.
    :param metrics: All metrics in the run.
    :param dataset_cols: Column names of the original dataset.
    :param by_dim: Metrics grouped by dimension.
    :param key_prefix: Widget key prefix.
    :return: None.
    """
    store = get_active_store()

    dim_cards = []
    for dim in sorted(by_dim.keys()):
        means: list[float] = []
        for m in by_dim[dim]:
            s = dispatch._cached_metric_summary(store, tag, m)
            mean = s.get("mean_score")
            if mean is not None:
                means.append(float(mean))
        dim_mean = sum(means) / len(means) if means else None
        dim_cards.append((dim, dim_mean, len(by_dim[dim])))

    if dim_cards:
        cols = st.columns(len(dim_cards))
        for col, (dim, dim_mean, n) in zip(cols, dim_cards):
            col.metric(
                f"{icon_for(dim)} {dim}",
                f"{dim_mean:.1%}" if dim_mean is not None else "—",
                delta=f"{n} metric{'s' if n != 1 else ''}",
                delta_color="off",
            )

    st.divider()

    st.markdown("**Column × metric heatmap**")
    heatmap_data = _cached_get_heatmap(store, tag)
    heatmap_metrics = {row["dq_metric"] for row in heatmap_data}
    excluded_by_dim: dict[str, list[str]] = {}
    for name in metrics:
        if name in heatmap_metrics:
            continue
        excluded_by_dim.setdefault(_dimension_for(name), []).append(name)
    if excluded_by_dim:
        text = (
            "Not column-level, so not shown in the heatmap — see the matching "
            "dimension tab:"
        )
        for dim in sorted(excluded_by_dim):
            names = ", ".join(f"`{n}`" for n in sorted(excluded_by_dim[dim]))
            text += f"  \n&nbsp;&nbsp;&nbsp;&nbsp;• {dim} ({names})"
        st.caption(text)
    heatmap.render(pd.DataFrame(heatmap_data), dataset_cols)


@st.fragment
def _render_temporal_tab(
    tag: str,
    metrics: list[str],
    runs: list[dict],
    key_prefix: str = "",
) -> None:
    """
    Render the Comparison-over-time tab: multi-metric trends across runs.

    :param tag: The currently selected experiment tag (used to seed defaults).
    :param metrics: Metrics in the current run.
    :param runs: All known runs.
    :param key_prefix: Widget key prefix.
    :return: None.
    """
    store = get_active_store()

    if len(runs) < 2:
        st.info(
            "At least two runs are required to compare metric values over time."
        )
        return

    st.markdown("**Compare runs**")

    selected_tags = _render_temporal_run_picker(runs, tag, key_prefix)
    if not selected_tags:
        st.caption("Select at least one run to see the trend.")
        return

    selected_metrics = _render_temporal_metric_picker(store, selected_tags, tag, key_prefix)
    if not selected_metrics:
        return

    selected_set = set(selected_tags)
    per_metric = {
        m: [
            r
            for r in _cached_load_temporal(store, m, "")
            if r.get("tag") in selected_set
        ]
        for m in selected_metrics
    }
    temporal.render_multi_metric(per_metric)


def _render_temporal_run_picker(runs: list[dict], current_tag: str, key_prefix: str) -> list[str]:
    """
    Render the multiselect that picks which runs to overlay on the temporal chart.

    Defaults to runs sharing the current run's dataset.

    :param runs: All known runs.
    :param current_tag: The currently selected experiment tag.
    :param key_prefix: Widget key prefix.
    :return: The list of selected run tags.
    """
    current_run = next((r for r in runs if r["experiment_tag"] == current_tag), None)
    current_dataset = (current_run or {}).get("dataset_name", "")

    auto_tags = [
        r["experiment_tag"]
        for r in runs
        if current_dataset and r.get("dataset_name") == current_dataset
    ] or [current_tag]

    all_tags = [r["experiment_tag"] for r in runs]
    tag_labels = {
        r["experiment_tag"]: (
            f"{r['experiment_tag']}  ·  {r.get('dataset_name', '?')}  ·  "
            f"{(r.get('timestamp') or '')[:19]}"
        )
        for r in runs
    }

    return st.multiselect(
        "Runs to include",
        options=all_tags,
        default=auto_tags,
        format_func=lambda t: tag_labels.get(t, t),
        key=f"{key_prefix}temporal_tab_runs_{current_tag}",
        help="Auto-selected runs share the current run's dataset; add or remove "
             "any others to override.",
    )


def _render_temporal_metric_picker(
    store: SQLiteResultStore,
    selected_tags: list[str],
    current_tag: str,
    key_prefix: str,
) -> list[str]:
    """
    Render the multiselect that picks which metrics to overlay on the temporal chart.

    Options are sorted and prefixed by dimension so the dropdown reads as
    visually grouped. Initial default is one metric per dimension. Stale
    entries are pruned when their metric is missing from the union of selected
    runs.

    :param store: The active result store.
    :param selected_tags: Run tags currently selected on the chart.
    :param current_tag: The currently active experiment tag (used to namespace state).
    :param key_prefix: Widget key prefix.
    :return: The list of selected metric names (empty if none and a hint is shown).
    """
    available_metrics: set[str] = set()
    for t in selected_tags:
        available_metrics.update(_cached_get_metrics(store, t))
    if not available_metrics:
        st.caption("No metrics found in the selected runs.")
        return []

    metrics_by_dim: dict[str, list[str]] = {}
    for m in sorted(available_metrics):
        metrics_by_dim.setdefault(_dimension_for(m), []).append(m)

    metric_to_dim = {m: d for d, ms in metrics_by_dim.items() for m in ms}
    ordered_options: list[str] = []
    for dim in sorted(metrics_by_dim.keys()):
        ordered_options.extend(metrics_by_dim[dim])

    master_key = f"{key_prefix}temporal_tab_metrics_{current_tag}"

    valid_metrics = set(ordered_options)
    if master_key in st.session_state:
        cleaned = [m for m in st.session_state[master_key] if m in valid_metrics]
        if cleaned != st.session_state[master_key]:
            st.session_state[master_key] = cleaned
    else:
        st.session_state[master_key] = [
            ms[0] for ms in metrics_by_dim.values() if ms
        ]

    def _fmt(metric_name: str) -> str:
        dim = metric_to_dim.get(metric_name, "Other")
        return f"{dim} · {short_name(metric_name)}"

    selected_metrics = st.multiselect(
        "Metrics to overlay",
        options=ordered_options,
        format_func=_fmt,
        key=master_key,
    )

    if not selected_metrics:
        st.caption("Select one or more metrics above to see the trend.")
    return selected_metrics


@st.fragment
def _render_dimension_tab(
    tag: str,
    dimension: str,
    metric_names: list[str],
    dataset_cols: list[str],
    key_prefix: str = "",
) -> None:
    """
    Render the per-dimension tab: header KPI + adaptive comparison + per-metric expanders.

    :param tag: The selected experiment tag.
    :param dimension: Dimension being rendered.
    :param metric_names: Metrics in this dimension.
    :param dataset_cols: Column names of the original dataset.
    :param key_prefix: Widget key prefix.
    :return: None.
    """
    store = get_active_store()

    summaries: dict[str, dict] = {
        m: dispatch._cached_metric_summary(store, tag, m) for m in metric_names
    }

    pills_key = f"{key_prefix}dim_chips_{tag}_{dimension}"
    active = st.pills(
        "Metrics",
        options=metric_names,
        default=metric_names,
        selection_mode="multi",
        key=pills_key,
        format_func=short_name,
        label_visibility="collapsed",
    ) or []

    if not active:
        st.info("Select at least one metric to compare.")
        return

    active_summaries = {m: summaries[m] for m in active}
    dimension_header.render(dimension, active_summaries)

    # Skip the comparison view when only one metric is active — the per-metric
    # view below is the only meaningful display, and a one-bar "comparison"
    # would be redundant.
    single_metric = len(active) == 1
    if not single_metric:
        st.divider()
        if dimension == "Consistency":
            _render_consistency_comparison(
                store, tag, active, active_summaries, key_prefix,
            )
        else:
            _render_primary_comparison(
                store, tag, dimension, active, active_summaries, dataset_cols, key_prefix,
            )

    st.divider()
    st.subheader("Single-metric view")
    for metric_name in active:
        s = summaries[metric_name]
        granularity = s.get("primary_granularity", "")
        n = s.get("count", 0)
        with st.expander(
            f"**{metric_name}**  ·  {granularity}  ·  {n:,} DQ Measurement Results",
            expanded=single_metric,
        ):
            st.markdown(f"### {metric_name}")
            dispatch.render(store, tag, metric_name, dataset_cols, key_prefix=key_prefix)


def _render_primary_comparison(
    store,
    tag: str,
    dimension: str,
    active: list[str],
    active_summaries: dict[str, dict],
    dataset_cols: list[str],
    key_prefix: str = "",
) -> None:
    """
    Pick one primary chart per granularity present, in preference order.

    :param store: The active result store.
    :param tag: The selected experiment tag.
    :param dimension: Dimension being rendered.
    :param active: Active metric names.
    :param active_summaries: Per-metric summaries keyed by metric name.
    :param dataset_cols: Column names of the original dataset.
    :param key_prefix: Widget key prefix.
    :return: None.
    """
    by_gran: dict[str, list[str]] = {}
    for m in active:
        gran = active_summaries[m].get("primary_granularity", "table")
        by_gran.setdefault(gran, []).append(m)

    # Skip a table-granularity bucket containing a single metric: the resulting
    # comparison is one KPI tile, which already appears in the dimension header
    # and the per-metric view below — purely redundant.
    visible = {
        gran: ms for gran, ms in by_gran.items()
        if not (gran == "table" and len(ms) == 1)
    }

    rendered_any = False
    for gran in _GRANULARITY_PREFERENCE:
        metrics_for_gran = visible.get(gran)
        if not metrics_for_gran:
            continue
        rendered_any = True
        cache_key = f"{key_prefix}{tag}::{dimension}::{gran}"

        if len(visible) > 1:
            st.caption(f"{gran.capitalize()}-level comparison")

        if gran == "column":
            _render_column_comparison(store, tag, metrics_for_gran, dataset_cols, cache_key)
        elif gran == "cell":
            _render_cell_comparison(store, tag, metrics_for_gran, cache_key, dataset_cols)
        elif gran == "row":
            _render_row_comparison(store, tag, metrics_for_gran, cache_key)
        elif gran == "table":
            _render_table_comparison(store, tag, metrics_for_gran)

    if not rendered_any:
        st.caption("No comparable scores in this dimension.")


def _render_consistency_comparison(
    store,
    tag: str,
    active: list[str],
    active_summaries: dict[str, dict],
    key_prefix: str = "",
) -> None:
    """
    Render the comparison view for the Consistency dimension.

    Each consistency metric contributes its per-sub-unit breakdown rows (FD
    descriptor for table-granularity rule-count metrics, column for
    cell-granularity rule-based metrics, etc.). The renderer drops any metric
    whose sub-units don't overlap with another's automatically.

    :param store: The active result store.
    :param tag: The selected experiment tag.
    :param active: Active metric names.
    :param active_summaries: Per-metric summaries keyed by metric name.
    :param key_prefix: Widget key prefix.
    :return: None.
    """
    per_metric_rows: dict[str, list[dict]] = {}
    for metric_name in active:
        s = active_summaries.get(metric_name, {})
        gran = s.get("primary_granularity", "table")
        rows = _consistency_breakdown_rows(store, tag, metric_name, gran, s)
        if rows:
            per_metric_rows[metric_name] = rows

    cache_key = f"{key_prefix}{tag}::Consistency::breakdown"
    metric_breakdown.render(per_metric_rows, cache_key)


def _consistency_breakdown_rows(
    store, tag: str, metric_name: str, gran: str, summary: dict
) -> list[dict]:
    """
    Pull a list of (sub-unit label, DQvalue) rows for one consistency metric.

    :param store: The active result store.
    :param tag: The selected experiment tag.
    :param metric_name: Metric name.
    :param gran: Primary granularity (``table``, ``cell``, ``column`` or other).
    :param summary: The metric's summary dict.
    :return: A list of ``{sub, DQvalue}`` rows.
    """
    if gran == "table":
        results = dispatch._cached_table_results(store, tag, metric_name)
        out: list[dict] = []
        for i, r in enumerate(results):
            expl = r.get("DQexplanation") or {}
            fd_key = next((k for k in expl.keys() if ":" in k), None)
            if fd_key:
                det, dep = fd_key.split(":", 1)
                sub = f"{det} → {dep}"
            else:
                sub = f"#{i + 1}"
            out.append({"sub": sub, "DQvalue": r.get("DQvalue")})
        return out

    if gran == "cell":
        agg = dispatch._cached_column_agg(store, tag, metric_name)
        if agg is None or agg.empty:
            return []
        return [
            {"sub": str(r["column"]), "DQvalue": r.get("mean_score")}
            for _, r in agg.iterrows()
        ]

    if gran == "column":
        res = dispatch._cached_column_results(store, tag, metric_name)
        if res is None or res.empty:
            return []
        return [
            {"sub": str(r["column"]), "DQvalue": r.get("DQvalue")}
            for _, r in res.iterrows()
        ]

    mean = summary.get("mean_score")
    if mean is None:
        return []
    return [{"sub": "(mean)", "DQvalue": mean}]


def _render_column_comparison(
    store, tag: str, metric_names: list[str], dataset_cols: list[str], cache_key: str
) -> None:
    per_metric_rows: dict[str, list[dict]] = {}
    for m in metric_names:
        df = dispatch._cached_column_results(store, tag, m)
        if df is None or df.empty:
            continue
        per_metric_rows[m] = [
            {"sub": str(r["column"]), "DQvalue": r.get("DQvalue")}
            for _, r in df.iterrows()
        ]
    metric_breakdown.render(per_metric_rows, cache_key, sub_order=dataset_cols)


def _render_cell_comparison(
    store, tag: str, metric_names: list[str], cache_key: str, dataset_cols: list[str]
) -> None:
    """
    Render the cell-granularity comparison: per-column means + worst-cells table.

    The histogram view of each metric still lives in the single-metric expander.

    :param store: The active result store.
    :param tag: The selected experiment tag.
    :param metric_names: Metrics being compared.
    :param cache_key: Stable cache key for the chart builders.
    :param dataset_cols: Column names of the original dataset.
    :return: None.
    """
    per_metric_rows: dict[str, list[dict]] = {}
    for m in metric_names:
        df = dispatch._cached_column_agg(store, tag, m)
        if df is None or df.empty:
            continue
        per_metric_rows[m] = [
            {"sub": str(r["column"]), "DQvalue": r.get("mean_score")}
            for _, r in df.iterrows()
        ]
    metric_breakdown.render(per_metric_rows, f"{cache_key}::bar", sub_order=dataset_cols)

    # The worst-cells table answers a different question (where are the actual
    # offenders) and has no overlap problem (one column per metric in a wide pivot).
    per_metric_worst = {
        m: dispatch._cached_worst(store, tag, m) for m in metric_names
    }
    multi_metric_worst_table.render(per_metric_worst, "cell", cache_key)


def _render_row_comparison(
    store, tag: str, metric_names: list[str], cache_key: str
) -> None:
    per_metric_worst = {
        m: dispatch._cached_worst(store, tag, m, "row") for m in metric_names
    }
    multi_metric_worst_table.render(per_metric_worst, "row", cache_key)


def _render_table_comparison(
    store, tag: str, metric_names: list[str]
) -> None:
    """
    Render a compact per-metric KPI strip for table-granularity metrics.

    Per-result cards live in the single-metric view to avoid blowing up the
    dimension comparison when one metric has many results (e.g.
    ``countFDViolations`` produces one result per functional dependency).

    :param store: The active result store.
    :param tag: The selected experiment tag.
    :param metric_names: Metrics being compared.
    :return: None.
    """
    cols = st.columns(max(1, len(metric_names)))
    for col, metric_name in zip(cols, metric_names):
        s = dispatch._cached_metric_summary(store, tag, metric_name)
        mean = s.get("mean_score")
        count = int(s.get("count", 0) or 0)
        pct = f"{mean:.1%}" if mean is not None else "—"
        col.metric(
            short_name(metric_name),
            pct,
            delta=f"{count} result{'s' if count != 1 else ''}",
            delta_color="off",
        )


@st.dialog("Export JSON")
def _export_confirm_dialog(tag: str, result_count: int, key_prefix: str = "") -> None:
    """
    Render the warning dialog shown for large exports.

    Sets state to ``"queued"`` on confirm so the dialog's ``st.rerun()`` closes
    the modal before any computation begins.

    :param tag: The experiment tag being exported.
    :param result_count: The total number of results in the run.
    :param key_prefix: Widget key prefix used by the calling fragment so the
        state transition lands on the same session key the fragment reads.
    :return: None.
    """
    st.warning(
        f"This run contains **{result_count:,} results**. "
        "Generating the JSON file may take several minutes.",
        icon="⚠️",
    )
    col_cancel, col_confirm = st.columns(2)
    with col_cancel:
        if st.button("Cancel", width="stretch", key=f"export_dlg_cancel_{key_prefix}{tag}"):
            st.rerun()
    with col_confirm:
        if st.button("Export JSON", type="primary", width="stretch", key=f"export_dlg_confirm_{key_prefix}{tag}"):
            st.session_state[f"_export_state_{key_prefix}{tag}"] = "queued"
            st.rerun()


@st.fragment
def _render_export_fragment(
    store: SQLiteResultStore, tag: str, result_count: int, key_prefix: str = "",
) -> None:
    """
    Render the lazy, two-phase JSON export button.

    Phase 1 ("queued"): render a spinner element. this ForwardMsg is flushed
    to the browser immediately, then call ``st.rerun(scope="fragment")`` to
    stop this run. The browser now shows the spinner (and the dialog is gone).
    Phase 2 ("computing"): the fragment reruns, calls ``_cached_export_json``,
    and updates the status to ``complete`` when done.

    :param store: The active result store.
    :param tag: The experiment tag being exported.
    :param result_count: Total number of results in the run.
    :param key_prefix: Widget key prefix.
    :return: None.
    """
    state_key = f"_export_state_{key_prefix}{tag}"
    state = st.session_state.get(state_key, "idle")

    if state == "idle":
        if st.button("Export JSON", width="stretch", key=f"export_btn_{key_prefix}{tag}"):
            if result_count > _EXPORT_WARN_THRESHOLD:
                _export_confirm_dialog(tag, result_count, key_prefix)
            else:
                st.session_state[state_key] = "queued"
                st.rerun(scope="fragment")

    elif state == "queued":
        # Send a spinner delta to the browser NOW (before any heavy work),
        # then immediately rerun the fragment to do the actual computation.
        st.status("Starting export…", state="running")
        st.session_state[state_key] = "computing"
        st.rerun(scope="fragment")

    elif state == "computing":
        with st.status("Generating export…") as status:
            data = _cached_export_json(store, tag)
            st.session_state[state_key] = "done"
            status.update(
                label="Export ready — click to download",
                state="complete",
                expanded=False,
            )
        st.download_button(
            "Download JSON",
            data=data,
            file_name=f"{tag}_results.json",
            mime="application/json",
            width="stretch",
            key=f"export_download_{key_prefix}{tag}",
        )

    elif state == "done":
        data = _cached_export_json(store, tag)
        st.download_button(
            "Download JSON",
            data=data,
            file_name=f"{tag}_results.json",
            mime="application/json",
            width="stretch",
            key=f"export_download_{key_prefix}{tag}",
        )


@st.fragment
def _render_import(key_prefix: str = "") -> None:
    """
    Render the "Import results from JSON" expander panel.

    :param key_prefix: Widget key prefix.
    :return: None.
    """
    with st.expander("Import results from JSON", expanded=True):
        st.write("Import a previously exported results file to restore a run.")
        uploaded = st.file_uploader(
            "Results JSON file",
            type=["json"],
            key=f"{key_prefix}import_uploader",
        )
        tag_override = st.text_input(
            "Experiment tag (leave blank to keep original)",
            key=f"{key_prefix}import_tag",
        )
        if uploaded and st.button("Import", key=f"{key_prefix}do_import"):
            try:
                raw = json.loads(uploaded.read())
                if isinstance(raw, dict) and isinstance(raw.get("results"), list):
                    raw = raw["results"]
                if not isinstance(raw, list):
                    st.error(
                        "Expected a JSON array of result objects, or an "
                        "object with a 'results' array."
                    )
                    return
                tag = tag_override.strip() or (raw[0].get("experimentTag") if raw else "imported")
                dq_results = [
                    DQResult(
                        timestamp=pd.Timestamp(r["timestamp"]) if r.get("timestamp") else pd.Timestamp.now(),
                        DQdimension=r.get("DQdimension", ""),
                        DQmetric=r.get("DQmetric", ""),
                        DQgranularity=r.get("DQgranularity", ""),
                        DQvalue=r.get("DQvalue"),
                        DQexplanation=r.get("DQexplanation"),
                        runtime=r.get("runtime"),
                        tableName=r.get("tableName"),
                        columnNames=r.get("columnNames"),
                        rowIndex=r.get("rowIndex"),
                        configJson=r.get("configJson"),
                    )
                    for r in raw
                ]
                get_active_store().save_run(dq_results, RunMetadata(
                    experiment_tag=tag,
                    dataset_name=raw[0].get("dataset", "imported") if raw else "imported",
                ))
                AppState.clear_results_caches()
                st.success(f"Imported {len(dq_results):,} results under tag `{tag}`.")
                st.session_state[f"{key_prefix}show_import"] = False
                st.rerun()
            except Exception as exc:
                st.error(f"Import failed: {exc}")


def _render_run_selector(runs, key_prefix: str = "") -> str | None:
    """
    Render the run selector dropdown at the top of the page.

    :param runs: All known runs.
    :param key_prefix: Widget key prefix.
    :return: The selected experiment tag, or the last-known tag if no run is selectable.
    """
    last_tag = AppState.get_last_experiment_tag()

    if not runs:
        return last_tag

    options = [r["experiment_tag"] for r in runs]
    labels = {
        r["experiment_tag"]: (
            f"{r['experiment_tag']}  ·  {r['dataset_name']}  ·  "
            f"{r['result_count']:,} results  ·  {r['timestamp'][:19]}"
        )
        for r in runs
    }

    default_idx = 0
    if last_tag and last_tag in options:
        default_idx = options.index(last_tag)

    return st.selectbox(
        "Select run",
        options=options,
        index=default_idx,
        format_func=lambda t: labels.get(t, t),
        label_visibility="collapsed",
        key=f"{key_prefix}results_run_selector",
    )
