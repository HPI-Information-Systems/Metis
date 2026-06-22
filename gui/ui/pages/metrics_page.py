"""Step 2: Select metrics and configure them."""
from __future__ import annotations

import math
import os
import re

import streamlit as st

from core.metric_catalog import (
    MetricInfo,
    get_catalog,
    get_compute_blockers,
    get_metrics_by_dimension,
)
from demo.demo_metric_configs import DEMO_CONFIG_DISPLAY, DEMO_METRICS
from metis.utils.dq_granularity import DQGranularity
from ui.components.config_editors import (
    callable_editor,
    simple_editor,
    timeliness_editor,
)
from theme import HPI_ORANGE
from ui.icons import icon_for
from ui.state import AppState

_FILTER_OPTIONS: list[str] = ["All", "No config", "Needs config", "Python rules", "FD rules"]

# HyFD / AIDFD line format:  [table.col1, table.col2]->table.col3
_HYFD_RE: re.Pattern[str] = re.compile(r"\[([^\]]+)\]->([^\s\[#]+)")

CARD_BORDER_CALLABLE: str = HPI_ORANGE
CARD_BORDER_FD: str = "#7b2d8b"
CARD_BORDER_REFERENCE: str = "#2778c4"
CARD_BORDER_CONFIG_REQUIRED: str = "#7b2d8b"
CARD_BORDER_NEUTRAL: str = "#e0e0e0"

ROW_CAP_FALLBACK_STEP: int = 1_000
ROW_CAP_FALLBACK_MAX: int = 500_000


def render(demo_mode: bool = False) -> None:
    """
    Render the Metrics selection page (or its read-only demo variant).

    :param demo_mode: When True, render the demo-mode selector with read-only configs.
    :return: None.
    """
    if demo_mode:
        _render_demo_metrics()
        return

    st.session_state.setdefault(AppState.widget_key("metrics", "search"), "")
    st.session_state.setdefault(AppState.widget_key("metrics", "filter"), "All")

    df = AppState.get_df()
    by_dim = get_metrics_by_dimension()
    all_metrics_flat = [m for metrics in by_dim.values() for m in metrics]
    selected = _reconcile_selected_with_widgets(
        list(AppState.get_selected_metrics()),
        [m.name for m in all_metrics_flat],
    )

    st.header("Select metrics")
    col_search, col_summary = st.columns([4, 1], vertical_alignment="bottom")
    with col_search:
        search = st.text_input(
            "Search",
            placeholder="Filter by name or dimension…",
            icon=":material/search:",
            label_visibility="collapsed",
            key=AppState.widget_key("metrics", "search"),
        )
    with col_summary:
        st.caption(f"{len(selected)} / {len(all_metrics_flat)} selected")

    active_filter = st.pills(
        "Filter",
        options=_FILTER_OPTIONS,
        default="All",
        selection_mode="single",
        label_visibility="collapsed",
        key=AppState.widget_key("metrics", "filter"),
    ) or "All"

    all_filtered = _apply_filters(all_metrics_flat, search, active_filter)
    filtered_by_dim = {
        dim: [m for m in all_filtered if m.dimension == dim]
        for dim in by_dim
    }
    is_searching = bool(search.strip()) or active_filter != "All"

    if not all_filtered:
        st.info("No metrics match. Clear the search or change the filter.")
        AppState.set_selected_metrics(selected)
        return

    overview_label = ":material/list: Overview"
    tab_labels = [overview_label] + [_tab_label(dim) for dim in by_dim]
    _inject_tab_count_badges("metrics_tabs", by_dim, filtered_by_dim, selected, overview_offset=1)
    with st.container(key="metrics_tabs"):
        tabs = st.tabs(tab_labels)

    with tabs[0]:
        selectable_all = [m.name for m in all_filtered if not m.unavailable_reason]
        _render_select_controls(
            scope_id="all",
            selectable_names=selectable_all,
            selected=selected,
            select_label="Select all visible",
            deselect_label="Deselect all visible",
            select_icon=":material/check_box:",
            deselect_icon=":material/check_box_outline_blank:",
            button_type="secondary",
            peer_key_prefixes=("chk_", "chk_ov_"),
        )
        if is_searching:
            st.caption(f"{len(all_filtered)} metric(s) match")
        else:
            st.caption(f"{len(all_filtered)} metric(s) across {len(by_dim)} dimension(s)")
        _render_overview_list(all_filtered, selected, df)

    for tab, dimension in zip(tabs[1:], by_dim):
        with tab:
            metrics = filtered_by_dim.get(dimension, [])
            if not metrics:
                st.caption("No metrics match the current filter.")
                continue

            selectable = [m.name for m in metrics if not m.unavailable_reason]
            _render_select_controls(
                scope_id=dimension,
                selectable_names=selectable,
                selected=selected,
                select_label="All",
                deselect_label="None",
                select_icon=":material/select_all:",
                deselect_icon=":material/deselect:",
                button_type="tertiary",
                peer_key_prefixes=("chk_", "chk_ov_"),
            )

            for info in metrics:
                _render_metric_card(
                    info, selected, df,
                    key_prefix="chk_",
                    peer_prefix="chk_ov_",
                )

    AppState.set_selected_metrics(selected)
    blockers = get_compute_blockers(selected, AppState.get_metric_configs())
    if not selected:
        st.caption("Select at least one metric to continue.")
    elif blockers:
        lines = "\n".join(f"- {b}" for b in blockers)
        st.warning(f"**Fix the following before computing:**\n\n{lines}")
    else:
        st.caption(
            f"{len(selected)} metric{'s' if len(selected) != 1 else ''} selected. "
            "Click **Compute →** below to run."
        )


def _render_select_controls(
    scope_id: str,
    selectable_names: list[str],
    selected: list[str],
    *,
    select_label: str,
    deselect_label: str,
    select_icon: str,
    deselect_icon: str,
    button_type: str,
    peer_key_prefixes: tuple[str, ...] = ("chk_",),
) -> None:
    """
    Render a Select-all / Deselect-all button pair scoped to a list of metric names.

    :param scope_id: Stable identifier (e.g. ``"all"`` or a dimension name) used in the widget keys.
    :param selectable_names: Metric names the buttons may toggle (excludes unavailable metrics).
    :param selected: The mutable list of currently selected metric names.
    :param select_label: Label for the Select-all button.
    :param deselect_label: Label for the Deselect-all button.
    :param select_icon: Material icon for the Select-all button.
    :param deselect_icon: Material icon for the Deselect-all button.
    :param button_type: Streamlit button type (``"secondary"``, ``"tertiary"``, …).
    :return: None.
    """
    col_sel, col_desel, _ = st.columns([1, 1, 5] if button_type == "tertiary" else [1, 1, 4])
    sel_key = f"_metrics_selall_{scope_id}"
    desel_key = f"_metrics_deselall_{scope_id}"

    with col_sel:
        if st.button(
            select_label,
            key=sel_key,
            icon=select_icon,
            type=button_type,
            disabled=not selectable_names,
        ):
            for name in selectable_names:
                if name not in selected:
                    selected.append(name)
                for prefix in peer_key_prefixes:
                    st.session_state[f"{prefix}{name}"] = True
            AppState.set_selected_metrics(selected)
            st.rerun()

    with col_desel:
        if st.button(
            deselect_label,
            key=desel_key,
            icon=deselect_icon,
            type=button_type,
            disabled=not selectable_names,
        ):
            for name in selectable_names:
                if name in selected:
                    selected.remove(name)
                for prefix in peer_key_prefixes:
                    st.session_state[f"{prefix}{name}"] = False
            AppState.set_selected_metrics(selected)
            st.rerun()


def _tab_label(dimension: str) -> str:
    """
    Build a stable dimension tab label.

    The label intentionally omits any selection count: Streamlit identifies
    tabs by their label, so embedding a changing count would reset the active
    tab on every click. The visible ``(n_sel/n_total)`` badge is added by
    :func:`_inject_tab_count_badges` as a CSS ``::after`` pseudo-element, which
    lives entirely in the browser and never affects the underlying label.

    :param dimension: Dimension name.
    :return: A formatted tab label.
    """
    return f"{icon_for(dimension)} {dimension}"


def _inject_tab_count_badges(
    container_key: str,
    by_dim: dict[str, list[MetricInfo]],
    visible_by_dim: dict[str, list[MetricInfo]],
    selected: list[str],
    overview_offset: int = 0,
) -> None:
    """
    Inject a ``(n_sel/n_total)`` badge after each dimension tab via CSS ``::after``.

    Streamlit identifies tabs by their label string, so writing the count into
    the label would reset the active tab on every click. A pseudo-element is
    rendered purely client-side and leaves the label string untouched, so the
    active tab survives clicks and search-filter changes.

    The visible-count fallback mirrors the original tab-label behaviour: when
    every metric in a dimension is filtered out, the badge shows the
    dimension's full size rather than ``0/0``.

    :param container_key: ``st.container`` key under which the tabs are rendered.
        Used to scope the injected CSS so it doesn't bleed onto other tab strips.
    :param by_dim: All metrics grouped by dimension (defines tab order).
    :param visible_by_dim: Metrics currently passing the search/filter, grouped by dimension.
    :param selected: Currently selected metric names.
    :return: None.
    """
    rules: list[str] = []
    if overview_offset:
        n_total = sum(len(ms) for ms in visible_by_dim.values()) or sum(
            len(ms) for ms in by_dim.values()
        )
        n_sel = sum(
            1
            for dim, ms in (visible_by_dim or by_dim).items()
            for m in ms
            if m.name in selected
        )
        rules.append(
            f'.st-key-{container_key} [data-baseweb="tab-list"] '
            f'button[data-testid="stTab"]:nth-of-type(1)::after '
            f'{{ content: " ({n_sel}/{n_total})"; opacity: 0.6; '
            f'margin-left: 4px; font-weight: 400; }}'
        )
    for i, (dim, all_metrics) in enumerate(by_dim.items(), start=1 + overview_offset):
        visible = visible_by_dim.get(dim, [])
        n_sel = sum(1 for m in visible if m.name in selected)
        n_total = len(visible) if visible else len(all_metrics)
        rules.append(
            f'.st-key-{container_key} [data-baseweb="tab-list"] '
            f'button[data-testid="stTab"]:nth-of-type({i})::after '
            f'{{ content: " ({n_sel}/{n_total})"; opacity: 0.6; '
            f'margin-left: 4px; font-weight: 400; }}'
        )
    st.markdown("<style>" + "".join(rules) + "</style>", unsafe_allow_html=True)


def _render_overview_list(metrics: list[MetricInfo], selected: list[str], df) -> None:
    """
    Render every metric grouped by dimension with interactive checkboxes that
    stay in sync with the dimension-tab checkboxes via ``_sync_widget_state``.

    Inline config editors and the row-cap input are hidden here (``show_config
    =False``) — they live in the dimension tab so their widget keys don't
    collide with the Overview tab in the same render pass.

    :param metrics: Metrics to render (already filtered).
    :param selected: Mutable list of currently selected metric names.
    :param df: The active dataframe.
    :return: None.
    """
    by_dim: dict[str, list[MetricInfo]] = {}
    for info in metrics:
        by_dim.setdefault(info.dimension, []).append(info)
    for dimension in sorted(by_dim.keys()):
        st.markdown(f"##### {icon_for(dimension)} {dimension}")
        for info in by_dim[dimension]:
            _render_metric_card(
                info, selected, df,
                key_prefix="chk_ov_",
                peer_prefix="chk_",
                show_config=False,
            )


def _reconcile_selected_with_widgets(
    selected: list[str],
    metric_names: list[str],
    key_prefix: str = "chk_",
) -> list[str]:
    """
    Reconcile the saved selection with the live checkbox session state.

    The selection persisted in AppState reflects only the *previous* render's
    writes, so the global counter and per-tab caption would otherwise lag a
    click behind. Reading ``st.session_state`` here brings them in sync with
    the user's most recent click on the same render pass.

    :param selected: The selection loaded from AppState.
    :param metric_names: All metric names whose checkboxes may have toggled.
    :param key_prefix: Prefix used for the metric checkbox session-state keys.
    :return: A new selection list reflecting the current widget state.
    """
    result = list(selected)
    for name in metric_names:
        widget_value = st.session_state.get(f"{key_prefix}{name}")
        if widget_value is True and name not in result:
            result.append(name)
        elif widget_value is False and name in result:
            result.remove(name)
    return result


def _render_demo_metrics() -> None:
    st.header("Select metrics")
    st.info(
        "**Demo mode**: Pre-computed metrics are selectable. "
        "Configurations are shown read-only to illustrate what each metric requires."
    )

    by_dim = get_metrics_by_dimension()
    all_metrics_flat = [m for metrics in by_dim.values() for m in metrics]

    init_key = AppState.widget_key("metrics", "demo_init")
    if not AppState.get_demo_selected_metrics() and not st.session_state.get(init_key):
        AppState.set_demo_selected_metrics(list(DEMO_METRICS))
        st.session_state[init_key] = True

    demo_selected = _reconcile_selected_with_widgets(
        list(AppState.get_demo_selected_metrics()),
        [m.name for m in all_metrics_flat],
        key_prefix="demo_chk_",
    )

    n_available = sum(1 for m in all_metrics_flat if m.name in DEMO_METRICS)
    st.caption(f"{len(demo_selected)} / {n_available} selected")

    overview_label = ":material/list: Overview"
    tab_labels = [overview_label] + [_tab_label(dim) for dim in by_dim]
    _inject_tab_count_badges(
        "demo_metrics_tabs",
        by_dim,
        {dim: list(ms) for dim, ms in by_dim.items()},
        demo_selected,
        overview_offset=1,
    )
    with st.container(key="demo_metrics_tabs"):
        tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.caption(f"{len(all_metrics_flat)} metric(s) across {len(by_dim)} dimension(s)")
        for dimension in sorted(by_dim.keys()):
            st.markdown(f"##### {icon_for(dimension)} {dimension}")
            for info in by_dim[dimension]:
                _render_demo_metric_card(
                    info,
                    demo_selected,
                    DEMO_METRICS,
                    DEMO_CONFIG_DISPLAY,
                    key_prefix="demo_chk_ov_",
                    peer_prefix="demo_chk_",
                    show_config=False,
                )

    for tab, (dimension, metrics) in zip(tabs[1:], by_dim.items()):
        with tab:
            for info in metrics:
                _render_demo_metric_card(
                    info,
                    demo_selected,
                    DEMO_METRICS,
                    DEMO_CONFIG_DISPLAY,
                    key_prefix="demo_chk_",
                    peer_prefix="demo_chk_ov_",
                )

    AppState.set_demo_selected_metrics(demo_selected)

    if not demo_selected:
        st.caption("Select at least one metric to continue.")
    else:
        st.caption(
            f"{len(demo_selected)} metric{'s' if len(demo_selected) != 1 else ''} selected — "
            "click **Compute →** below to load pre-computed results."
        )


def _render_demo_metric_card(
    info: MetricInfo,
    demo_selected: list[str],
    available_metrics: list[str],
    config_display: dict,
    *,
    key_prefix: str = "demo_chk_",
    peer_prefix: str | None = None,
    show_config: bool = True,
) -> None:
    is_available = info.name in available_metrics
    border_color = _card_border_color(info) if is_available else CARD_BORDER_NEUTRAL

    with st.container(border=True):
        st.markdown(
            f'<div style="border-left:4px solid {border_color};'
            f'margin:-8px -12px 8px -12px;padding:0;border-radius:2px 0 0 2px"></div>',
            unsafe_allow_html=True,
        )
        col_check, col_badges = st.columns([3, 2])

        primary_key = f"{key_prefix}{info.name}"
        cb_kwargs: dict = {}
        if peer_prefix is not None:
            cb_kwargs["on_change"] = _sync_widget_state
            cb_kwargs["args"] = (primary_key, f"{peer_prefix}{info.name}")

        with col_check:
            new_val = st.checkbox(
                _format_name(info.name),
                value=info.name in demo_selected,
                key=primary_key,
                disabled=not is_available,
                **cb_kwargs,
            )
            if is_available:
                if new_val and info.name not in demo_selected:
                    demo_selected.append(info.name)
                elif not new_val and info.name in demo_selected:
                    demo_selected.remove(info.name)
            if info.description:
                st.caption(info.description)
            if not is_available:
                st.caption(":material/lock: Not available in demo")

        with col_badges:
            _render_badges(info)
            if is_available:
                st.badge("Precomputed", color="green", icon=":material/bolt:")

        if show_config and new_val and is_available and info.name in config_display:
            st.divider()
            _render_demo_config_display(config_display[info.name])


def _render_demo_config_display(config_info: dict) -> None:
    """
    Show metric configuration read-only (for demo mode).

    :param config_info: A descriptor dict from ``DEMO_CONFIG_DISPLAY``.
    :return: None.
    """
    config_type = config_info.get("type")

    if config_type == "fd":
        st.caption("**Functional dependency rules (pre-configured):**")
        for det, deps in config_info["rules"].items():
            st.write(f"`{det}` → {', '.join(f'`{d}`' for d in deps)}")

    elif config_type == "callable":
        desc = config_info.get("description", "")
        src = config_info.get("source_file", "")
        if desc:
            st.caption(desc)
        st.caption(f"**Python rules** from `{os.path.basename(src)}`")
        if src and os.path.exists(src):
            with st.expander("View rule source", expanded=False):
                with open(src) as f:
                    st.code(f.read(), language="python")

    elif config_type == "timeliness":
        st.caption("**Timeliness columns (pre-configured):**")
        for col, cfg in config_info.get("columns", {}).items():
            st.write(
                f"`{col}`: decline rate **{cfg['decline_rate']}**, "
                f"ingestion date from `{cfg['ingestion_date_column']}`"
            )

    elif config_type == "datarange":
        st.caption("**Expected value ranges (pre-configured):**")
        for col, (low, high) in config_info.get("intervals", {}).items():
            st.write(f"`{col}`: between **{low}** and **{high}**")


def _apply_filters(metrics: list, search: str, active_filter: str) -> list:
    """
    Apply the search box and the active pill filter to a list of metrics.

    :param metrics: Source list of :class:`MetricInfo`.
    :param search: Free-text query (case-insensitive substring match on name + dimension).
    :param active_filter: Label of the active pill (``"All"`` disables the pill filter).
    :return: A new list with metrics that match both filters.
    """
    result = metrics
    if search:
        q = search.lower()
        result = [m for m in result if q in m.name.lower() or q in m.dimension.lower()]
    if active_filter == "No config":
        result = [m for m in result if not any([
            m.callable_config, m.fd_json_config, m.config_required, m.requires_reference,
        ])]
    elif active_filter == "Needs config":
        result = [m for m in result if m.config_required or m.fd_json_config or m.callable_config]
    elif active_filter == "Python rules":
        result = [m for m in result if m.callable_config]
    elif active_filter == "FD rules":
        result = [m for m in result if m.fd_json_config]
    return result


def _card_border_color(info: MetricInfo) -> str:
    """Pick the left-border color of a metric card based on its config kind."""
    if info.callable_config:
        return CARD_BORDER_CALLABLE
    if info.fd_json_config:
        return CARD_BORDER_FD
    if info.requires_reference:
        return CARD_BORDER_REFERENCE
    if info.config_required:
        return CARD_BORDER_CONFIG_REQUIRED
    return CARD_BORDER_NEUTRAL


def _sync_widget_state(source_key: str, peer_key: str) -> None:
    """
    Mirror a checkbox's new value to its peer in another tab.

    Streamlit renders every tab in the same script run, so the Overview tab
    and the dimension tabs each create their own checkbox per metric. This
    callback keeps both widgets' session_state in sync so a click in either
    place takes effect immediately on the next render.

    :param source_key: The widget key that just changed.
    :param peer_key: The peer widget key that should mirror the new value.
    :return: None.
    """
    if source_key in st.session_state:
        st.session_state[peer_key] = st.session_state[source_key]


def _render_metric_card(
    info: MetricInfo,
    selected: list[str],
    df,
    *,
    key_prefix: str = "chk_",
    peer_prefix: str | None = None,
    show_config: bool = True,
) -> None:
    """
    Render a single metric card (checkbox + badges + optional inline config).

    :param info: The metric metadata.
    :param selected: Mutable list of currently selected metric names.
    :param df: The active dataframe (passed to inline config editors).
    :param key_prefix: Prefix for the checkbox session_state key. The Overview
        tab uses ``chk_ov_`` and the dimension tabs use ``chk_`` so both can
        coexist in the same render pass without colliding.
    :param peer_prefix: When set, the checkbox writes its new value to
        ``{peer_prefix}{info.name}`` via an ``on_change`` callback so the peer
        widget reflects the toggle on the next render.
    :param show_config: When False, skip the inline config editor and the
        row-cap input. The Overview tab uses ``False`` to avoid creating
        duplicate config-widget keys; configuration stays in the dimension tab.
    :return: None.
    """
    border_color = _card_border_color(info)
    with st.container(border=True):
        st.markdown(
            f'<div style="border-left:4px solid {border_color};'
            f'margin:-8px -12px 8px -12px;padding:0;border-radius:2px 0 0 2px"></div>',
            unsafe_allow_html=True,
        )
        col_check, col_badges = st.columns([3, 2])

        primary_key = f"{key_prefix}{info.name}"
        cb_kwargs: dict = {}
        if peer_prefix is not None:
            cb_kwargs["on_change"] = _sync_widget_state
            cb_kwargs["args"] = (primary_key, f"{peer_prefix}{info.name}")

        with col_check:
            label = _format_name(info.name)
            new_val = st.checkbox(
                label,
                value=info.name in selected,
                key=primary_key,
                disabled=info.unavailable_reason is not None,
                **cb_kwargs,
            )
            if not info.unavailable_reason:
                if new_val and info.name not in selected:
                    selected.append(info.name)
                elif not new_val and info.name in selected:
                    selected.remove(info.name)
            if info.description:
                st.caption(info.description)
            if info.unavailable_reason:
                st.caption(f":material/warning: {info.unavailable_reason}")

        with col_badges:
            _render_badges(info)

        if show_config and new_val and (info.config_class or info.fd_json_config or info.callable_config):
            st.divider()
            _render_inline_config(info, df)

        if show_config and new_val and info.cell_granularity:
            _render_max_rows(info.name)


def _render_max_rows(metric_name: str) -> None:
    """
    Render an inline row-cap input for cell-granularity metrics.

    :param metric_name: Metric the cap applies to.
    :return: None.
    """
    df = AppState.get_df()
    n_rows = len(df) if df is not None else 0

    if n_rows > 0:
        step = 10 ** max(0, math.floor(math.log10(max(10, n_rows // 10))))
        max_val = n_rows
    else:
        step = ROW_CAP_FALLBACK_STEP
        max_val = ROW_CAP_FALLBACK_MAX

    st.caption("Cell-level metric - produces one result per cell.")
    val = st.number_input(
        "Row limit (0 = no cap)",
        min_value=0,
        max_value=max_val,
        value=AppState.get_metric_max_rows(metric_name) or 0,
        step=step,
        key=f"max_rows_{metric_name}",
        help=f"Caps the number of rows processed. 0 means all {n_rows:,} rows.",
    )
    AppState.set_metric_max_rows(metric_name, int(val))


def _render_badges(info: MetricInfo) -> None:
    """Render the right-hand badge strip describing a metric's config type and dependencies."""
    with st.container(horizontal=True):
        if info.callable_config:
            st.badge("Python rules", color="orange", icon=":material/code:")
        if info.fd_json_config:
            st.badge("FD rules", color="violet", icon=":material/account_tree:")
        if info.requires_reference:
            st.badge("Needs reference", color="blue", icon=":material/compare_arrows:")
        if info.config_required and not info.callable_config and not info.fd_json_config:
            st.badge("Config required", color="violet", icon=":material/settings:")


def _render_inline_config(info: MetricInfo, df) -> None:
    """
    Dispatch to the appropriate inline config editor for a metric.

    :param info: The metric metadata.
    :param df: The active dataframe.
    :return: None.
    """
    if info.callable_config:
        cfg = callable_editor.render(info.name, info.config_class, key_prefix=info.name)
        if cfg is not None:
            AppState.set_metric_config(info.name, cfg)
        return

    if info.fd_json_config:
        _render_fd_config(info.name, df)
        return

    if info.name == "timeliness_heinrich":
        cfg = timeliness_editor.render(
            info.config_class,
            key_prefix=info.name,
            df_columns=list(df.columns),
            df=df,
        )
        if cfg is not None:
            AppState.set_metric_config(info.name, cfg)
        return

    if info.config_class:
        # If cell-level output isn't recommended for this metric, pre-select
        # column-axis aggregation so the user doesn't have to know that the
        # default produces a degenerate per-cell view.
        field_defaults: dict[str, object] = {}
        if (
            info.recommended_granularities
            and DQGranularity.CELL not in info.recommended_granularities
            and any(f.name == "aggregation_axis" for f in info.config_fields)
        ):
            field_defaults["aggregation_axis"] = "index"
        cfg = simple_editor.render(
            info.config_class,
            key_prefix=info.name,
            field_defaults=field_defaults or None,
        )
        if cfg is not None:
            AppState.set_metric_config(info.name, cfg)


def _render_fd_config(metric_name: str, df) -> None:
    """
    Render the inline FD-rule config editor (file import + manual rule builder).

    :param metric_name: Metric the rules apply to.
    :param df: The active dataframe (for the column dropdowns).
    :return: None.
    """
    cols = list(df.columns)

    with st.expander("Import from file (HyFD / AIDFD format)", expanded=False):
        st.caption(
            "Upload a file produced by HyFD, AIDFD, or a similar tool. "
            "Each line should follow the format `[table.col1]->table.col2`. "
            "Multi-column LHS dependencies are skipped — this metric only supports "
            "single-column determinants."
        )
        uploaded = st.file_uploader(
            "FD file",
            type=["txt", "csv"],
            key=f"fd_upload_{metric_name}",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            if st.button("Import rules from file", key=f"fd_import_{metric_name}"):
                msg = _import_fd_file(uploaded, metric_name, cols)
                st.session_state[f"_fd_import_msg_{metric_name}"] = msg
                st.rerun()
        msg_key = f"_fd_import_msg_{metric_name}"
        if msg_key in st.session_state:
            st.success(st.session_state.pop(msg_key))

    st.write("**Add rule manually**")
    c1, c2 = st.columns(2)
    with c1:
        det = st.selectbox("Determinant column", cols, key=f"fd_det_{metric_name}")
    with c2:
        deps = st.multiselect(
            "Dependent column(s)",
            [c for c in cols if c != det],
            key=f"fd_deps_{metric_name}",
        )

    if st.button("Add rule", key=f"fd_add_{metric_name}"):
        if det and deps:
            existing: dict = AppState.get_metric_configs().get(metric_name) or {}
            existing[det] = list(set(existing.get(det, []) + deps))
            AppState.set_metric_config(metric_name, existing)

    current: dict = AppState.get_metric_configs().get(metric_name) or {}
    if current:
        st.write("**Current rules:**")
        for d, dependents in list(current.items()):
            col_rule, col_del = st.columns([4, 1])
            with col_rule:
                st.write(f"`{d}` → {', '.join(f'`{dep}`' for dep in dependents)}")
            with col_del:
                if st.button("✕", key=f"fd_del_{metric_name}_{d}"):
                    del current[d]
                    AppState.set_metric_config(metric_name, current)
                    st.rerun()
    else:
        st.caption("No rules defined yet.")


def _import_fd_file(uploaded_file, metric_name: str, available_cols: list[str]) -> str:
    """
    Parse an HyFD / AIDFD file and merge rules into the metric config.

    :param uploaded_file: A Streamlit ``UploadedFile``.
    :param metric_name: Metric the rules apply to.
    :param available_cols: Columns available in the loaded dataframe.
    :return: A status message summarizing the import (counts of imported / skipped rules).
    """
    table_name = AppState.get_table_name() or ""
    content = uploaded_file.read().decode("utf-8", errors="replace")

    added = 0
    skipped_multi = 0
    skipped_unknown = 0

    existing: dict = AppState.get_metric_configs().get(metric_name) or {}

    for line in content.splitlines():
        match = _HYFD_RE.search(line)
        if not match:
            continue
        lhs_raw, rhs_raw = match.groups()
        lhs_cols = [_strip_table_prefix(c.strip(), table_name) for c in lhs_raw.split(",")]
        rhs_col = _strip_table_prefix(rhs_raw.strip(), table_name)

        if len(lhs_cols) != 1:
            skipped_multi += 1
            continue

        det = lhs_cols[0]
        if det not in available_cols or rhs_col not in available_cols:
            skipped_unknown += 1
            continue

        existing[det] = list(set(existing.get(det, []) + [rhs_col]))
        added += 1

    AppState.set_metric_config(metric_name, existing)

    parts = [f"{added} rule{'s' if added != 1 else ''} imported"]
    if skipped_multi:
        parts.append(f"{skipped_multi} multi-column LHS skipped")
    if skipped_unknown:
        parts.append(f"{skipped_unknown} unknown column(s) skipped")
    return " · ".join(parts)


def _strip_table_prefix(col: str, table_name: str) -> str:
    """
    Strip ``table.csv.`` or ``table.`` prefix from a column name.

    Falls back to stripping any leading ``word.`` segment when ``table_name`` is
    unknown, so that ``adult.education`` resolves to ``education`` even without
    knowing the table name up front.

    :param col: The column reference to clean.
    :param table_name: The known table name (or empty string when unknown).
    :return: The cleaned column name.
    """
    if table_name:
        table_base = table_name.rsplit(".", 1)[0] if "." in table_name else table_name
        for prefix in (f"{table_name}.csv.", f"{table_name}.", f"{table_base}.csv.", f"{table_base}."):
            if col.startswith(prefix):
                return col[len(prefix):]
    dot = col.find(".")
    if dot != -1:
        return col[dot + 1:]
    return col


def _format_name(name: str) -> str:
    """
    Format a registry name like ``completeness_nullRatio`` for display.

    :param name: The metric registry name (``dimension_metric``).
    :return: A display string like ``"Completeness: Null ratio"``.
    """
    parts = name.split("_", 1)
    if len(parts) == 2:
        dim, metric = parts
        metric_spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", metric)
        metric_cased = metric_spaced[:1].upper() + metric_spaced[1:]
        return f"{dim.capitalize()}: {metric_cased}"
    return name
