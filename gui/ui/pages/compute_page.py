"""Step 3: Run selected metrics and show progress."""
from __future__ import annotations

import streamlit as st

from core.metric_catalog import get_catalog
from core.metric_runner import RunError, run_all
from core.result_store import RunMetadata, SQLiteResultStore
from ui.state import AppState

LARGE_CELL_RUN_RESULT_THRESHOLD: int = 500_000


def render(store, demo_mode: bool = False) -> None:
    """
    Render the Compute step: validate, run metrics, and persist results.

    :param store: The active result store.
    :param demo_mode: When True, render the demo-mode short-circuit instead of the real run UI.
    :return: None.
    """
    if demo_mode:
        _render_demo_compute()
        return

    st.header("Compute")

    df = AppState.get_df()
    selected = AppState.get_selected_metrics()

    errors = _validate(selected)
    if errors:
        AppState.set_run_requested(False)
        st.error("Cannot run — fix the following issues first:")
        for msg in errors:
            st.warning(msg)
        if st.button("← Back to Metrics", key="back_to_metrics"):
            AppState.set_step(1)
            st.rerun()
        return

    tag = AppState.get_experiment_tag() or AppState.get_dataset_name() or "run"

    if AppState.get_run_requested():
        AppState.set_run_requested(False)
        _run(selected, df, store, tag)
    elif AppState.get_last_experiment_tag():
        last = AppState.get_last_experiment_tag()
        errs = AppState.get_last_errors()
        st.success(f"Last run: `{last}` completed.")
        if errs:
            st.warning(f"{len(errs)} metric(s) had errors in the last run.")
        if st.button("Run again", key="run_again"):
            _run(selected, df, store, tag)
    else:
        st.info(
            "Use **← Back** to return to Metrics, then click **Compute →** to start a run."
        )


def _validate(selected: list[str]) -> list[str]:
    """
    Check selected metrics against current state and return blocking error messages.

    :param selected: Selected metric names.
    :return: A list of error messages (empty when ready to run).
    """
    catalog = get_catalog()
    errors: list[str] = []

    for name in selected:
        info = catalog.get(name)
        if info is None:
            continue
        if info.requires_reference and AppState.get_reference_df() is None:
            errors.append(
                f"**{name}** requires a reference dataset. "
                "Upload one in the Dataset step."
            )
        if info.config_required:
            cfg = AppState.get_metric_configs().get(name)
            if not cfg:
                errors.append(
                    f"**{name}** requires configuration. "
                    "Set it in the Metrics step."
                )
    return errors


def _warn_large_cell_runs(selected: list[str], df) -> None:
    """
    Warn the user when a cell-granularity metric will produce a very large result set.

    :param selected: Selected metric names.
    :param df: The active dataframe.
    :return: None.
    """
    catalog = get_catalog()
    if df is None:
        return
    n_rows = len(df)
    n_cols = len(df.columns)
    for name in selected:
        info = catalog.get(name)
        if not (info and info.cell_granularity):
            continue
        cap = AppState.get_metric_max_rows(name) or 0
        effective_rows = min(n_rows, cap) if cap else n_rows
        estimated = effective_rows * n_cols
        if estimated > LARGE_CELL_RUN_RESULT_THRESHOLD:
            st.warning(
                f"**{name}** is a cell-level metric and will produce "
                f"~{estimated:,} results on this dataset. "
                f"Saving may take a while. Consider setting a row limit "
                f"or using column-level aggregation in the metric config."
            )


def _run(
    selected: list[str],
    df,
    store: SQLiteResultStore,
    tag: str,
) -> None:
    """
    Run all selected metrics with a progress bar and persist their results.

    :param selected: Selected metric names.
    :param df: The active dataframe.
    :param store: The active result store.
    :param tag: Experiment tag used to label the run.
    :return: None.
    """
    reference = AppState.get_reference_df()
    configs = AppState.get_metric_configs()
    max_rows = AppState.get_all_metric_max_rows()

    st.write(
        f"Running **{len(selected)}** metric(s) on "
        f"**{AppState.get_dataset_name()}**  ·  tag `{tag}`"
    )

    _warn_large_cell_runs(selected, df)

    progress_bar = st.progress(0.0)
    status_text = st.empty()
    per_metric_status = st.container()

    def on_progress(i: int, total: int, name: str) -> None:
        progress_bar.progress(i / total)
        status_text.text(f"Running {name}  ({i + 1}/{total})")

    results, errors = run_all(
        selected, df, reference, configs, on_progress,
        max_rows_by_metric=max_rows or None,
    )

    progress_bar.progress(1.0)
    status_text.empty()

    error_map = {e.metric: e for e in errors}
    with per_metric_status:
        for name in selected:
            if name in error_map:
                err: RunError = error_map[name]
                with st.expander(f"✗  {name} — {err.error}", expanded=False):
                    st.code(err.tb, language="python")
            else:
                batch_count = sum(1 for r in results if r.DQmetric == name)
                st.success(f"✓  {name} — {batch_count:,} result(s)")

    if results:
        metadata = RunMetadata(
            experiment_tag=tag,
            dataset_name=AppState.get_dataset_name(),
            table_name=AppState.get_table_name() or None,
        )
        n = len(results)
        status_text.text(f"Saving {n:,} results to the database…")
        store.save_run(results, metadata)
        status_text.empty()

        AppState.clear_results_caches()

        AppState.set_last_errors(errors)
        AppState.set_last_experiment_tag(tag)

        st.success(f"Done — {n:,} results saved.")
    else:
        st.error("All metrics failed. No results were saved.")


def _render_demo_compute() -> None:
    """
    Short-circuit the Compute step in demo mode and auto-advance to Results.

    Demo results are seeded at startup, so there is nothing to compute here —
    the experiment tag is set and the wizard step bumped to 3.

    :return: None.
    """
    st.header("Compute")

    demo_selected = AppState.get_demo_selected_metrics()
    st.write(
        f"Loading pre-computed results for "
        f"**{len(demo_selected)}** metric(s) on the restaurant dataset…"
    )

    AppState.set_last_experiment_tag("demo")

    try:
        AppState.clear_results_caches()
    except Exception:
        pass

    st.success("Pre-computed results loaded.")
    AppState.set_demo_step(3)
    st.rerun()
