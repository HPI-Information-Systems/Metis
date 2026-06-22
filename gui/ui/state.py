from __future__ import annotations

import pandas as pd
import streamlit as st


class AppState:
    """Static wrapper around ``st.session_state`` for typed, named app state.

    Two kinds of state are exposed: persistent app data (dataset, selected
    metrics, last run) and ephemeral wizard navigation. Widget-local state
    (search boxes, expander toggles) does not belong here; pages namespace
    those keys themselves via :meth:`widget_key`.
    """

    @staticmethod
    def get_df() -> pd.DataFrame | None:
        return st.session_state.get("df")

    @staticmethod
    def set_df(df: pd.DataFrame) -> None:
        st.session_state["df"] = df

    @staticmethod
    def get_reference_df() -> pd.DataFrame | None:
        return st.session_state.get("reference_df")

    @staticmethod
    def set_reference_df(df: pd.DataFrame | None) -> None:
        st.session_state["reference_df"] = df

    @staticmethod
    def get_dataset_name() -> str:
        return st.session_state.get("dataset_name", "")

    @staticmethod
    def set_dataset_name(name: str) -> None:
        st.session_state["dataset_name"] = name

    @staticmethod
    def get_table_name() -> str:
        return st.session_state.get("table_name", "")

    @staticmethod
    def set_table_name(name: str) -> None:
        st.session_state["table_name"] = name

    @staticmethod
    def get_experiment_tag() -> str:
        return st.session_state.get("experiment_tag", "")

    @staticmethod
    def set_experiment_tag(tag: str) -> None:
        st.session_state["experiment_tag"] = tag

    @staticmethod
    def get_selected_metrics() -> list[str]:
        return st.session_state.get("selected_metrics", [])

    @staticmethod
    def set_selected_metrics(metrics: list[str]) -> None:
        st.session_state["selected_metrics"] = metrics

    @staticmethod
    def get_metric_configs() -> dict:
        return st.session_state.get("metric_configs", {})

    @staticmethod
    def set_metric_config(metric_name: str, config) -> None:
        configs = st.session_state.get("metric_configs", {})
        configs[metric_name] = config
        st.session_state["metric_configs"] = configs

    @staticmethod
    def get_last_results() -> list[dict]:
        return st.session_state.get("last_results", [])

    @staticmethod
    def set_last_results(results: list[dict]) -> None:
        st.session_state["last_results"] = results

    @staticmethod
    def get_last_errors() -> list:
        return st.session_state.get("last_errors", [])

    @staticmethod
    def set_last_errors(errors: list) -> None:
        st.session_state["last_errors"] = errors

    @staticmethod
    def get_last_experiment_tag() -> str | None:
        return st.session_state.get("last_experiment_tag")

    @staticmethod
    def set_last_experiment_tag(tag: str) -> None:
        st.session_state["last_experiment_tag"] = tag

    @staticmethod
    def get_step() -> int:
        return st.session_state.get("wizard_step", 0)

    @staticmethod
    def set_step(step: int) -> None:
        st.session_state["wizard_step"] = step

    @staticmethod
    def get_run_requested() -> bool:
        return st.session_state.get("run_requested", False)

    @staticmethod
    def set_run_requested(val: bool) -> None:
        st.session_state["run_requested"] = val

    @staticmethod
    def get_metric_max_rows(metric_name: str) -> int | None:
        return st.session_state.get("metric_max_rows", {}).get(metric_name)

    @staticmethod
    def set_metric_max_rows(metric_name: str, n: int) -> None:
        caps = st.session_state.get("metric_max_rows", {})
        caps[metric_name] = n
        st.session_state["metric_max_rows"] = caps

    @staticmethod
    def get_all_metric_max_rows() -> dict[str, int]:
        return st.session_state.get("metric_max_rows", {})

    @staticmethod
    def get_demo_mode_chosen() -> bool:
        return st.session_state.get("demo_mode_chosen", False)

    @staticmethod
    def set_demo_mode_chosen(val: bool) -> None:
        st.session_state["demo_mode_chosen"] = val

    @staticmethod
    def get_browser_path() -> str | None:
        """
        Return the browser-mode landing-page choice ('demo' or 'own').

        :return: 'demo', 'own', or None if the user has not chosen yet.
        """
        return st.session_state.get("browser_path")

    @staticmethod
    def set_browser_path(path: str) -> None:
        st.session_state["browser_path"] = path

    @staticmethod
    def get_demo_step() -> int:
        return st.session_state.get("demo_wizard_step", 0)

    @staticmethod
    def set_demo_step(step: int) -> None:
        st.session_state["demo_wizard_step"] = step

    @staticmethod
    def get_demo_selected_metrics() -> list[str]:
        return st.session_state.get("demo_selected_metrics", [])

    @staticmethod
    def set_demo_selected_metrics(metrics: list[str]) -> None:
        st.session_state["demo_selected_metrics"] = metrics

    @staticmethod
    def widget_key(page: str, name: str) -> str:
        """
        Build a namespaced ``st.session_state`` key for a widget on a page.

        :param page: Page identifier (e.g. "metrics", "results").
        :param name: Widget-local name (e.g. "search", "filter").
        :return: Namespaced key like ``_metrics_search``.
        """
        return f"_{page}_{name}"

    @staticmethod
    def clear_results_caches() -> None:
        """
        Invalidate all results-page DB caches and the dispatch caches.

        :return: None.
        """
        from ui.pages.results_page import (
            _cached_export_json,
            _cached_get_columns,
            _cached_get_heatmap,
            _cached_get_metrics,
            _cached_list_runs,
            _cached_load_temporal,
        )
        from visualization import dispatch

        _cached_list_runs.clear()
        _cached_get_metrics.clear()
        _cached_get_columns.clear()
        _cached_get_heatmap.clear()
        _cached_load_temporal.clear()
        _cached_export_json.clear()
        dispatch.invalidate_caches()

    @staticmethod
    def reset() -> None:
        """
        Clear all wizard state so the user can start fresh with a new dataset.

        :return: None.
        """
        for key in (
            "df", "reference_df", "dataset_name", "table_name",
            "experiment_tag", "selected_metrics", "metric_configs",
            "last_results", "last_errors", "run_requested",
        ):
            st.session_state.pop(key, None)
        st.session_state["wizard_step"] = 0
