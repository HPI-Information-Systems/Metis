"""Config editor for rule-based consistency metrics that require callable rules.

Renders a Python code editor with syntax highlighting (streamlit-ace), then
``exec()``s the code to extract ``attribute_rules`` and ``tuple_rules`` and
build the config object. Safe for local use, but is not browser-sandboxed.
"""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

try:
    from streamlit_ace import st_ace
    _ACE_AVAILABLE = True
except ImportError:
    _ACE_AVAILABLE = False


@dataclass(frozen=True)
class CallableTemplate:
    """Editor seed for a callable-rule metric: starter code plus return-type hint."""
    code: str
    return_type: str


_HINRICHS_TEMPLATE = CallableTemplate(
    return_type="**float** (degree of violation — 0 = perfect)",
    code='''\
# Hinrichs rule-based consistency
# Each rule receives a cell value and returns a degree of violation (float).
#   0.0 = perfectly consistent,  higher = more violation
# Row score = 1 / (1 + sum_of_violations)

import pandas as pd

# Per-column rules: map each column name to a list of rule functions.
attribute_rules = {
    # Example — age must be between 0 and 120:
    # "age": [
    #     lambda x: 0.0 if isinstance(x, (int, float)) and 0 <= x <= 120 else 1.0,
    # ],
}

# Per-row (tuple) rules: list of functions that receive a pd.Series.
tuple_rules = [
    # Example — start_date must not exceed end_date:
    # lambda row: 0.0 if row["start_date"] <= row["end_date"] else 1.0,
]
''',
)

_PIPINO_TEMPLATE = CallableTemplate(
    return_type="**bool** (True = consistent)",
    code='''\
# Pipino rule-based consistency
# Each rule receives a cell value and returns True (consistent) or False (violation).
# Row score = fraction of rules satisfied.

import pandas as pd

# Per-column rules: map each column name to a list of rule functions.
attribute_rules = {
    # Example — age must be between 0 and 120:
    # "age": [
    #     lambda x: isinstance(x, (int, float)) and 0 <= x <= 120,
    # ],
}

# Per-row (tuple) rules: list of functions that receive a pd.Series.
tuple_rules = [
    # Example — start_date must not exceed end_date:
    # lambda row: row["start_date"] <= row["end_date"],
]
''',
)

_GENERIC_TEMPLATE = CallableTemplate(
    return_type="the value the metric expects (see its documentation)",
    code='''\
# Rule-based consistency
# Define attribute_rules (per-column) and/or tuple_rules (per-row).
# Consult the metric's documentation for the expected return type of each rule.

import pandas as pd

# Per-column rules: map each column name to a list of rule functions.
attribute_rules = {
    # "column_name": [lambda x: ...],
}

# Per-row (tuple) rules: list of functions that receive a pd.Series.
tuple_rules = [
    # lambda row: ...,
]
''',
)

# Explicit, GUI-side mapping from metric name to its editor template. New
# rule-based metrics fall back to ``_GENERIC_TEMPLATE`` until a tailored entry
# is added here.
_TEMPLATES: dict[str, CallableTemplate] = {
    "consistency_ruleBasedHinrichs": _HINRICHS_TEMPLATE,
    "consistency_ruleBasedPipino": _PIPINO_TEMPLATE,
}


def _template_for(metric_name: str) -> CallableTemplate:
    """
    Return the editor template registered for a metric, or the generic fallback.

    :param metric_name: Registry name of the metric being configured.
    :return: A :class:`CallableTemplate` with starter code and return-type hint.
    """
    return _TEMPLATES.get(metric_name, _GENERIC_TEMPLATE)


def render(metric_name: str, config_class, key_prefix: str):
    """
    Render the callable rule editor and return a config when rules are applied.

    :param metric_name: Metric this editor is configuring (drives the template choice).
    :param config_class: The dataclass to build when ``Apply`` is clicked.
    :param key_prefix: Streamlit widget key prefix.
    :return: A ``config_class`` instance once rules are applied, or ``None``.
    """
    template = _template_for(metric_name)

    st.caption(
        f"Rules return {template.return_type}. "
        "Define `attribute_rules` (per-column) and/or `tuple_rules` (per-row), then click **Apply**."
    )
    st.warning(
        "The code below is executed with full access to your machine when "
        "you click Apply. Only run rules that you wrote or trust. This "
        "editor is intended for local use, not multi user hosting.",
        icon="⚠️",
    )

    code_key = f"_callable_code_{key_prefix}"
    if code_key not in st.session_state:
        st.session_state[code_key] = template.code

    code = _code_editor(
        value=st.session_state[code_key],
        key=f"_callable_ta_{key_prefix}",
    )
    st.session_state[code_key] = code

    err_key = f"_callable_err_{key_prefix}"
    cfg_key = f"_callable_cfg_{key_prefix}"

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        if st.button("Apply rules", key=f"_callable_apply_{key_prefix}", type="primary"):
            result = _exec_and_build(code, config_class)
            if isinstance(result, Exception):
                st.session_state[err_key] = str(result)
                st.session_state.pop(cfg_key, None)
            else:
                st.session_state[cfg_key] = result
                st.session_state.pop(err_key, None)

    with col_status:
        if err_key in st.session_state:
            st.error(f"Error: {st.session_state[err_key]}")
        elif cfg_key in st.session_state:
            cfg = st.session_state[cfg_key]
            n_attr = len(cfg.attribute_rules or {})
            n_tuple = len(cfg.tuple_rules or [])
            st.success(
                f"Rules applied — {n_attr} column rule set(s), {n_tuple} tuple rule(s)."
            )

    return st.session_state.get(cfg_key)


def _code_editor(value: str, key: str, height: int = 300) -> str:
    """
    Render a Python code editor with syntax highlighting when available.

    :param value: Initial code text.
    :param key: Streamlit widget key.
    :param height: Editor height in pixels.
    :return: The current editor contents.
    """
    if _ACE_AVAILABLE:
        result = st_ace(
            value=value,
            language="python",
            theme="tomorrow",
            key=key,
            height=height,
            font_size=13,
            tab_size=4,
            wrap=False,
            auto_update=True,
            show_gutter=True,
            show_print_margin=False,
        )
        return result if result is not None else value
    return st.text_area(
        "Rule definitions (Python)",
        value=value,
        height=height,
        key=key,
        help="Standard Python: import statements, lambdas, and named functions are all supported.",
    )


def _exec_and_build(code: str, config_class):
    """
    Exec user Python code and construct the rule config object.

    :param code: The Python source from the editor.
    :param config_class: The dataclass to build.
    :return: A ``config_class`` instance, or an ``Exception`` on failure.
    """
    namespace: dict = {}
    try:
        exec(compile(code, "<callable_editor>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        return exc

    attribute_rules = namespace.get("attribute_rules") or None
    tuple_rules = namespace.get("tuple_rules") or None

    if not attribute_rules and not tuple_rules:
        return ValueError(
            "No rules defined. Add at least one entry to attribute_rules or tuple_rules."
        )

    try:
        return config_class(attribute_rules=attribute_rules, tuple_rules=tuple_rules)
    except Exception as exc:
        return exc
