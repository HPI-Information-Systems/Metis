"""Generic config editor: render a form driven by dataclass field annotations."""
from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import Field
from typing import Any, get_args, get_origin

import streamlit as st

from metis.metric.config import MetricConfig


def render(
    config_class: type,
    key_prefix: str,
    field_defaults: dict[str, Any] | None = None,
) -> MetricConfig | None:
    """
    Render a form for each field in a ``MetricConfig`` dataclass.

    :param config_class: The dataclass type to render.
    :param key_prefix: Streamlit widget key prefix used to namespace this form.
    :param field_defaults: Optional per-field default values used to seed widgets
        on first render only — Streamlit session state takes over once the
        widget exists. Use this to pre-pick a more sensible value than the
        dataclass default for GUI users.
    :return: A populated ``config_class`` instance, or ``None`` if validation failed.
    """
    try:
        hints = typing.get_type_hints(config_class)
    except Exception:
        hints = {}

    overrides = field_defaults or {}
    values: dict[str, Any] = {}
    for f in dataclasses.fields(config_class):
        annotation = hints.get(f.name, f.type)
        if f.name in overrides:
            default = overrides[f.name]
        else:
            default = f.default if not isinstance(f.default, dataclasses._MISSING_TYPE) else None
        widget_key = f"{key_prefix}__{f.name}"
        values[f.name] = _render_field(f, annotation, default, widget_key)

    try:
        cfg = config_class(**values)
        cfg.validate()
        return cfg
    except (TypeError, ValueError) as e:
        st.error(f"Config error: {e}")
        return None


def _render_field(field: Field, annotation: Any, default: Any, key: str) -> Any:
    """
    Render the widget for a single dataclass field.

    :param field: The dataclass field (used for the visible label).
    :param annotation: The (possibly Optional-wrapped) type annotation.
    :param default: Default value to seed the widget with.
    :param key: Streamlit widget key.
    :return: The widget's current value.
    """
    annotation = _unwrap_optional(annotation)
    origin = get_origin(annotation)
    args = get_args(annotation)
    name = field.name

    if origin is typing.Literal:
        options = list(args)
        idx = options.index(default) if default in options else 0
        return st.selectbox(
            name,
            options=options,
            index=idx,
            key=key,
            format_func=lambda x: str(x) if x is not None else "None",
        )

    if annotation is bool:
        return st.checkbox(
            name,
            value=bool(default) if default is not None else False,
            key=key,
        )

    if annotation is float:
        return st.number_input(
            name,
            value=float(default) if default is not None else 0.0,
            key=key,
        )

    if annotation is int:
        return st.number_input(
            name,
            value=int(default) if default is not None else 0,
            step=1,
            key=key,
        )

    if annotation is str:
        val = st.text_input(
            name,
            value=str(default) if default is not None else "",
            key=key,
        )
        return val if val else None

    val = st.text_input(
        name,
        value=str(default) if default is not None else "",
        key=key,
        help=f"Type: {annotation}",
    )
    return val if val else default


def _unwrap_optional(annotation: Any) -> Any:
    """
    Strip ``Optional[X]`` / ``Union[X, None]`` / ``X | None`` to return ``X``.

    :param annotation: The type annotation.
    :return: The annotation with the ``None`` arm removed when applicable.
    """
    origin = get_origin(annotation)

    if origin is typing.Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]

    if hasattr(types, "UnionType") and isinstance(annotation, types.UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]

    return annotation
