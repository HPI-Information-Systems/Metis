"""Resolve a metric config's reference field into usable reference data.

Metric configs are frequently loaded from JSON, which cannot carry a
DataFrame. A reference field therefore stores a *source*, and this module
turns that source into the shape the metric actually wants. File paths route
through the project's :class:`~metis.loader.csv_loader.CSVLoader` so a
reference honours the same parsing options as the primary dataset.

A dataset can also carry a *default* reference, declared once for the whole
dataset rather than per metric. While :func:`default_reference` is active, a
reference field left at ``None`` resolves to that default, and a metric's own
``reference`` still wins when it is set.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Union

import pandas as pd

from metis.loader.csv_loader import CSVLoader
from metis.utils.data_config import DataConfig

ReferenceSource = Union[pd.DataFrame, str, Mapping, list, set, None]

_VOCABULARY_COLUMN = "value"


class MissingReferenceError(ValueError):
    """Raised when a metric that requires a reference has none to use."""


_default_reference: ContextVar[pd.DataFrame | None] = ContextVar(
    "metis_default_reference", default=None
)


@contextmanager
def default_reference(frame: pd.DataFrame | None) -> Iterator[None]:
    """
    Make ``frame`` the reference for configs that do not set their own.

    The orchestrator and the GUI wrap a metric run in this so that a
    dataset-level reference reaches the metrics that require one. Passing
    ``None`` leaves no default in place.

    :param frame: The dataset-level reference, or ``None``.
    :return: A context manager. The default is cleared again on exit.
    """
    token = _default_reference.set(frame)
    try:
        yield
    finally:
        _default_reference.reset(token)


def get_default_reference() -> pd.DataFrame | None:
    """
    Return the dataset-level default reference, if one is active.

    :return: The active default reference, or ``None``.
    """
    return _default_reference.get()


def has_reference(source: ReferenceSource) -> bool:
    """
    Return whether a reference field resolves to any reference data.

    Config classes use this in ``validate`` so that a field left at ``None``
    is accepted while a dataset-level default is active.

    :param source: The configured reference source.
    :return: ``True`` if the source is set or a default reference is active.
    """
    return source is not None or get_default_reference() is not None


def load_reference_frame(source: ReferenceSource) -> pd.DataFrame | None:
    """
    Resolve a reference source into a DataFrame.

    Accepted sources:

    - ``None``: the dataset-level default if :func:`default_reference` is
      active, otherwise ``None``.
    - ``pd.DataFrame``: returned unchanged. Used by the GUI and notebooks.
    - ``str``: a CSV path, loaded with default parsing options.
    - ``Mapping``: a :class:`DataConfig` payload, so callers can set
      ``delimiter``, ``encoding``, ``usecols`` and the rest. Must contain
      ``file_name``.
    - ``list`` or ``set``: inline values, wrapped into a single column named
      ``value``.

    :param source: The configured reference source.
    :raises TypeError: If the source is of an unsupported type.
    :raises ValueError: If a mapping source omits ``file_name``.
    :return: The reference DataFrame, or ``None``.
    """
    if source is None:
        return get_default_reference()

    if isinstance(source, pd.DataFrame):
        return source

    if isinstance(source, str):
        return _load_csv({"file_name": source})

    if isinstance(source, Mapping):
        if "file_name" not in source:
            raise ValueError(
                "A mapping reference source must include 'file_name'. "
                f"Got keys: {sorted(source)}."
            )
        return _load_csv(source)

    if isinstance(source, (list, set, tuple)):
        return pd.DataFrame({_VOCABULARY_COLUMN: list(source)})

    raise TypeError(
        f"Unsupported reference source type {type(source).__name__}. Provide a "
        f"DataFrame, a CSV path, a DataConfig mapping, or a list/set of values."
    )


def load_reference_vocabulary(source: ReferenceSource) -> set[str] | None:
    """
    Resolve a reference source into a lower-cased vocabulary.

    Lists and sets are used directly. Every other source is resolved through
    :func:`load_reference_frame` and must yield exactly one column. A
    dataset-level default reference is a table, not a vocabulary, so it is
    never used here.

    :param source: The configured reference source.
    :raises ValueError: If the resolved frame does not have exactly one column.
    :return: The vocabulary as a set of normalised strings, or ``None``.
    """
    if source is None:
        return None

    if isinstance(source, (list, set, tuple)):
        # Drop missing values so an inline list behaves like the frame path,
        # which calls dropna(). A config loaded from JSON can legitimately
        # contain null, and stringifying that would put "none" in the
        # vocabulary.
        return {
            str(v).strip().lower()
            for v in source
            if v is not None and not (isinstance(v, float) and pd.isna(v))
        }

    frame = load_reference_frame(source)
    if frame is None:
        return None
    if frame.shape[1] != 1:
        raise ValueError(
            f"A vocabulary reference must have a single column, got "
            f"{frame.shape[1]}: {list(frame.columns)}."
        )
    return {
        str(v).strip().lower()
        for v in frame.iloc[:, 0].dropna().unique()
    }


def describe_reference_source(source: ReferenceSource) -> Any:
    """
    Return a JSON-serialisable echo of a reference source.

    Metric configs record themselves into ``DQResult.configJson``, which must
    be serialisable. A live DataFrame is summarised by shape rather than
    embedded. A source left at ``None`` echoes the dataset-level default when
    one is active, so the result records which reference was actually used.

    :param source: The configured reference source.
    :return: A JSON-safe representation of the source.
    """
    if source is None:
        default = get_default_reference()
        if default is None:
            return None
        return {"dataset_default": _describe_frame(default)}
    if isinstance(source, str):
        return source
    if isinstance(source, pd.DataFrame):
        return _describe_frame(source)
    if isinstance(source, Mapping):
        return dict(source)
    if isinstance(source, (list, set, tuple)):
        return list(source)
    return str(source)


def _describe_frame(frame: pd.DataFrame) -> dict:
    """
    Summarise a DataFrame by shape so it can be stored as JSON.

    :param frame: The DataFrame to summarise.
    :return: A JSON-safe summary of the frame.
    """
    return {
        "dataframe": {
            "rows": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
        }
    }


def _load_csv(payload: Mapping) -> pd.DataFrame:
    """
    Load a CSV through the project's CSVLoader.

    :param payload: A ``DataConfig`` payload containing at least ``file_name``.
    :return: The loaded DataFrame.
    """
    config_dict = {"name": "reference", **dict(payload)}
    return CSVLoader().load(DataConfig(config_dict))
