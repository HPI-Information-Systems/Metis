"""Transparent caching decorator for data-profiling functions.

Usage::

    from metis.profiling.cache import cached
    from metis.utils.data_profiling.single_column.cardinalities.null_values import (
        null_count as _null_count,
    )

    null_count = cached(_null_count)

The wrapped function has the *exact same* signature as the original.
When ``DataProfileManager`` is initialised and a context (dataset / table)
is active, results are automatically looked up in – and written to – the
profile database.  When the manager is *not* initialised the function simply
delegates to the original implementation (zero overhead apart from the
isinstance check).
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import pandas as pd


def cached(fn: Callable) -> Callable:
    """Return a wrapper around *fn* that caches via ``DataProfileManager``."""

    @functools.wraps(fn)
    def wrapper(data: pd.Series | pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        from metis.profiling.data_profile_manager import DataProfileManager

        # If the manager is not active, fall through to the original function.
        if not DataProfileManager.is_initialized():
            return fn(data, *args, **kwargs)

        manager = DataProfileManager.get_instance()
        if manager.dataset is None or manager.table is None:
            return fn(data, *args, **kwargs)

        # Derive column names from the data argument.
        if isinstance(data, pd.Series):
            if data.name is None:
                return fn(data, *args, **kwargs)
            column_names = [str(data.name)]
        else:
            column_names = [str(c) for c in data.columns]

        # Include column names from any extra Series args (e.g. jaccard's column2)
        # so that the cache key distinguishes different column pairs.
        for arg in args:
            if isinstance(arg, pd.Series) and arg.name is not None:
                column_names.append(str(arg.name))

        # Build optional config dict from extra arguments (if any).
        task_config = _build_config(fn, args, kwargs) or None

        # Cache lookup
        cached_value = manager.lookup(column_names, fn.__name__, task_config)
        if cached_value is not None:
            return cached_value

        # Compute
        result = fn(data, *args, **kwargs)

        # Store
        manager.store(
            column_names=column_names,
            dp_task_name=fn.__name__,
            value=result,
            task_config=task_config,
        )

        return result

    return wrapper


def _build_config(fn: Callable, args: tuple, kwargs: dict) -> dict | None:
    """Turn extra positional/keyword args into a JSON-safe config dict."""
    if not args and not kwargs:
        return None

    import inspect

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())[1:]  # skip 'data'

    config: dict = {}
    for i, val in enumerate(args):
        if isinstance(val, pd.Series):
            continue
        if i < len(params):
            config[params[i]] = val
    config.update(kwargs)

    if not config:
        return None
    return config
