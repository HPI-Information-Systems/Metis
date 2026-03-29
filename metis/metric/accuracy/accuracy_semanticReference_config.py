from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from metis.metric.config import MetricConfig


@dataclass
class accuracy_semanticReference_config(MetricConfig):
    """Configuration for ``accuracy_semanticReference`` (ISO/IEC 25024 Acc-I-2).

    :param key_column: If set, ``data`` and ``reference`` are joined on this
        column before per-cell comparison. If ``None`` (default), rows are
        compared positionally and ``data`` and ``reference`` must have the
        same length.
    """

    reference_file_path: str | Path
    key_column: str | None = None

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "key_column": self.key_column,
        }

    def validate(self):
        if self.key_column is not None and not isinstance(self.key_column, str):
            raise ValueError(
                f"key_column must be a string or None, got {type(self.key_column)}."
            )
