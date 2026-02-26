from dataclasses import dataclass, field
from typing import Optional

from metis.metric.config import MetricConfig


@dataclass(kw_only=True)
class readability_base_config(MetricConfig):
    """
    Shared base configuration for readability metrics.

    :param sample_size: Number of rows to sample for evaluation. If None, all rows are used.
                        Default: None
    :param random_seed: Seed for reproducible sampling.
                        Default: 13
    :param min_token_length: Minimum character length for a token to be evaluated.
                             Shorter tokens are ignored.
                             Default: 2
    :param abbr_csv: Path to a CSV file containing known abbreviations. If None, no
                     abbreviation list is loaded.
                     Default: None
    :param ignore_numeric_columns: If True, only object/string columns are evaluated.
                                   Numeric columns are skipped.
                                   Default: True
    :param compute_schema: If True, readability of column names (schema labels) is
                           computed in addition to cell content.
                           Default: True
    """

    sample_size: Optional[int] = field(default=None)
    random_seed: int = field(default=13)
    min_token_length: int = field(default=2)
    abbr_csv: Optional[str] = field(default=None)
    ignore_numeric_columns: bool = field(default=True)
    compute_schema: bool = field(default=True)

    def __post_init__(self):
        if self.sample_size is not None:
            if not isinstance(self.sample_size, int) or self.sample_size < 1:
                raise ValueError(f"sample_size must be a positive integer or None, got {self.sample_size!r}")

        if not isinstance(self.random_seed, int):
            raise ValueError(f"random_seed must be an integer, got {type(self.random_seed)}")

        if not isinstance(self.min_token_length, int) or self.min_token_length < 1:
            raise ValueError(f"min_token_length must be a positive integer, got {self.min_token_length!r}")

        if self.abbr_csv is not None and not isinstance(self.abbr_csv, str):
            raise ValueError(f"abbr_csv must be a string path or None, got {type(self.abbr_csv)}")

        if not isinstance(self.ignore_numeric_columns, bool):
            raise ValueError(f"ignore_numeric_columns must be boolean, got {type(self.ignore_numeric_columns)}")

        if not isinstance(self.compute_schema, bool):
            raise ValueError(f"compute_schema must be boolean, got {type(self.compute_schema)}")

    def _base_json(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "sample_size": self.sample_size,
            "random_seed": self.random_seed,
            "min_token_length": self.min_token_length,
            "abbr_csv": self.abbr_csv,
            "ignore_numeric_columns": self.ignore_numeric_columns,
            "compute_schema": self.compute_schema,
        }

    def to_json(self):
        return self._base_json()