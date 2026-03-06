"""Importer for simple column->value profiling tasks."""

from typing import Any, Dict, List

from .base import BaseImporter, auto_detect_type


class ScalarImporter(BaseImporter):
    """Importer for scalar profiling tasks (column, value pairs).

    Handles: null_count, null_percentage, distinct_count, row_count, uniqueness,
    value_length_min/max/mean/median, constancy, most_frequent_value,
    interquartile_range, basic_type, data_type, size, decimals, data_class, domain
    """

    def __init__(self, task_name: str):
        self._task_name = task_name

    @property
    def task_name(self) -> str:
        return self._task_name

    def parse_file(self, file_path: str, table_name: str) -> List[Dict[str, Any]]:
        rows = self.read_csv(file_path)
        return [
            {
                "column_names": [row["column"]],
                "value": auto_detect_type(row["value"]),
            }
            for row in rows
        ]

    def parse_inline(
        self, values: List[Dict[str, Any]], table_name: str
    ) -> List[Dict[str, Any]]:
        return [
            {
                "column_names": [v["column"]],
                "value": v["value"],
            }
            for v in values
        ]


# Pre-instantiated importers for scalar tasks
SCALAR_TASKS = [
    "null_count",
    "null_percentage",
    "distinct_count",
    "row_count",
    "uniqueness",
    "value_length_min",
    "value_length_max",
    "value_length_mean",
    "value_length_median",
    "constancy",
    "most_frequent_value",
    "interquartile_range",
    "basic_type",
    "data_type",
    "size",
    "decimals",
    "data_class",
    "domain",
]


def create_scalar_importers() -> Dict[str, ScalarImporter]:
    """Create ScalarImporter instances for all scalar tasks."""
    return {task: ScalarImporter(task) for task in SCALAR_TASKS}
