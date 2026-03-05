"""Importer for quartiles profiling task."""

from typing import Any, Dict, List

from .base import BaseImporter


class QuartilesImporter(BaseImporter):
    """Importer for the quartiles task."""

    @property
    def task_name(self) -> str:
        return "quartiles"

    def parse_file(self, file_path: str, table_name: str) -> List[Dict[str, Any]]:
        """Parse CSV with columns: column, Q1, Q2, Q3"""
        rows = self.read_csv(file_path)
        return [
            {
                "column_names": [row["column"]],
                "value": {
                    "Q1": float(row["Q1"]),
                    "Q2": float(row["Q2"]),
                    "Q3": float(row["Q3"]),
                },
            }
            for row in rows
        ]

    def parse_inline(
        self, values: List[Dict[str, Any]], table_name: str
    ) -> List[Dict[str, Any]]:
        return [
            {
                "column_names": [v["column"]],
                "value": {"Q1": v["Q1"], "Q2": v["Q2"], "Q3": v["Q3"]},
            }
            for v in values
        ]
