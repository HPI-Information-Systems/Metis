"""Importer for patterns profiling task."""

from collections import defaultdict
from typing import Any, Dict, List

from .base import BaseImporter


class PatternsImporter(BaseImporter):
    """Importer for the patterns task."""

    @property
    def task_name(self) -> str:
        return "patterns"

    def parse_file(self, file_path: str, table_name: str) -> List[Dict[str, Any]]:
        """Parse CSV with columns: column, pattern, count, frequency"""
        rows = self.read_csv(file_path)

        # Group patterns by column
        column_patterns: Dict[str, List[Dict]] = defaultdict(list)
        for row in rows:
            column_patterns[row["column"]].append(
                {
                    "pattern": row["pattern"],
                    "count": int(row["count"]),
                    "frequency": float(row["frequency"]),
                }
            )

        return [
            {
                "column_names": [col],
                "value": patterns,
            }
            for col, patterns in column_patterns.items()
        ]

    def parse_inline(
        self, values: List[Dict[str, Any]], table_name: str
    ) -> List[Dict[str, Any]]:
        return [
            {
                "column_names": [v["column"]],
                "value": v["patterns"],
            }
            for v in values
        ]
