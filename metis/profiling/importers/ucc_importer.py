"""Importer for unique column combinations."""

from typing import Any, Dict, List

from .base import BaseImporter


class UCCImporter(BaseImporter):
    """Importer for unique column combinations (ucc task)."""

    @property
    def task_name(self) -> str:
        return "ucc"

    @property
    def profile_type(self) -> str:
        return "dependency"

    def parse_file(self, file_path: str, table_name: str) -> List[Dict[str, Any]]:
        """Parse CSV with column: columns (comma-separated in quotes for multi)."""
        rows = self.read_csv(file_path)
        return [
            {
                "column_names": self._parse_columns(row["columns"]),
                "value": {"columns": self._parse_columns(row["columns"])},
            }
            for row in rows
        ]

    def parse_inline(
        self, values: List[Dict[str, Any]], table_name: str
    ) -> List[Dict[str, Any]]:
        return [
            {
                "column_names": sorted(v["columns"]),
                "value": {"columns": v["columns"]},
            }
            for v in values
        ]

    @staticmethod
    def _parse_columns(raw: str) -> List[str]:
        """Parse column list (may be comma-separated)."""
        return sorted([c.strip() for c in raw.split(",")])
