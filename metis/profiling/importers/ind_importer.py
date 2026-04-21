"""Importer for inclusion dependencies."""

from typing import Any, Dict, List

from .base import BaseImporter


class INDImporter(BaseImporter):
    """Importer for inclusion dependencies (ind task)."""

    @property
    def task_name(self) -> str:
        return "ind"

    @property
    def profile_type(self) -> str:
        return "dependency"

    def parse_file(self, file_path: str, table_name: str) -> List[Dict[str, Any]]:
        """Parse CSV with columns: dependent, referenced, referenced_table."""
        rows = self.read_csv(file_path)
        return [
            {
                "column_names": sorted(
                    self._parse_columns(row["dependent"])
                    + self._parse_columns(row["referenced"])
                ),
                "value": {
                    "dependent": self._parse_columns(row["dependent"]),
                    "referenced": self._parse_columns(row["referenced"]),
                    "referenced_table": row.get("referenced_table"),
                },
            }
            for row in rows
        ]

    def parse_inline(
        self, values: List[Dict[str, Any]], table_name: str
    ) -> List[Dict[str, Any]]:
        return [
            {
                "column_names": sorted(v["dependent"] + v["referenced"]),
                "value": {
                    "dependent": v["dependent"],
                    "referenced": v["referenced"],
                    "referenced_table": v.get("referenced_table"),
                },
            }
            for v in values
        ]

    @staticmethod
    def _parse_columns(raw: str) -> List[str]:
        """Parse column list (may be comma-separated)."""
        return [c.strip() for c in raw.split(",")]
