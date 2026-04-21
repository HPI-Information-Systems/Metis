"""Importer for Jaccard similarity tasks."""

from typing import Any, Dict, List

from .base import BaseImporter, auto_detect_type


class JaccardImporter(BaseImporter):
    """Importer for jaccard_similarity and jaccard_similarity_ngrams tasks."""

    def __init__(self, task_name: str):
        self._task_name = task_name

    @property
    def task_name(self) -> str:
        return self._task_name

    def parse_file(self, file_path: str, table_name: str) -> List[Dict[str, Any]]:
        """Parse CSV with columns: column1, column2, [n], value."""
        rows = self.read_csv(file_path)
        results = []

        for row in rows:
            col1 = row["column1"]
            col2 = row["column2"]
            value = float(row["value"])

            profile = {
                "column_names": sorted([col1, col2]),
                "value": value,
            }

            # For ngrams, include n as task_config
            if "n" in row and row["n"]:
                profile["task_config"] = {"n": int(row["n"])}

            results.append(profile)

        return results

    def parse_inline(
        self, values: List[Dict[str, Any]], table_name: str
    ) -> List[Dict[str, Any]]:
        results = []

        for v in values:
            profile = {
                "column_names": sorted([v["column1"], v["column2"]]),
                "value": v["value"],
            }

            if "n" in v:
                profile["task_config"] = {"n": v["n"]}

            results.append(profile)

        return results
