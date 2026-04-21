"""Importer for histogram profiling tasks."""

from collections import defaultdict
from typing import Any, Dict, List

from .base import BaseImporter


class HistogramImporter(BaseImporter):
    """Importer for equi_width_histogram and equi_depth_histogram tasks."""

    def __init__(self, task_name: str):
        self._task_name = task_name

    @property
    def task_name(self) -> str:
        return self._task_name

    def parse_file(self, file_path: str, table_name: str) -> List[Dict[str, Any]]:
        """Parse CSV with columns: column, bin_min, bin_max, count"""
        rows = self.read_csv(file_path)

        # Group bins by column
        column_bins: Dict[str, List[Dict]] = defaultdict(list)
        for row in rows:
            column_bins[row["column"]].append(
                {
                    "min": float(row["bin_min"]),
                    "max": float(row["bin_max"]),
                    "count": int(row["count"]),
                }
            )

        return [
            {
                "column_names": [col],
                "value": self._bins_to_histogram(bins),
            }
            for col, bins in column_bins.items()
        ]

    def parse_inline(
        self, values: List[Dict[str, Any]], table_name: str
    ) -> List[Dict[str, Any]]:
        return [
            {
                "column_names": [v["column"]],
                "value": self._bins_to_histogram(v["bins"]),
            }
            for v in values
        ]

    @staticmethod
    def _bins_to_histogram(bins: List[Dict]) -> Dict:
        """Convert list of bin dicts to histogram format.

        Input: [{"min": 0, "max": 30, "count": 100}, ...]
        Output: {"bin_edges": [(0, 30), ...], "frequencies": [100, ...]}
        """
        bin_edges = [(b["min"], b["max"]) for b in bins]
        frequencies = [b["count"] for b in bins]
        return {"bin_edges": bin_edges, "frequencies": frequencies}
