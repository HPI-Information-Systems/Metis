"""Importer for functional dependencies with HyFD/AIDFD/CFDFinder parsers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from .base import BaseImporter

if TYPE_CHECKING:
    from metis.profiling.data_profile_manager import DataProfileManager


class FDImporter(BaseImporter):
    """Importer for functional dependencies (fd task).

    Supports:
    - JSON inline: {"lhs": ["col1"], "rhs": "col2"}
    - HyFD/AIDFD format: [table.col1, table.col2]->table.col3
    - CFDFinder format: [table.col1]->table.col2#(pattern1);(pattern2)
    """

    # HyFD/AIDFD: [table.col1, table.col2]->table.col3
    HYFD_PATTERN = re.compile(r"\[([^\]]+)\]->([^\s\[#]+)")

    # CFDFinder: [table.col1]->table.col2#(pattern)
    CFD_PATTERN = re.compile(r"\[([^\]]+)\]->([^#\s]+)#(.+)")

    @property
    def task_name(self) -> str:
        return "fd"

    @property
    def profile_type(self) -> str:
        return "dependency"

    def parse_file(self, file_path: str, table_name: str) -> List[Dict[str, Any]]:
        """Parse FD output file (HyFD, AIDFD, or CFDFinder format)."""
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        fds: List[Dict[str, Any]] = []

        for line in content.splitlines():
            # Try CFDFinder first (more specific: requires #)
            match = self.CFD_PATTERN.search(line)
            if match:
                lhs_raw, rhs_raw, pattern_tableau = match.groups()
                lhs = self._parse_columns(lhs_raw, table_name)
                rhs = self._parse_column(rhs_raw, table_name)
                fds.append(
                    {
                        "column_names": sorted(lhs + [rhs]),
                        "value": {"lhs": lhs, "rhs": rhs},
                        "task_config": {"pattern_tableau": pattern_tableau},
                    }
                )
                continue

            # Fall back to HyFD/AIDFD
            match = self.HYFD_PATTERN.search(line)
            if match:
                lhs_raw, rhs_raw = match.groups()
                lhs = self._parse_columns(lhs_raw, table_name)
                rhs = self._parse_column(rhs_raw, table_name)
                fds.append(
                    {
                        "column_names": sorted(lhs + [rhs]),
                        "value": {"lhs": lhs, "rhs": rhs},
                    }
                )

        return fds

    def parse_inline(
        self, values: List[Dict[str, Any]], table_name: str
    ) -> List[Dict[str, Any]]:
        """Parse inline FD definitions."""
        return [
            {
                "column_names": sorted(v["lhs"] + [v["rhs"]]),
                "value": {"lhs": v["lhs"], "rhs": v["rhs"]},
            }
            for v in values
        ]

    def import_to_manager(
        self,
        config: Dict[str, Any],
        manager: DataProfileManager,
        dataset: str,
        table: str,
    ) -> int:
        """Import FDs using the dedicated store_fd method."""
        source = config.get("source", "imported")

        if "file" in config:
            profiles = self.parse_file(config["file"], table)
        elif "values" in config:
            profiles = self.parse_inline(config["values"], table)
        else:
            raise ValueError("FD config must have 'file' or 'values'")

        for profile in profiles:
            fd_value = profile["value"]
            manager.store_fd(
                lhs=fd_value["lhs"],
                rhs=fd_value["rhs"],
                dataset=dataset,
                table=table,
                source=source,
            )

        return len(profiles)

    @staticmethod
    def _parse_columns(raw: str, table_name: str) -> List[str]:
        """Parse comma-separated column list, stripping table prefix."""
        cols = [c.strip() for c in raw.split(",")]
        return [FDImporter._strip_table_prefix(c, table_name) for c in cols]

    @staticmethod
    def _parse_column(raw: str, table_name: str) -> str:
        """Parse single column, stripping table prefix."""
        return FDImporter._strip_table_prefix(raw.strip(), table_name)

    @staticmethod
    def _strip_table_prefix(col: str, table_name: str) -> str:
        """Strip table.csv. or table. prefix from column name."""
        # Handle both "table.csv.col" and "table.col" formats
        prefixes = [
            f"{table_name}.csv.",
            f"{table_name}.",
        ]
        for prefix in prefixes:
            if col.startswith(prefix):
                return col[len(prefix) :]
        # Also try without extension
        table_base = table_name.rsplit(".", 1)[0] if "." in table_name else table_name
        prefixes = [
            f"{table_base}.csv.",
            f"{table_base}.",
        ]
        for prefix in prefixes:
            if col.startswith(prefix):
                return col[len(prefix) :]
        return col
