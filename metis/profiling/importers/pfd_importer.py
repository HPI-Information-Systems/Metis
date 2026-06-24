"""Importer for partial functional dependencies (pfd task)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

from .base import BaseImporter
from .fd_importer import FDImporter

if TYPE_CHECKING:
    from metis.profiling.data_profile_manager import DataProfileManager


class PFDImporter(BaseImporter):
    """Importer for partial functional dependencies (pfd task).

    Each entry carries the FD plus its partial threshold rho in (0, 1] and a
    gpdep-based genuineness weight in [0, 1]. Consumed by the
    ``consistency_cpfd`` metric.

    Supports:
    - JSON inline:
        {"lhs": ["col1"], "rhs": "col2", "partial": 0.95, "gpdep": 0.87}
    - Partial HyFD text format (one pFD per line):
        [table.csv.col1, table.csv.col2]->table.csv.col3#0.95#0.87
        i.e. ``[<lhs>]-><rhs>#<partialScore>#<weightScore>``
    """

    # Partial HyFD: [table.csv.col1, table.csv.col2]->table.csv.col3#0.95#0.87
    # The LHS may be empty (e.g. "[]->table.csv.col#1.0#0.0") for constant-column
    # claims emitted by the discoverer; those are kept as pFDs with lhs=[].
    PFD_PATTERN = re.compile(
        r"\[([^\]]*)\]->([^\s#]+)#([\d.]+)#([\d.]+)"
    )

    @property
    def task_name(self) -> str:
        return "pfd"

    @property
    def profile_type(self) -> str:
        return "dependency"

    def parse_file(self, file_path: str, table_name: str) -> List[Dict[str, Any]]:
        path = Path(file_path)
        content = path.read_text(encoding="utf-8")

        pfds: List[Dict[str, Any]] = []
        for line in content.splitlines():
            match = self.PFD_PATTERN.search(line)
            if not match:
                continue
            lhs_raw, rhs_raw, partial, gpdep = match.groups()
            lhs = (
                FDImporter._parse_columns(lhs_raw, table_name)
                if lhs_raw.strip()
                else []
            )
            rhs = FDImporter._parse_column(rhs_raw, table_name)
            pfds.append(
                {
                    "column_names": sorted(lhs + [rhs]),
                    "value": {
                        "lhs": lhs,
                        "rhs": rhs,
                        "partial": float(partial),
                        "gpdep": float(gpdep),
                    },
                }
            )
        return pfds

    def parse_inline(
        self, values: List[Dict[str, Any]], table_name: str
    ) -> List[Dict[str, Any]]:
        return [
            {
                "column_names": sorted(v["lhs"] + [v["rhs"]]),
                "value": {
                    "lhs": v["lhs"],
                    "rhs": v["rhs"],
                    "partial": float(v["partial"]),
                    "gpdep": float(v["gpdep"]),
                },
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
        source = config.get("source", "imported")

        if "file" in config:
            profiles = self.parse_file(config["file"], table)
        elif "values" in config:
            profiles = self.parse_inline(config["values"], table)
        else:
            raise ValueError("pFD config must have 'file' or 'values'")

        for profile in profiles:
            pfd = profile["value"]
            manager.store_pfd(
                lhs=pfd["lhs"],
                rhs=pfd["rhs"],
                partial=pfd["partial"],
                gpdep=pfd["gpdep"],
                dataset=dataset,
                table=table,
                source=source,
            )

        return len(profiles)
