"""Base class for data profile importers."""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from metis.profiling.data_profile_manager import DataProfileManager


def auto_detect_type(value: str) -> int | float | bool | str:
    """Auto-detect Python type from string value."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


class BaseImporter(ABC):
    """Abstract base class for data profile importers.

    Each importer handles one or more related task types (e.g., ScalarImporter
    handles all simple column->value tasks).
    """

    @property
    @abstractmethod
    def task_name(self) -> str:
        """The profiling task name this importer handles."""
        ...

    @property
    def profile_type(self) -> str:
        """Profile type for storage (single_column, dependency, etc.)."""
        return "single_column"

    @abstractmethod
    def parse_file(self, file_path: str, table_name: str) -> List[Dict[str, Any]]:
        """Parse an external file and return a list of profile dicts.

        Args:
            file_path: Path to the file to parse
            table_name: Name of the table (for column name extraction)

        Returns:
            List of dicts with keys: column_names, value, task_config (optional)
        """
        ...

    @abstractmethod
    def parse_inline(
        self, values: List[Dict[str, Any]], table_name: str
    ) -> List[Dict[str, Any]]:
        """Parse inline JSON values.

        Args:
            values: List of value dicts from the config
            table_name: Name of the table

        Returns:
            List of dicts with keys: column_names, value, task_config (optional)
        """
        ...

    def import_to_manager(
        self,
        config: Dict[str, Any],
        manager: DataProfileManager,
        dataset: str,
        table: str,
    ) -> int:
        """Import profiles from config into the DataProfileManager.

        Args:
            config: The task config dict with 'source' and either 'file' or 'values'
            manager: DataProfileManager instance
            dataset: Dataset identifier
            table: Table name

        Returns:
            Number of profiles imported
        """
        source = config.get("source", "imported")

        if "file" in config:
            profiles = self.parse_file(config["file"], table)
        elif "values" in config:
            profiles = self.parse_inline(config["values"], table)
        else:
            raise ValueError(
                f"Config for {self.task_name} must have 'file' or 'values'"
            )

        for profile in profiles:
            manager.store(
                column_names=profile["column_names"],
                dp_task_name=self.task_name,
                value=profile["value"],
                task_config=profile.get("task_config"),
                profile_type=self.profile_type,
                source=source,
                dataset=dataset,
                table=table,
            )

        return len(profiles)

    @staticmethod
    def read_csv(file_path: str) -> List[Dict[str, str]]:
        """Read a CSV file and return list of row dicts."""
        path = Path(file_path)
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
