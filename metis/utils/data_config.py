from pathlib import Path
from typing import Dict, List

import metis.globals


class DataConfig:
    def __init__(
        self,
        config: Dict,
    ):
        if "file_name" not in config:
            raise ValueError(f"Data config must include 'file_name' field.")
        if "name" not in config:
            raise ValueError(f"Data config must include 'name' field.")
        self.name: str = config["name"]
        self.file_name: str = Path(metis.globals.data_root) / config["file_name"]
        self.reference_file_name: str | None = (
            Path(metis.globals.data_root) / config["reference_file_name"]
            if config.get("reference_file_name")
            else None
        )
        self.loader: str | None = config.get("loader")
        self.delimiter: str = config.get("delimiter", ",")
        self.encoding: str = config.get("encoding", "utf-8")
        self.header: int = config.get("header", 0)
        self.nrows: int | None = config.get("nrows")
        self.usecols: List[str] | None = config.get("usecols")
        self.parse_dates: bool = config.get("parse_dates", False)
        self.decimals: str = config.get("decimals", ".")
        self.thousands: str | None = config.get("thousands")
        self.decimals: str = config.get("decimals", ".")
        self.thousands: str | None = config.get("thousands")
