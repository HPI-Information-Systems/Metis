from pathlib import Path
from typing import Dict, List

import pandas as pd

from metis.utils.logging import logger
from metis.utils.result import DQResult


class CSVWriter:
    def __init__(self, writer_config: Dict) -> None:
        if "path" not in writer_config:
            raise ValueError(
                f"{self.__class__.__name__} requires a 'path' in the configuration."
            )

        self.path = Path(writer_config["path"])
        if not self.path.suffix == ".csv":
            raise ValueError(
                f"{self.__class__.__name__} path must end with .csv extension."
            )

        if self.path.exists():
            logger.warning(
                f"{self.__class__.__name__} path {self.path} already exists and will be overwritten."
            )

    def write(self, results: List[DQResult]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result.as_json() for result in results]).to_csv(
            self.path, index=False
        )
        logger.info(f"Results saved to {self.path.absolute()}")
