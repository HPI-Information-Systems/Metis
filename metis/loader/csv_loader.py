import csv
import itertools

import pandas as pd

from metis.loader.loader import DataLoader
from metis.utils.data_config import DataConfig
from metis.utils.dataset_checks import DatasetLoadError
from metis.utils.logging import logger


class CSVLoader(DataLoader):
    def load(
        self,
        config: DataConfig,
        file_field: str = "file_name",
        config_label: str = "the data config",
    ) -> pd.DataFrame:
        """
        Load data from a CSV file specified by the config.

        :param config: DataConfig object containing the CSV parsing details.
        :param file_field: The config field that names the file, used in
            messages.
        :param config_label: The config that holds the file and its parsing
            options, used in messages.
        :raises DatasetLoadError: If the file is missing, empty, not in the
            configured encoding or not a valid CSV file. The message names
            the file and the config field to check.
        :return: DataFrame containing the loaded data.
        """
        try:
            data = pd.read_csv(
                config.file_name,
                delimiter=config.delimiter,
                encoding=config.encoding,
                header=config.header,
                nrows=config.nrows,
                usecols=config.usecols,
                parse_dates=config.parse_dates,
                decimal=config.decimals,
                thousands=config.thousands
            )
        except FileNotFoundError as e:
            raise DatasetLoadError(
                f"Dataset '{config.name}' could not be loaded: the file "
                f"'{config.file_name}' does not exist. Check '{file_field}' in "
                f"{config_label}. It is resolved relative to the data directory."
            ) from e
        except pd.errors.EmptyDataError as e:
            raise DatasetLoadError(
                f"Dataset '{config.name}' could not be loaded: the file "
                f"'{config.file_name}' is empty."
            ) from e
        except UnicodeDecodeError as e:
            raise DatasetLoadError(
                f"Dataset '{config.name}' could not be loaded: the file "
                f"'{config.file_name}' is not valid {config.encoding}. Set "
                f"'encoding' in {config_label} to the encoding of the file."
            ) from e
        except pd.errors.ParserError as e:
            raise DatasetLoadError(
                f"Dataset '{config.name}' could not be loaded: the file "
                f"'{config.file_name}' is not a valid CSV file ({e}). Check "
                f"'delimiter' in {config_label} and the rows of the file."
            ) from e
        except ValueError as e:
            raise DatasetLoadError(
                f"Dataset '{config.name}' could not be loaded from "
                f"'{config.file_name}': {e}"
            ) from e
        self._warn_on_duplicate_columns(config)
        return data

    @staticmethod
    def _warn_on_duplicate_columns(config: DataConfig) -> None:
        """
        Warn when the header row repeats a column name.

        pandas silently renames the repeats to ``name.1``, ``name.2`` and so
        on, so the check reads the raw header row. It does nothing when the
        header is not a single row index or the row cannot be read.

        :param config: DataConfig object containing the CSV parsing details.
        :return: None.
        """
        if not isinstance(config.header, int) or isinstance(config.header, bool):
            return
        if config.header < 0:
            return
        try:
            with open(config.file_name, encoding=config.encoding, newline="") as f:
                rows = (row for row in csv.reader(f, delimiter=config.delimiter) if row)
                names = next(itertools.islice(rows, config.header, None), [])
        except (OSError, ValueError, TypeError, csv.Error):
            return
        repeated = [n for n in dict.fromkeys(names) if n and names.count(n) > 1]
        if repeated:
            logger.warning(
                f"Dataset '{config.name}': the header row of '{config.file_name}' "
                f"repeats the column name(s) {repeated}. pandas renamed the "
                "repeats to 'name.1', 'name.2' and so on."
            )
