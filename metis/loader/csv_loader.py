import pandas as pd

from metis.loader.loader import DataLoader
from metis.utils.data_config import DataConfig


class CSVLoader(DataLoader):
    def load(self, config: DataConfig) -> pd.DataFrame:
        """
        Load data from a CSV file specified by the config.

        :param config: DataConfig object containing the CSV parsing details.
        :return: DataFrame containing the loaded data.
        """

        return pd.read_csv(
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
