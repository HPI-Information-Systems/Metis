from abc import ABC, abstractmethod
import pandas as pd

from metis.utils.data_config import DataConfig

class DataLoader(ABC):
    @abstractmethod
    def load(self, config: DataConfig) -> pd.DataFrame:
        """
        Load data from a source defined by the config.

        :param config: Configuration object containing the data source details.
        :return: DataFrame containing the loaded data.
        """
        pass
