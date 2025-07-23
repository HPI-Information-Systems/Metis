from abc import ABC, abstractmethod
import pandas as pd

class DataLoader(ABC):
    @abstractmethod
    def load(self, config: str) -> pd.DataFrame:
        """
        Load data from a source defined by the config.
        
        :param config: Configuration string or path to the configuration file.
        :return: DataFrame containing the loaded data.
        """
        pass