import pandas as pd
from typing import List, Union

from metis.utils.result import DQResult
from metis.metric.metric import Metric

class Completeness(Metric):
    def assess(self, data: pd.DataFrame, reference: Union[pd.DataFrame, None] = None, metric_config: Union[str, None] = None) -> List[DQResult]:
        """
        Assess the completeness of the data by checking for missing values.
        
        :param data: DataFrame to assess.
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing completeness results.
        """
        results = []
        total_rows = len(data)
        
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            completeness = (total_rows - int(missing_count)) / total_rows
            
            result = DQResult(
                mesTime=pd.Timestamp.now(),
                DQvalue=completeness,
                DQdimension="Completeness",
                DQmetric="Completeness",
                columnNames=[column],
            )
            results.append(result)
        
        return results