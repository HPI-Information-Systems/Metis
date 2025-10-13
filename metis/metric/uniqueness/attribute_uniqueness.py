import pandas as pd
from typing import List, Union

from metis.utils.result import DQResult
from metis.metric.metric import Metric

class AttributeUniqueness(Metric):
    def assess(self, data: pd.DataFrame, reference: Union[pd.DataFrame, None] = None, metric_config: Union[str, None] = None) -> List[DQResult]:
        """
        Assess the uniqueness for each attribute of a dataset by checking for unique values. 
        
        :param data: DataFrame to assess.
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing completeness results.
        """
        results = []
        total_rows = len(data)
        
        for column in data.columns:
            # Count values that appear exactly once (not duplicated)
            unique_count = (~data[column].duplicated(keep=False)).sum()
            uniqueness = unique_count / total_rows if total_rows > 0 else 0
            
            # Attributes with 100% unique values are candidate keys
            annotations = {}
            if uniqueness == 1.0:
                annotations = {"CandidateKey": "CandidateKey"}

            result = DQResult(
                mesTime=pd.Timestamp.now(),
                DQvalue=uniqueness,
                DQdimension="AttributeUniqueness",
                DQmetric="AttributeUniqueness",
                columnNames=[column],
                DQannotations=annotations
            )
            results.append(result)
        
        return results