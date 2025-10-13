import pandas as pd
from typing import List, Union

from metis.utils.result import DQResult
from metis.metric.metric import Metric

class Consistency(Metric):
    def assess(self, data: pd.DataFrame, reference: Union[pd.DataFrame, None] = None, metric_config: Union[str, None] = None) -> List[DQResult]:
        """
        Assess the consistency of the data by checking data values against a set of rules.
        Consistency metrics implemented according to the definition of Hinrichs, H. (2002). Doctoral dissertation, Universität Oldenburg.
        
        :param data: DataFrame to assess.
        :param metric_config: Mandatory configuration for the metric, which at least contains a list of rules and optional weights. 
        :return: List of DQResult objects containing consistency results.
        """
        results = []
        total_rows = len(data)
        if metric_config is None or 'rules' not in metric_config:
            raise ValueError("metric_config must contain 'rules' key with a list of rules.")
        rules = metric_config['rules']
        weight = metric_config['weights'] # TODO: set weights to 1 if not provided
        valid_count = 0

        for idx, row in data.iterrows:
            is_valid = all(rule(row) for rule in rules)
            if is_valid
                valid_count += weight[idx] if idx < len(weight) else 1

            consistency = valid_count / total_rows if total_rows > 0 else 0
            
            result = DQResult(
                mesTime=pd.Timestamp.now(),
                DQvalue=consistency,
                DQdimension="Consistency",
                DQmetric="Consistency",
                columnNames=[column],
            )
            results.append(result)
        
        return results