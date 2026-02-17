from typing import List, Union
import pandas as pd

class DQResult:
    def __init__(
            self,
            mesTime: pd.Timestamp,
            DQdimension: str,
            DQmetric: str,
            DQgranularity: str,
            DQvalue: float,
            DQexplanation: Union[dict, None] = None,
            runtime: Union[float, None] = None,
            tableName: Union[str, None] = None,
            columnNames: Union[List[str], None] = None,
            rowIndex: Union[int, None] = None,
            experimentTag:  Union[str, None] = None,
            dataset: Union[str, None] = None,
            configJson: Union[dict, None] = None,
    ):
        """Create a data-quality result representing a single assessed value.

        Required arguments
        - `mesTime: pd.Timestamp`: The time at which the result was assessed.
        - `DQdimension: str`: Data quality dimension assessed (e.g. 'completeness', 'accuracy').
        - `DQmetric: str`: Name of the specific metric within the dimension.
        - `DQgranularity: str`: Granularity of the metric (e.g. 'column', 'table', 'cell').
        - `DQvalue: float`: Numeric outcome of the assessment (quantitative only).

        Optional arguments
        - `DQexplanation: Optional[dict]`: Arbitrary additional information
            produced by the metric (no fixed schema required).
        - `runtime: Optional[float]`: Time taken to compute the metric, in seconds.
        - `tableName: Optional[str]`: Table name within the dataset. Also
            typically set by the `metis.dq_orchestrator.DQOrchestrator`.
        - `columnNames: Optional[List[str]]`: Columns that this result pertains to.
            For a column-level metric this is typically a single-item list; for
            a table-level metric this may be `None` or an empty list.
        - `rowIndex: Optional[int]`: Row index associated with the result. Use
            together with `columnNames` to represent a cell-level result, or for
            row-based metrics.
        - `experimentTag: Optional[str]`: Tag to identify a specific run
        - `dataset: Optional[str]`: Dataset identifier. This is commonly set by
            the orchestrator and may be left `None` when creating results manually.
        - `configJson: Optional[dict]`: Configuration used for the metric as a JSON object.

        Notes
        - The `metis.dq_orchestrator.DQOrchestrator`
            will populate `dataset` and `tableName` when it assembles results
            across metrics and datasets.
        - `DQvalue` currently expects a quantitative (float) score. If you
            need to encode non-numeric outcomes consider using `DQexplanation`
            to store auxiliary information while keeping `DQvalue` numeric.
        """
        self._mesTime = mesTime
        self._DQdimension = DQdimension
        self._DQmetric = DQmetric
        self._DQgranularity = DQgranularity
        self._DQvalue = DQvalue
        self._DQexplanation = DQexplanation
        self._runtime = runtime
        self._tableName = tableName
        self._columnNames = columnNames
        self._rowIndex = rowIndex
        self._experimentTag = experimentTag
        self._dataset = dataset
        self._configJson = configJson

    @property
    def mesTime(self):
        return self._mesTime

    @mesTime.setter
    def mesTime(self, value):
        self._mesTime = value

    @property
    def DQdimension(self):
        return self._DQdimension

    @DQdimension.setter
    def DQdimension(self, value):
        self._DQdimension = value

    @property
    def DQmetric(self):
        return self._DQmetric

    @DQmetric.setter
    def DQmetric(self, value):
        self._DQmetric = value

    @property
    def DQgranularity(self):
        return self._DQgranularity

    @DQgranularity.setter
    def DQgranularity(self, value):
        self._DQgranularity = value

    @property
    def DQvalue(self):
        return self._DQvalue

    @DQvalue.setter
    def DQvalue(self, value):
        self._DQvalue = value

    @property
    def DQexplanation(self):
        return self._DQexplanation

    @DQexplanation.setter
    def DQexplanation(self, value):
        self._DQexplanation = value

    @property
    def runtime(self):
        return self._runtime

    @runtime.setter
    def runtime(self, value):
        self._runtime = value

    @property
    def tableName(self):
        return self._tableName

    @tableName.setter
    def tableName(self, value):
        self._tableName = value

    @property
    def columnNames(self):
        return self._columnNames

    @columnNames.setter
    def columnNames(self, value):
        self._columnNames = value

    @property
    def rowIndex(self):
        return self._rowIndex

    @rowIndex.setter
    def rowIndex(self, value):
        self._rowIndex = value

    @property
    def experimentTag(self):
        return self._experimentTag

    @experimentTag.setter
    def experimentTag(self, value):
        self._experimentTag = value

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value
    
    @property
    def configJson(self):
        return self._configJson

    @configJson.setter
    def configJson(self, value):
        self._configJson = value

    def as_json(self):
        return {
            "mesTime": self._mesTime,
            "DQdimension": self._DQdimension,
            "DQmetric": self._DQmetric,
            "DQgranularity": self._DQgranularity,
            "DQvalue": self._DQvalue,
            "DQexplanation": self._DQexplanation,
            "runtime": self._runtime,
            "tableName": self._tableName,
            "columnNames": self._columnNames,
            "rowIndex": self._rowIndex,
            "experimentTag": self._experimentTag,
            "dataset": self._dataset,
            "configJson": self._configJson,
        }