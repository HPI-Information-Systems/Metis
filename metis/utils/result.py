from typing import List, Union
import pandas as pd

class DQResult:
    def __init__(
            self,
            mesTime: pd.Timestamp,
            DQvalue: float,
            DQdimension: str,
            DQmetric: str,
            columnNames: Union[List[str], None] = None,
            rowIndex: Union[int, None] = None,
            DQannotations: Union[dict, None] = None,
            dataset: Union[str, None] = None,
            tableName: Union[str, None] = None,
    ):
        """Create a data-quality result representing a single assessed value.

        Required arguments
        - `mesTime: pd.Timestamp`: The time at which the result was assessed.
        - `DQvalue: float`: Numeric outcome of the assessment (quantitative only).
        - `DQdimension: str`: Data quality dimension assessed (e.g. 'completeness', 'accuracy').
        - `DQmetric: str`: Name of the specific metric within the dimension.

        Optional arguments
        - `columnNames: Optional[List[str]]`: Columns that this result pertains to.
            For a column-level metric this is typically a single-item list; for
            a table-level metric this may be `None` or an empty list.
        - `rowIndex: Optional[int]`: Row index associated with the result. Use
            together with `columnNames` to represent a cell-level result, or for
            row-based metrics.
        - `DQannotations: Optional[dict]`: Arbitrary additional information
            produced by the metric (no fixed schema required).
        - `dataset: Optional[str]`: Dataset identifier. This is commonly set by
            the orchestrator and may be left `None` when creating results manually.
        - `tableName: Optional[str]`: Table name within the dataset. Also
            typically set by the `metis.dq_orchestrator.DQOrchestrator`.

        Notes
        - The `metis.dq_orchestrator.DQOrchestrator`
            will populate `dataset` and `tableName` when it assembles results
            across metrics and datasets.
        - `DQvalue` currently expects a quantitative (float) score. If you
            need to encode non-numeric outcomes consider using `DQannotations`
            to store auxiliary information while keeping `DQvalue` numeric.
        """
        self._mesTime = mesTime
        self._DQvalue = DQvalue
        self._DQdimension = DQdimension
        self._DQmetric = DQmetric
        self._dataset = dataset
        self._tableName = tableName
        self._columnNames = columnNames
        self._rowIndex = rowIndex
        self._DQannotations = DQannotations

    @property
    def mesTime(self):
        return self._mesTime

    @mesTime.setter
    def mesTime(self, value):
        self._mesTime = value

    @property
    def DQvalue(self):
        return self._DQvalue

    @DQvalue.setter
    def DQvalue(self, value):
        self._DQvalue = value

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
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value

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
    def DQannotations(self):
        return self._DQannotations

    @DQannotations.setter
    def DQannotations(self, value):
        self._DQannotations = value

    def as_json(self):
        return {
            "mesTime": self._mesTime,
            "DQvalue": self._DQvalue,
            "DQdimension": self._DQdimension,
            "DQmetric": self._DQmetric,
            "dataset": self._dataset,
            "tableName": self._tableName,
            "columnNames": self._columnNames,
            "rowIndex": self._rowIndex,
            "DQannotations": self._DQannotations
        }