from typing import List, Union

class DQResult:
    def __init__(
        self,
        mesTime: float,
        DQvalue: float,
        DQdimension: str,
        DQmetric: str,
        columnName: Union[str, None] = None,
        rowIndex: Union[int, None] = None,
        DQannotations: Union[dict, None] = None,
        dataset: Union[str, None] = None,
        tableName: Union[str, None] = None,
    ):
        self._mesTime = mesTime
        self._DQvalue = DQvalue
        self._DQdimension = DQdimension
        self._DQmetric = DQmetric
        self._dataset = dataset
        self._tableName = tableName
        self._columnName = columnName
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
    def columnName(self):
        return self._columnName

    @columnName.setter
    def columnName(self, value):
        self._columnName = value

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
            "columnName": self._columnName,
            "rowIndex": self._rowIndex,
            "DQannotations": self._DQannotations
        }