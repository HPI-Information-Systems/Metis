from abc import ABC, abstractmethod
from typing import List

from metis.utils.result import DQResult

class DQResultWriter(ABC):
    @abstractmethod
    def write(self, results: List[DQResult]) -> None:
        pass