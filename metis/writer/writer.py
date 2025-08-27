from abc import ABC, abstractmethod
from typing import List, Dict

from metis.utils.result import DQResult

class DQResultWriter(ABC):
    @abstractmethod
    def __init__(self, writer_config: Dict) -> None:
        pass

    @abstractmethod
    def write(self, results: List[DQResult]) -> None:
        pass