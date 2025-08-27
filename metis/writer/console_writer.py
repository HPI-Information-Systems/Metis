from typing import List, Dict

from metis.utils.result import DQResult

class ConsoleWriter:
    def __init__(self, writer_config: Dict = None) -> None:
        pass

    def write(self, results: List[DQResult]) -> None:
        for result in results:
            print(result.as_json())