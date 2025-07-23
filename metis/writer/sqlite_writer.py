import sqlite3
from typing import List

from metis.writer.writer import DQResultWriter
from metis.utils.result import DQResult

class SQLiteWriter(DQResultWriter):
    def __init__(self, db_name: str, table_name: str) -> None:
        self.db_name = db_name
        self.table_name = table_name

    def write(self, results: List[DQResult]) -> None:
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        
        for result in results:
            print(f"Writing result: {result.as_json()}")
            cur.execute(f'''
                INSERT INTO {self.table_name} (mesTime, DQvalue, DQdimension, DQmetric, columnName, rowIndex, DQannotations, dataset, tableName)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.mesTime.timestamp(),
                result.DQvalue,
                result.DQdimension,
                result.DQmetric,
                result.columnName,
                result.rowIndex,
                result.DQannotations,
                result.dataset,
                result.tableName
            ))
        
        conn.commit()
        conn.close()