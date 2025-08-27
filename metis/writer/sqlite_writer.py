import sqlite3
import os
import json
from typing import List

from metis.writer.writer import DQResultWriter
from metis.utils.result import DQResult

class SQLiteWriter(DQResultWriter):
    def __init__(self, writer_config) -> None:
        if not "db_name" in writer_config or not "table_name" in writer_config:
            raise ValueError("SQLite writer config must include 'db_name' and 'table_name' fields.")
        self.db_name = writer_config["db_name"]
        self.table_name = writer_config["table_name"]

        if not os.path.exists(self.db_name): #TODO: Make this modular for different databases
            conn = sqlite3.connect(self.db_name)
            cur = conn.cursor()
            cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mes_time TEXT NOT NULL,
                dq_value REAL NOT NULL,
                dq_dimension TEXT NOT NULL,
                dq_metric TEXT NOT NULL,
                column_name JSONB,
                row_index INTEGER,
                dq_annotations JSONB,
                dataset TEXT,
                table_name TEXT
            )
            ''')
            conn.commit()
            conn.close()

    def write(self, results: List[DQResult]) -> None:
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        
        for result in results:
            print(f"Writing result: {result.as_json()}")
            cur.execute(f'''
                INSERT INTO {self.table_name} (mes_time, dq_value, dq_dimension, dq_metric, column_name, row_index, dq_annotations, dataset, table_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.mesTime.to_pydatetime(),
                result.DQvalue,
                result.DQdimension,
                result.DQmetric,
                json.dumps(result.columnNames),
                result.rowIndex,
                json.dumps(result.DQannotations),
                result.dataset,
                result.tableName
            ))
        
        conn.commit()
        conn.close()