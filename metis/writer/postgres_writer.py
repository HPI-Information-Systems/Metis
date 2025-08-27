import json
import psycopg2
from typing import List

from metis.writer.writer import DQResultWriter
from metis.utils.result import DQResult

class PostgresWriter(DQResultWriter):
    def __init__(self, writer_config) -> None:
        required_keys = ("db_user", "db_pass", "db_name", "db_host", "db_port")
        if not all(k in writer_config for k in required_keys):
            raise ValueError("Postgres writer config must include 'db_user', 'db_pass', 'db_name', 'db_host', and 'db_port' fields.")
        
        self.table_name = writer_config.get("table_name")
        self.DB_USER = writer_config.get("db_user")
        self.DB_PASS = writer_config.get("db_pass")
        self.DB_NAME = writer_config.get("db_name")
        self.DB_HOST = writer_config.get("db_host")
        self.DB_PORT = writer_config.get("db_port")

        conn = self.connect()
        self.create_db_schema(conn)
        conn.close()

    def connect(self):
        conn = psycopg2.connect(
            dbname=self.DB_NAME,
            user=self.DB_USER,
            password=self.DB_PASS,
            host=self.DB_HOST,
            port=self.DB_PORT
        )
        return conn

    def create_db_schema(self, conn):
        query = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                mes_time TIMESTAMP WITH TIME ZONE NOT NULL,
                dq_value DOUBLE PRECISION NOT NULL,
                dq_dimension TEXT NOT NULL,
                dq_metric TEXT NOT NULL,
                column_name JSONB,
                row_index INTEGER,
                dq_annotations JSONB,
                dataset TEXT,
                table_name TEXT
            );
        """
        try:
            cursor = conn.cursor()
            cursor.execute(query=query)
            conn.commit()
            cursor.close()
        except Exception as e:
            print(f'Error when saving or connecting to DB: {e}')

    def write(self, results: List[DQResult]) -> None:
        conn = self.connect()
        cur = conn.cursor()
        
        for result in results:
            print(f"Writing result: {result.as_json()}")
            cur.execute(f'''
                INSERT INTO {self.table_name} (mes_time, dq_value, dq_dimension, dq_metric, column_name, row_index, dq_annotations, dataset, table_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
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