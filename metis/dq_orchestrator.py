from typing import List
import pandas as pd
import json
import os
import sqlite3

import metis.globals
from metis.metric import Metric
from metis.utils.data_config import DataConfig
from metis.utils.result import DQResult
from metis.loader.csv_loader import CSVLoader
from metis.writer.sqlite_writer import SQLiteWriter

class DQOrchestrator:
    def __init__(self, db_name="dq_repository.db", result_table_name="dq_results") -> None:
        self.dataframes = {}
        self.data_paths = {}
        self.results = {} #TODO: Decide what to do with these in memory results

        if not os.path.exists(db_name): #TODO: Make this modular for different databases
            conn = sqlite3.connect(db_name)
            cur = conn.cursor()
            cur.execute(f'''
            CREATE TABLE IF NOT EXISTS {result_table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mesTime REAL NOT NULL,
                DQvalue REAL NOT NULL,
                DQdimension TEXT NOT NULL,
                DQmetric TEXT NOT NULL,
                columnName TEXT,
                rowIndex INTEGER,
                DQannotations TEXT,
                dataset TEXT,
                tableName TEXT
            )
            ''')
            conn.commit()
            conn.close()

        self.writer = SQLiteWriter(db_name, result_table_name)

    def load(self, data_loader_configs: List[str]) -> None:
        for config_path in data_loader_configs:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                config = DataConfig(config_data)
                config.file_name = os.path.join(metis.globals.data_root, config.file_name)
                if config.loader == "CSV":
                    loader = CSVLoader()
                    dataframe = loader.load(config)
                    self.dataframes[config.name] = dataframe
                    self.data_paths[config.name] = config_path

                else:
                    raise ValueError(f"Unsupported loader type: {config_data.get('loader', None)}")
            
    def assess(self, metrics: List[str], metric_configs: List[str]) -> None:
        results = []
        
        for metric, metric_config in zip(metrics, metric_configs):
            metric_class = Metric.registry.get(metric)
            if not metric_class:
                raise ValueError(f"Metric {metric} is not registered.")
            metric_instance = metric_class()
            for df_name, df in self.dataframes.items():
                incomplete_metric_results = metric_instance.assess(df, metric_config=metric_config) #TODO: Add reference data support
                for result in incomplete_metric_results:
                    result.tableName = df_name
                    result.dataset = self.data_paths[df_name]
                    results.append(result)

        self.writer.write(results)


    def getDQResult(query: str) -> List[DQResult]:
        pass