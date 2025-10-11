import json
from typing import Dict, List, Type

import pandas as pd

from metis.loader.csv_loader import CSVLoader
from metis.metric import Metric
from metis.utils.data_config import DataConfig
from metis.utils.result import DQResult
from metis.writer.console_writer import ConsoleWriter
from metis.writer.postgres_writer import PostgresWriter
from metis.writer.sqlite_writer import SQLiteWriter


class DQOrchestrator:
    def __init__(self, writer_config_path: str | None = None) -> None:
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.reference_dataframes: Dict[str, pd.DataFrame] = {}
        self.data_paths: Dict[str, str] = {}
        self.results: Dict[str, DQResult] = (
            {}
        )  # TODO: Decide what to do with these in memory results

        self.writer = ConsoleWriter({})
        if writer_config_path:
            with open(writer_config_path, "r") as f:
                writer_config = json.load(f)
            if not "writer_name" in writer_config:
                raise ValueError("Writer config must include 'writer_name' field.")
            if writer_config["writer_name"] == "sqlite":
                self.writer = SQLiteWriter(writer_config)
            elif writer_config["writer_name"] == "postgres":
                self.writer = PostgresWriter(writer_config)

    def load(self, data_loader_configs: List[str]) -> None:
        for config_path in data_loader_configs:
            with open(config_path, "r") as f:
                config_data = json.load(f)
                config = DataConfig(config_data)

                if config.loader == "CSV":
                    loader = CSVLoader()
                    dataframe = loader.load(config)
                    self.dataframes[config.name] = dataframe
                    self.data_paths[config.name] = config_path

                    if config.reference_file_name:
                        reference_config = DataConfig(config_data)
                        reference_config.file_name = config.reference_file_name
                        reference_dataframe = loader.load(reference_config)
                        self.reference_dataframes[config.name] = reference_dataframe
                else:
                    raise ValueError(
                        f"Unsupported loader type: {config_data.get('loader', None)}"
                    )

    def assess(self, metrics: List[str], metric_configs: List[str | None]) -> None:
        results = []

        for metric, metric_config in zip(metrics, metric_configs):
            metric_class: Type[Metric] | None = Metric.registry.get(metric)
            if not metric_class:
                raise ValueError(f"Metric {metric} is not registered.")
            metric_instance: Metric = metric_class()
            for df_name, df in self.dataframes.items():
                incomplete_metric_results = metric_instance.assess(
                    data=df,
                    reference=self.reference_dataframes.get(df_name),
                    metric_config=metric_config,
                )
                for result in incomplete_metric_results:
                    result.tableName = df_name
                    result.dataset = self.data_paths[df_name]
                    results.append(result)

        self.writer.write(results)

    def get_dq_result(self, query: str) -> List[DQResult]:
        return []
