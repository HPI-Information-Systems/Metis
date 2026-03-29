import json
import time
import traceback
from typing import Dict, List, Type

import pandas as pd

from metis.database import Database
from metis.loader.csv_loader import CSVLoader
from metis.metric import Metric
from metis.metric.config import MetricConfig
from metis.profiling.data_profile_manager import DataProfileManager
from metis.profiling.importers import get_importer
from metis.utils.data_config import DataConfig
from metis.utils.logging import logger
from metis.utils.result import DQResult
from metis.writer.console_writer import ConsoleWriter
from metis.writer.csv_writer import CSVWriter
from metis.writer.database_writer import DatabaseWriter

FALLBACK_RESULTS_FILE = "dq_results_fallback.csv"


class DQOrchestrator:
    def __init__(self, writer_config_path: str | None = None) -> None:
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.data_paths: Dict[str, str] = {}
        self.results: Dict[str, DQResult] = (
            {}
        )  # TODO: Decide what to do with these in memory results

        self.writer = ConsoleWriter({})
        if writer_config_path:
            with open(writer_config_path, "r") as f:
                writer_config = json.load(f)
            if "writer_name" not in writer_config:
                raise ValueError("Writer config must include 'writer_name' field.")

            if writer_config["writer_name"] in ("sqlite", "postgres"):
                # Create central DB instance
                db = Database(writer_config["writer_name"], writer_config)
                self.writer = DatabaseWriter(db)
            elif writer_config["writer_name"] == "csv":
                self.writer = CSVWriter(writer_config)

        # Initialize profile cache using the central DB instance.
        # No caching if no DB is configured.
        if Database.is_initialized():
            DataProfileManager.initialize(Database.get_instance())

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

                    # Import pre-computed data profiles
                    if config.data_profiles:
                        self._import_data_profiles(
                            config.data_profiles, config_path, config.name
                        )
                else:
                    raise ValueError(
                        f"Unsupported loader type: {config_data.get('loader', None)}"
                    )

    def assess(
        self, metrics: List[str], metric_configs: List[str | MetricConfig | None]
    ) -> List[DQResult]:
        results: List[DQResult] = []

        for metric, metric_config in zip(metrics, metric_configs):
            metric_class: Type[Metric] | None = Metric.registry.get(metric)
            if not metric_class:
                raise ValueError(f"Metric {metric} is not registered.")
            metric_instance: Metric = metric_class()
            for df_name, df in self.dataframes.items():
                # Set profiling context so cached functions know the active dataset.
                if DataProfileManager.is_initialized():
                    DataProfileManager.get_instance().set_context(
                        dataset=self.data_paths[df_name], table=df_name
                    )
                measure_runtime = self._should_measure_runtime(metric_config)
                if measure_runtime:
                    start = time.perf_counter()
                    incomplete_metric_results = metric_instance.assess(
                        data=df,
                        metric_config=metric_config,
                    )
                    elapsed = time.perf_counter() - start
                    for result in incomplete_metric_results:
                        result.runtime = elapsed
                else:
                    incomplete_metric_results = metric_instance.assess(
                        data=df,
                        metric_config=metric_config,
                    )
                for result in incomplete_metric_results:
                    result.tableName = df_name
                    result.dataset = self.data_paths[df_name]
                    results.append(result)

        try:
            logger.info(
                f"Writing {len(results)} results using {self.writer.__class__.__name__}"
            )
            self.writer.write(results)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error writing results: {e}")
            try:
                logger.warning("Trying to save results to csv as fallback...")
                CSVWriter({"path": FALLBACK_RESULTS_FILE}).write(results)
            except Exception as e:
                logger.error(f"Failed to save results to csv: {e}")
                raise e
        return results

    def get_dq_result(self, query: str) -> List[DQResult]:
        return []

    def _should_measure_runtime(self, metric_config: MetricConfig | str | None) -> bool:
        if metric_config is None:
            return False

        if isinstance(metric_config, MetricConfig):
            return getattr(metric_config, "measure_runtime", False)

        try:
            parsed = json.loads(metric_config)
        except Exception:
            try:
                with open(metric_config, "r") as f:
                    parsed = json.load(f)
            except Exception:
                return False

        if not isinstance(parsed, dict):
            return False

        return bool(parsed.get("measure_runtime", False))

    def _import_data_profiles(self, profiles: dict, dataset: str, table: str) -> None:
        """Import pre-computed data profiles from config.

        Args:
            profiles: Dict mapping task_name -> {source, file|values}
            dataset: Dataset identifier (config path)
            table: Table name
        """
        if not DataProfileManager.is_initialized():
            return

        manager = DataProfileManager.get_instance()

        for task_name, task_config in profiles.items():
            try:
                importer = get_importer(task_name)
                count = importer.import_to_manager(task_config, manager, dataset, table)
            except KeyError as e:
                raise ValueError(f"Unknown data profile task: {task_name}") from e
