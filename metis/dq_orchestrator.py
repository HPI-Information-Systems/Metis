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
from metis.utils.dataset_checks import DatasetLoadError, ensure_not_empty
from metis.utils.garbage_collection import gc_paused
from metis.utils.logging import logger
from metis.utils.reference_loader import default_reference
from metis.utils.result import DQResult
from metis.writer.console_writer import ConsoleWriter
from metis.writer.csv_writer import CSVWriter
from metis.writer.database_writer import DatabaseWriter

FALLBACK_RESULTS_FILE = "dq_results_fallback.csv"


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
            config_data = self._read_data_config(config_path)
            config = DataConfig(config_data)

            if config.loader == "CSV":
                loader = CSVLoader()
                dataframe = loader.load(config)
                ensure_not_empty(dataframe, f"Dataset '{config.name}'")
                self.dataframes[config.name] = dataframe
                self.data_paths[config.name] = config_path

                if config.reference_file_name:
                    reference_config = DataConfig(config_data)
                    reference_config.file_name = config.reference_file_name
                    self.reference_dataframes[config.name] = loader.load(
                        reference_config, file_field="reference_file_name"
                    )

                # Import pre-computed data profiles
                if config.data_profiles:
                    self._import_data_profiles(
                        config.data_profiles, config_path, config.name
                    )
            else:
                raise ValueError(
                    f"Unsupported loader type: {config_data.get('loader', None)}"
                )

    @staticmethod
    def _read_data_config(config_path: str) -> dict:
        """
        Read a data config file.

        :param config_path: Path to the data config JSON file.
        :raises DatasetLoadError: If the file does not exist, is not valid
            JSON or does not hold a JSON object.
        :return: The parsed data config.
        """
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
        except FileNotFoundError as e:
            raise DatasetLoadError(
                f"The data config '{config_path}' does not exist."
            ) from e
        except json.JSONDecodeError as e:
            raise DatasetLoadError(
                f"The data config '{config_path}' is not valid JSON: {e}"
            ) from e
        if not isinstance(config_data, dict):
            raise DatasetLoadError(
                f"The data config '{config_path}' must hold a JSON object, "
                f"not {type(config_data).__name__}."
            )
        return config_data

    def assess(
        self, metrics: List[str], metric_configs: List[str | MetricConfig | None]
    ) -> List[DQResult]:
        """
        Assess every loaded dataset with the given metrics.

        :param metrics: Registry names of the metrics to run.
        :param metric_configs: One config per metric, in the same order.
            Use ``None`` for a metric's defaults.
        :raises ValueError: If the number of configs differs from the number
            of metrics, or a metric name is not registered.
        :raises EmptyDatasetError: If a loaded dataset has no rows or columns.
        :return: The results of all metrics on all datasets.
        """
        results: List[DQResult] = []

        if len(metrics) != len(metric_configs):
            raise ValueError(
                f"Got {len(metrics)} metric(s) but {len(metric_configs)} "
                "config(s). Pass one config per metric, in the same order, "
                "with None for a metric's defaults."
            )
        unknown = [m for m in metrics if m not in Metric.registry]
        if unknown:
            raise ValueError(f"Metric(s) not registered: {', '.join(unknown)}.")

        # Once, before any metric runs.
        for df_name, df in self.dataframes.items():
            ensure_not_empty(df, f"Dataset '{df_name}'")

        for metric, metric_config in zip(metrics, metric_configs):
            metric_class: Type[Metric] = Metric.registry[metric]
            metric_instance: Metric = metric_class()
            for df_name, df in self.dataframes.items():
                # Set profiling context so cached functions know the active dataset.
                if DataProfileManager.is_initialized():
                    DataProfileManager.get_instance().set_context(
                        dataset=self.data_paths[df_name], table=df_name
                    )
                reference = self._default_reference_for(metric_class, df_name)
                measure_runtime = self._should_measure_runtime(metric_config)
                if measure_runtime:
                    start = time.perf_counter()
                    with default_reference(reference), gc_paused():
                        incomplete_metric_results = metric_instance.assess(
                            data=df,
                            metric_config=metric_config,
                        )
                    elapsed = time.perf_counter() - start
                    for result in incomplete_metric_results:
                        result.runtime = elapsed
                else:
                    with default_reference(reference), gc_paused():
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

    def _default_reference_for(
        self, metric_class: Type[Metric], df_name: str
    ) -> pd.DataFrame | None:
        """
        Return the dataset-level reference a metric falls back to.

        Only metrics that cannot run without a reference get the default. A
        metric with an optional reference keeps its own fallback, for example
        the built-in word list of ``validity_outOfVocabulary``.

        :param metric_class: The metric about to be assessed.
        :param df_name: Name of the dataset the metric runs on.
        :return: The reference loaded from ``reference_file_name``, or ``None``.
        """
        meta = metric_class.meta
        if meta is None or not meta.requires_reference:
            return None
        return self.reference_dataframes.get(df_name)

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
