from dataclasses import dataclass
import re
from typing import Dict


@dataclass
class MetricConfig:
    """
    Base class for metric configuration.
    All metric configuration classes should inherit from this class.
    """

    @classmethod
    def from_dict(cls, config_dict: dict):
        """
        Create an instance of the configuration class from a dictionary.

        :param config_dict: A dictionary containing the configuration parameters.

        :return: An instance of the configuration class.
        """
        return cls(**config_dict)

    def validate(self):
        """
        Validate the configuration parameters.
        This method should be overridden by subclasses to implement specific validation logic.
        """
        pass


@dataclass
class DatasetDependentMetricConfig(MetricConfig):
    """
    Wrapper config for dataset-dependent metric configurations.
    This class can be used for metrics that require different configurations for different datasets. When passed to the orchestrator, the orchestrator will automatically select the appropriate configuration for each dataset based on the dataset name.
    """

    config_per_dataset: Dict[str, MetricConfig]

    def resolve_for_dataset(self, dataset_name: str) -> MetricConfig:
        """
        Resolve the appropriate configuration for a given dataset name.

        :param dataset_name: The name of the dataset for which to resolve the configuration.

        :return: The resolved MetricConfig for the given dataset name.

        :raises ValueError: If no matching configuration is found for the given dataset name.
        """
        for pattern, config in self.config_per_dataset.items():
            if re.match(pattern, dataset_name):
                return config
        raise ValueError(
            f"No matching configuration found for dataset '{dataset_name}'."
        )
