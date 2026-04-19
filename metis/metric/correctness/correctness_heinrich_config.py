from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from metis.metric.config import MetricConfig


@dataclass
class correctness_heinrich_config(MetricConfig):
    """
    Configuration class for the correctness_heinrich metric.

    :param reference_file_path: Path to the reference file that contains the correct values for the data. This file is used to compare against the assessed data in order to determine the correctness of the data. Must be of the same shape as the assessed data.
    :param superset_file_path: Optional path to a superset file that contains the full dataset beyond the reference file. The reference data will, in many cases, be a manually cleaned subset representing the real data. This allows the correctness measurements to be extrapolated on the full dataset. The superset data is used to calculate how well the reference data covers the superset data, to assess the certainty of correctness measurements.
    """

    reference_file_path_per_dataset: Dict[str, str | Path]
    superset_file_path_per_dataset: Dict[str, str | Path] | None = None
