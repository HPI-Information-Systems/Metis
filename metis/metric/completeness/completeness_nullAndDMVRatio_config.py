import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Literal

from metis.dismis.detection.detection import DETECTORS_LITERAL
from metis.dismis.utils.types import COLUMN_TYPES
from metis.metric.config import MetricConfig

VALID_AGGREGATION_AXES = ["index", "columns", None]


@dataclass
class completeness_nullAndDMVRatio_config(MetricConfig):
    """
    Configuration class for the completeness_nullAndDMVRatio metric.

    :param aggregation_axis: Axis along which to aggregate completeness ('index': aggregate each column; 'columns': aggregate each row, None (default): no aggregation).
    :param aggregate_all: Whether to aggregate all completeness results into a single value for the whole input data.
    :param dismis_config: Configuration for the DISMIS detector to use for DMV detection. If None, FAHES will be used for detection.
    """

    aggregation_axis: Literal["index", "columns", None] = None
    aggregate_all: bool = False
    dismis_config: "completeness_nullAndDMVRatio_config_dismis | None" = None
    explanatory_results_path: str | None = None

    @classmethod
    def from_dict(cls, config_dict: Dict):
        dismis_config = config_dict.get("dismis_config")
        if dismis_config is not None:
            dismis_config = completeness_nullAndDMVRatio_config_dismis.from_dict(
                dismis_config
            )
        return cls(
            aggregation_axis=config_dict.get("aggregation_axis", None),
            aggregate_all=config_dict.get("aggregate_all", False),
            dismis_config=dismis_config,
            measure_runtime=config_dict.get("measure_runtime", False),
            measure_memory=config_dict.get("measure_memory", False),
            disable_dq_explanations=config_dict.get("disable_dq_explanations", False),
        )

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "aggregation_axis": self.aggregation_axis,
            "aggregate_all": self.aggregate_all,
            "dismis_config": (
                dataclasses.asdict(self.dismis_config) if self.dismis_config else None
            ),
            "measure_runtime": self.measure_runtime,
            "measure_memory": self.measure_memory,
            "disable_dq_explanations": self.disable_dq_explanations,
        }

    def validate(self):
        if self.aggregation_axis not in VALID_AGGREGATION_AXES:
            raise ValueError(
                f"aggregation_axis must be one of {VALID_AGGREGATION_AXES} but was {self.aggregation_axis}"
            )
        if not isinstance(self.aggregate_all, bool):
            raise ValueError(
                f"aggregate_all must be a boolean value but was {type(self.aggregate_all)}"
            )


@dataclass
class completeness_nullAndDMVRatio_config_dismis(MetricConfig):
    value_embeddings_path: str
    example_dmvs_path: str
    example_embeddings_path: str
    column_types: Dict[str, COLUMN_TYPES]
    embedding_dim: int = 128
    detectors: List[DETECTORS_LITERAL] = dataclasses.field(
        default_factory=lambda: [
            "frequent_values_1",
            "frequent_values_10",
            "length_outlier_distribution",
            "type_feature",
            "type_feature_2",
            "sign_outlier_feature",
            "key_distance_outlier_distribution",
            "capital_letter_outlier_distribution",
            "non_alphanumerical_outlier_distribution",
            "repeated_substring1_outlier_distribution",
            "repeated_substring2_outlier_distribution",
            "repeated_substring3_outlier_distribution",
            "nan_outlier",
            "frequency_outlier",
            "bucket_knn_square",
            "BucketPDFGoF",
            "pyod_mad",
            "multi_similar_samples",
            "multi_semantic_outlier_new_dub",
            "quantile",
            "syntactic_outlier",
            "semantic_comments",
            "semantic_placeholder",
            "semantic_unsure",
            "semantic_valid",
        ]
    )
    models_dir: str = "metis/dismis/models"
