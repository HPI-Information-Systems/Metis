from typing import List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder

from metis.metric.config import MetricConfig
from metis.metric.correctness.correctness_heinrich_config import (
    correctness_heinrich_config,
)
from metis.metric.metric import Metric
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.numbers import clamp
from metis.utils.result import DQResult
from metis.utils.similarity_measures.levenshtein_distance import levenshtein_distance


class correctness_heinrich(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        Assess the correctness of the data by calculating the deviation from the reference.

        :param data: DataFrame to assess.
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing correctness results.
        """
        config = self.load_config(metric_config, correctness_heinrich_config)
        reference_data = pd.read_csv(config.reference_file_path)

        if data.shape != reference_data.shape:
            raise ValueError(
                f"Data and reference must have the same shape for correctness assessment. Got data shape {data.shape} and reference shape {reference_data.shape}."
            )

        representativeness = None
        if config.superset_file_path is not None:
            superset_data = pd.read_csv(config.superset_file_path)
            representativeness = self.measure_representativeness(
                reference_data, superset_data
            )
            self.logger.info(f"Representativeness: {representativeness:.4f}")

        results = []

        for col_name in data.columns:
            correctness_measurements = data[col_name].combine(
                reference_data[col_name],
                lambda x, y: self.measure_correctness(
                    x, reference_value=y, dtype=data[col_name].dtype
                ),
            )
            for row_index, correctness in enumerate(correctness_measurements):
                result = DQResult(
                    timestamp=pd.Timestamp.now(),
                    DQvalue=float(correctness),
                    DQdimension=DQDimension.CORRECTNESS,
                    DQmetric=self.__class__.__name__,
                    columnNames=[col_name],
                    rowIndex=row_index,
                    DQgranularity=DQGranularity.CELL,
                    DQexplanation=(
                        {"certainty": float(representativeness)}
                        if representativeness is not None
                        else None
                    ),
                )
                results.append(result)

        return results

    def measure_correctness(self, value, *, reference_value, dtype) -> float:
        if value == reference_value:
            return 1
        if pd.isna(value) or pd.isna(reference_value):
            return 0
        if pd.api.types.is_numeric_dtype(dtype):
            return clamp(
                1
                - abs(value - reference_value) / max(abs(reference_value), abs(value)),
                0,
                1,
            )
        if pd.api.types.is_string_dtype(dtype):
            max_len = max(len(str(value)), len(str(reference_value)))
            correctness = (
                1 - levenshtein_distance(str(value), str(reference_value)) / max_len
            )
            return correctness
        raise ValueError(
            f"Unsupported dtype for correctness measurement: {dtype} (value: {value}, reference_value: {reference_value})"
        )

    def encode_and_scale_data(self, data: pd.DataFrame):
        encoder = OrdinalEncoder(encoded_missing_value=-1).fit(
            data.select_dtypes(exclude=["object", "number"])
        )
        X = encoder.transform(data.select_dtypes(exclude=["object", "number"]))
        X = np.hstack((data.select_dtypes(include=["number"]).fillna(-1).to_numpy(), X))
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X

    def measure_representativeness(
        self, subset_data: pd.DataFrame, superset_data: pd.DataFrame
    ):
        n_components = 2  # min(X.shape[0], X.shape[1], 10)
        X = self.encode_and_scale_data(superset_data)
        X_pca = PCA(n_components=n_components).fit(X)
        n_features = X.shape[1]

        self.logger.info(f"Subset size: {subset_data.size / superset_data.size}")

        Y = self.encode_and_scale_data(subset_data)
        Y_pca = PCA(n_components=n_components).fit(Y)

        delta_lambda = (
            n_features
            / (n_features + n_components - 2)
            * np.sum(
                np.abs(
                    X_pca.explained_variance_ratio_ - Y_pca.explained_variance_ratio_
                )
            )
        )
        delta_theta = (
            2
            / np.pi
            * min(
                np.arccos(
                    np.clip(
                        np.dot(X_pca.components_[0], Y_pca.components_[0]), -1.0, 1.0
                    )
                ),
                np.arccos(
                    np.clip(
                        np.dot(X_pca.components_[0], -Y_pca.components_[0]), -1.0, 1.0
                    )
                ),
            )
        )

        self.logger.info(f"∆λ: {delta_lambda}")
        self.logger.info(f"∆θ: {delta_theta}")
        return 1 - (delta_lambda + delta_theta) / 2
