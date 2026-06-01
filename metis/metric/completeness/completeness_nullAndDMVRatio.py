from pathlib import Path
from typing import List, Literal

import pandas as pd

from metis.dismis.dismis import (
    DISMIS_EMPIRIC_PRECISION,
    DISMIS_EMPIRIC_RECALL,
    run_dismis_detection,
)
from metis.metric.completeness.completeness_nullAndDMVRatio_config import (
    completeness_nullAndDMVRatio_config,
)
from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.utils.disguised_missing_values.fahes.fahes import (
    FAHES_PRECISION,
    FAHES_RECALL,
    run_fahes,
)
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.result import DQResult

IS_VALID_MARKER = 0
IS_NULL_MARKER = 1
IS_DMV_MARKER = 2


class completeness_nullAndDMVRatio(Metric):
    def assess(
        self,
        data: pd.DataFrame,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        Assess the completeness of the data by checking for null values and disguised missing values. To detect disguised missing values, the FAHES algorithm by Qahtan et al. is applied to the data (paper: https://doi.org/10.1145/3219819.3220109). The completeness quality measurement is calculated as the ratio of valid values (non-null and non-disguised missing) to the total number of values. The metric can be configured using `completeness_nullAndDMVRatio_config` to calculate the completeness on column, row level, or table-level granularity.

        :param data: DataFrame to assess.
        :param reference: Optional reference DataFrame (not used in this metric).
        :param metric_config: Optional configuration for the metric.
        :return: List of DQResult objects containing completeness results.
        """

        config = self.load_config(metric_config, completeness_nullAndDMVRatio_config)

        marked_cells = pd.DataFrame(
            IS_VALID_MARKER, index=data.index, columns=data.columns
        )
        marked_cells[data.isna()] = IS_NULL_MARKER

        dismis_config = config.dismis_config
        metric_name_suffix = "_dismis" if dismis_config else "_fahes"
        certainty = None
        scores = None
        if dismis_config is None:
            dmvs = run_fahes(
                data,
                results_path=(
                    Path(config.explanatory_results_path) / "fahes"
                    if config.explanatory_results_path
                    else None
                ),
            )
            self.logger.info(f"Detected DMVs:\n{dmvs}")
            if dmvs is not None:
                for _, dmv_row in dmvs.iterrows():
                    col = dmv_row["Attribute Name"]
                    val = dmv_row["DMV"]
                    marked_cells.loc[data[col] == val, col] = IS_DMV_MARKER
            if not config.disable_dq_explanations:
                certainty = self.fahes_certainty(marked_cells)
                scores = certainty
        else:
            scores, predictions = run_dismis_detection(
                detectors=dismis_config.detectors,
                dataset=data,
                column_types=dismis_config.column_types,
                value_embeddings_path=dismis_config.value_embeddings_path,
                example_dmvs_path=dismis_config.example_dmvs_path,
                example_embeddings_path=dismis_config.example_embeddings_path,
                embedding_dim=dismis_config.embedding_dim,
                models_dir=dismis_config.models_dir,
                results_path=(
                    Path(config.explanatory_results_path) / "dismis"
                    if config.explanatory_results_path
                    else None
                ),
            )
            marked_cells[predictions == 1] = IS_DMV_MARKER
            if not config.disable_dq_explanations:
                certainty = self.dismis_certainty(marked_cells)

        completeness = (marked_cells == IS_VALID_MARKER).astype(int)

        if config.aggregation_axis is not None:
            mean_completeness = completeness.mean(axis=config.aggregation_axis)
            mean_certainty = (
                certainty.mean(axis=config.aggregation_axis)
                if certainty is not None
                else None
            )
            mean_scores = (
                scores.mean(axis=config.aggregation_axis)
                if scores is not None
                else None
            )

            if config.aggregate_all:
                table_completeness = mean_completeness.mean()
                table_certainty = (
                    mean_certainty.mean() if mean_certainty is not None else None
                )
                table_score = mean_scores.mean() if mean_scores is not None else None
                return [
                    DQResult(
                        timestamp=pd.Timestamp.now(),
                        DQvalue=table_completeness,
                        DQdimension=DQDimension.COMPLETENESS,
                        DQmetric=self.__class__.__name__ + metric_name_suffix,
                        columnNames=data.columns.tolist(),
                        DQexplanation=(
                            {
                                "certainty": float(table_certainty),
                                "score": float(table_score),
                            }
                            if table_certainty is not None and table_score is not None
                            else None
                        ),
                        DQgranularity=DQGranularity.TABLE,
                    )
                ]

            return self.create_aggregated_results(
                mean_completeness,
                mean_certainty,
                mean_scores,
                config.aggregation_axis,
                data.columns.tolist(),
                metric_name_suffix,
            )

        return self.create_flat_results(
            completeness, certainty, scores, metric_name_suffix
        )

    def fahes_certainty(self, marks: pd.DataFrame):
        # .replace with a dict sometimes throws an IndexError during pandas memory cleanup. Reason not yet identified, but using chained .replace calls seems to mitigate the issue.
        return (
            marks.replace(IS_VALID_MARKER, FAHES_RECALL)
            .replace(IS_NULL_MARKER, 1)
            .replace(IS_DMV_MARKER, FAHES_PRECISION)
        )

    def dismis_certainty(self, marks: pd.DataFrame):
        return (
            marks.replace(IS_VALID_MARKER, DISMIS_EMPIRIC_RECALL)
            .replace(IS_NULL_MARKER, 1)
            .replace(IS_DMV_MARKER, DISMIS_EMPIRIC_PRECISION)
        )

    def create_aggregated_results(
        self,
        mean_completeness: pd.Series,
        mean_certainty: pd.Series | None,
        mean_scores: pd.Series | None,
        aggregation_axis: Literal["index", "columns"],
        columns: List[str],
        metric_name_suffix: str,
    ) -> List[DQResult]:
        results = []
        granularity = (
            DQGranularity.ROW if aggregation_axis == "columns" else DQGranularity.COLUMN
        )
        if mean_certainty is not None and mean_scores is not None:
            for (index, completeness), certainty, score in zip(
                mean_completeness.items(), mean_certainty.values, mean_scores.values
            ):
                row_index = int(str(index)) if aggregation_axis == "columns" else None
                col_names = columns if aggregation_axis == "columns" else [str(index)]

                result = DQResult(
                    timestamp=pd.Timestamp.now(),
                    DQvalue=completeness,
                    DQdimension=DQDimension.COMPLETENESS,
                    DQmetric=self.__class__.__name__ + metric_name_suffix,
                    columnNames=col_names,
                    rowIndex=row_index,
                    DQexplanation={
                        "certainty": float(certainty),
                        "score": float(score),
                    },
                    DQgranularity=granularity,
                )
                results.append(result)
        else:
            for index, completeness in mean_completeness.items():
                row_index = int(str(index)) if aggregation_axis == "columns" else None
                col_names = columns if aggregation_axis == "columns" else [str(index)]

                result = DQResult(
                    timestamp=pd.Timestamp.now(),
                    DQvalue=completeness,
                    DQdimension=DQDimension.COMPLETENESS,
                    DQmetric=self.__class__.__name__ + metric_name_suffix,
                    columnNames=col_names,
                    rowIndex=row_index,
                    DQgranularity=granularity,
                )
                results.append(result)

        return results

    def create_flat_results(
        self,
        completeness: pd.DataFrame,
        certainty: pd.DataFrame | None,
        scores: pd.DataFrame | None,
        metric_name_suffix: str,
    ) -> List[DQResult]:
        results = []
        for col in completeness.columns:
            if certainty is not None and scores is not None:
                for row_index, (
                    completeness_value,
                    certainty_value,
                    score_value,
                ) in enumerate(
                    zip(
                        completeness[col].values,
                        certainty[col].values,
                        scores[col].values,
                    )
                ):
                    result = DQResult(
                        timestamp=pd.Timestamp.now(),
                        DQvalue=float(completeness_value),
                        DQdimension=DQDimension.COMPLETENESS,
                        DQmetric=self.__class__.__name__ + metric_name_suffix,
                        columnNames=[col],
                        rowIndex=row_index,
                        DQexplanation={
                            "certainty": float(certainty_value),
                            "score": float(score_value),
                        },
                        DQgranularity=DQGranularity.CELL,
                    )
                    results.append(result)
            else:
                for row_index, completeness_value in enumerate(
                    completeness[col].values
                ):
                    result = DQResult(
                        timestamp=pd.Timestamp.now(),
                        DQvalue=float(completeness_value),
                        DQdimension=DQDimension.COMPLETENESS,
                        DQmetric=self.__class__.__name__ + metric_name_suffix,
                        columnNames=[col],
                        rowIndex=row_index,
                        DQgranularity=DQGranularity.CELL,
                    )
                    results.append(result)
        return results
