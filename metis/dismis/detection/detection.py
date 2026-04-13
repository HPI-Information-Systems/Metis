from typing import Any, Callable, Dict, List, Literal

import pandas as pd

from metis.dismis.detection.detectors.bucketknn_square import (
    BucketKNN as BucketKNNSquare,
)
from metis.dismis.detection.detectors.detector import DMVDetector
from metis.dismis.detection.detectors.distribution import (
    BucketPDFGoF,
    DistributionFitDetector,
)
from metis.dismis.detection.detectors.dummy import NanDetector, QuantileDetector
from metis.dismis.detection.detectors.frequent_value import (
    FrequentValuesDetector,
    FrequentValuesDetector2,
)
from metis.dismis.detection.detectors.llm_classifier import LLMClassifierDetector
from metis.dismis.detection.detectors.pyod import (
    CapitalLetterOutlierDetectorDist,
    FrequencyOutlierDetector,
    KeyDistanceOutlierDetectorDist,
    KeyDistanceOutlierDetectorFeat,
    KeyDistanceOutlierDetectorOD,
    LengthOutlierDetectorDist,
    LengthOutlierDetectorFeat,
    LengthOutlierDetectorOD,
    MultiSemanticOutlierDetectorNew,
    NonAlphanumericalOutlierDetectorDist,
    PyODDetector,
    PyODDetector2,
    RepeatedSubstringOutlierDetectorDist,
    RepeatedSubstringOutlierDetectorFeat,
    RepeatedSubstringOutlierDetectorOD,
    SemanticDetector,
    SemanticOutlierDetector,
    SemanticOutlierDetectorNew,
    SemanticOutlierDetectorNewDub,
    SignOutlierDetectorFeat,
    TypeOutlierDetector,
    TypeOutlierDetector2,
)
from metis.dismis.detection.detectors.similar_samples import (
    FAISSNoDubSimilarSamplesDetector,
    FAISSSimilarSamplesDetector,
    MultiSimilarSamplesDetector,
)
from metis.dismis.detection.detectors.syntactic import SyntacticDetector
from metis.dismis.preparation.generate_example_dmvs import EXAMPLE_DMV_CATEGORIES
from metis.dismis.preparation.pollution.errors.error import DMV
from metis.dismis.utils.types import COLUMN_TYPES

FAST = False


def run_detector(
    detector_name,
    polluted_dataset,
    detector_instance,
    types,
    target_columns,
    embeddings,
):
    print(f"Running detector: {detector_name}")
    scores, predictions, times, assessed_columns = detector_instance(
        polluted_dataset, types, target_columns, embeddings
    )
    return detector_name, (scores, predictions), times, assessed_columns


DETECTORS_LITERAL = Literal[
    "frequent_values_1",
    "frequent_values_3",
    "frequent_values_10",
    "frequent_values_25",
    "bucket_knn_square",
    "distribution",
    "BucketPDFGoF",
    "pyod_knn",
    "pyod_lof",
    "pyod_iforest",
    "pyod_loda",
    "pyod_hbos",
    "pyod_cblof",
    "pyod_cof",
    "pyod_gmm",
    "pyod_mad",
    "RobustZ",
    "Quantile",
    "ESD",
    "pyod_mad2",
    "pyod_mad3",
    "frequent_values_1_2",
    "frequent_values_1_3",
    "type_feature_2",
    "frequency_outlier",
    "length_outlier_feature",
    "length_outlier_distribution",
    "length_outlier_MAD",
    "length_outlier_hbos",
    "repeated_substring1_outlier_feature",
    "repeated_substring1_outlier_distribution",
    "repeated_substring1_outlier_MAD",
    "repeated_substring1_outlier_hbos",
    "repeated_substring2_outlier_feature",
    "repeated_substring2_outlier_distribution",
    "repeated_substring2_outlier_MAD",
    "repeated_substring2_outlier_hbos",
    "repeated_substring3_outlier_feature",
    "repeated_substring3_outlier_distribution",
    "repeated_substring3_outlier_MAD",
    "repeated_substring3_outlier_hbos",
    "key_distance_outlier_feature",
    "key_distance_outlier_distribution",
    "key_distance_outlier_MAD",
    "key_distance_outlier_hbos",
    "capital_letter_outlier_distribution",
    "non_alphanumerical_outlier_distribution",
    "sign_outlier_feature",
    "semantic_outlier_10",
    "semantic_outlier_25",
    "semantic_outlier_100",
    "semantic_outlier_3_new",
    "semantic_outlier_10_new",
    "semantic_outlier_25_new",
    "semantic_outlier_100_new",
    "multi_semantic_outlier_new",
    "multi_semantic_outlier_new_dub",
    "semantic_outlier_3_new_dub",
    "semantic_outlier_10_new_dub",
    "semantic_outlier_25_new_dub",
    "semantic_outlier_100_new_dub",
    "type_feature",
    "nan_outlier",
    "quantile",
    "approximate_similar_samples_5",
    "approximate_similar_samples_corr_5",
    "approximate_similar_samples_10",
    "approximate_similar_samples_corr_10",
    "approximate_similar_samples_25",
    "approximate_similar_samples_corr_25",
    "no_duplicate_similar_samples_5",
    "no_duplicate_similar_samples_10",
    "no_duplicate_similar_samples_25",
    "multi_similar_samples",
    "syntactic_outlier",
    "llm_classifier",
    "semantic_comments",
    "semantic_placeholder",
    "semantic_unsure",
    "semantic_valid",
]


def run_detection_algorithms(
    polluted_dataset: pd.DataFrame,
    detectors: List[DETECTORS_LITERAL],
    column_types: Dict[str, COLUMN_TYPES],
    target_columns: List[str],
    example_DMVs: Dict[str, Dict[EXAMPLE_DMV_CATEGORIES, DMV]],
    embeddings: Dict[str, pd.DataFrame],
    LLM=None,
):
    """
    Run detection algorithms on the polluted dataset in parallel.

    Args:
        polluted_dataset (pd.DataFrame): The dataset with introduced errors.
        clean_dataset (pd.DataFrame): The dataset without introduced errors.
        detectors (List[DETECTORS_LITERAL]): List of detector names to run.
        types (Dict[str, str]): Dictionary mapping column names to their types. Expects either "numeric" or "categorical" as of now.
        target_columns (List[str]): List of columns to target for detection.
        LLM: Optional LLM instance for semantic detector.

    Returns:
        dict: A dictionary containing the results of each detector.
    """

    detector_mapper: Dict[DETECTORS_LITERAL, Callable[[Any], DMVDetector]] = {
        "frequent_values_1": lambda data: FrequentValuesDetector(
            num_neighbors=1, target_types=["numeric", "date"]
        ),
        "frequent_values_3": lambda data: FrequentValuesDetector(
            num_neighbors=3, target_types=["numeric", "date"]
        ),
        "frequent_values_10": lambda data: FrequentValuesDetector(
            num_neighbors=10, target_types=["numeric", "date"]
        ),
        "frequent_values_25": lambda data: FrequentValuesDetector(
            num_neighbors=25, target_types=["numeric", "date"]
        ),
        "bucket_knn_square": lambda data: BucketKNNSquare(
            k=10, target_types=["numeric", "date"]
        ),
        "distribution": lambda data: DistributionFitDetector(
            target_types=["numeric", "date"]
        ),
        "BucketPDFGoF": lambda data: BucketPDFGoF(target_types=["numeric", "date"]),
        "pyod_knn": lambda data: PyODDetector("KNN", target_types=["numeric", "date"]),
        "pyod_lof": lambda data: PyODDetector("LOF", target_types=["numeric", "date"]),
        "pyod_iforest": lambda data: PyODDetector(
            "IForest", target_types=["numeric", "date"]
        ),
        "pyod_loda": lambda data: PyODDetector(
            "LODA", target_types=["numeric", "date"]
        ),
        "pyod_hbos": lambda data: PyODDetector(
            "HBOS", target_types=["numeric", "text", "date"]
        ),
        "pyod_cblof": lambda data: PyODDetector(
            "CBLOF", target_types=["numeric", "date"]
        ),
        "pyod_lof": lambda data: PyODDetector("LOF", target_types=["numeric", "date"]),
        "pyod_cof": lambda data: PyODDetector("COF", target_types=["numeric", "date"]),
        "pyod_gmm": lambda data: PyODDetector("GMM", target_types=["numeric", "date"]),
        "pyod_mad": lambda data: PyODDetector("MAD", target_types=["numeric", "date"]),
        "RobustZ": lambda data: PyODDetector(
            "RobustZ", target_types=["numeric", "date"]
        ),
        "Quantile": lambda data: PyODDetector(
            "Quantile", target_types=["numeric", "date"]
        ),
        "ESD": lambda data: PyODDetector("ESD", target_types=["numeric", "date"]),
        "pyod_mad2": lambda data: PyODDetector2(
            "MAD", target_types=["numeric", "date"]
        ),
        "pyod_mad3": lambda data: PyODDetector2(
            "MAD", target_types=["numeric", "date"], nanvalue=1.0
        ),
        "frequent_values_1_2": lambda data: FrequentValuesDetector2(
            num_neighbors=1, target_types=["numeric", "date"]
        ),
        "frequent_values_1_3": lambda data: FrequentValuesDetector2(
            num_neighbors=1, target_types=["numeric", "date"], nanvalue=1.0
        ),
        "type_feature_2": lambda data: TypeOutlierDetector2(
            target_types=["numeric", "text", "categorical", "date"]
        ),
        "frequency_outlier": lambda data: FrequencyOutlierDetector(
            target_types=["categorical", "text"]
        ),
        "length_outlier_feature": lambda data: LengthOutlierDetectorFeat(
            target_types=["numeric", "text", "categorical", "date"]
        ),
        "length_outlier_distribution": lambda data: LengthOutlierDetectorDist(
            target_types=["numeric", "categorical", "date"]
        ),
        "length_outlier_MAD": lambda data: LengthOutlierDetectorOD(
            "MAD", target_types=["numeric", "text", "categorical", "date"]
        ),
        "length_outlier_hbos": lambda data: LengthOutlierDetectorOD(
            "HBOS", target_types=["numeric", "text", "categorical", "date"]
        ),
        "repeated_substring1_outlier_feature": lambda data: RepeatedSubstringOutlierDetectorFeat(
            substring_length=1, target_types=["numeric", "text", "categorical", "date"]
        ),
        "repeated_substring1_outlier_distribution": lambda data: RepeatedSubstringOutlierDetectorDist(
            substring_length=1, target_types=["numeric", "text", "categorical", "date"]
        ),
        "repeated_substring1_outlier_MAD": lambda data: RepeatedSubstringOutlierDetectorOD(
            "MAD",
            substring_length=1,
            target_types=["numeric", "text", "categorical", "date"],
        ),
        "repeated_substring1_outlier_hbos": lambda data: RepeatedSubstringOutlierDetectorOD(
            "HBOS",
            substring_length=1,
            target_types=["numeric", "text", "categorical", "date"],
        ),
        "repeated_substring2_outlier_feature": lambda data: RepeatedSubstringOutlierDetectorFeat(
            substring_length=2, target_types=["numeric", "text", "categorical", "date"]
        ),
        "repeated_substring2_outlier_distribution": lambda data: RepeatedSubstringOutlierDetectorDist(
            substring_length=2, target_types=["numeric", "text", "categorical", "date"]
        ),
        "repeated_substring2_outlier_MAD": lambda data: RepeatedSubstringOutlierDetectorOD(
            "MAD",
            substring_length=2,
            target_types=["numeric", "text", "categorical", "date"],
        ),
        "repeated_substring2_outlier_hbos": lambda data: RepeatedSubstringOutlierDetectorOD(
            "HBOS",
            substring_length=2,
            target_types=["numeric", "text", "categorical", "date"],
        ),
        "repeated_substring3_outlier_feature": lambda data: RepeatedSubstringOutlierDetectorFeat(
            substring_length=3, target_types=["numeric", "text", "categorical", "date"]
        ),
        "repeated_substring3_outlier_distribution": lambda data: RepeatedSubstringOutlierDetectorDist(
            substring_length=3, target_types=["numeric", "text", "categorical", "date"]
        ),
        "repeated_substring3_outlier_MAD": lambda data: RepeatedSubstringOutlierDetectorOD(
            "MAD",
            substring_length=3,
            target_types=["numeric", "text", "categorical", "date"],
        ),
        "repeated_substring3_outlier_hbos": lambda data: RepeatedSubstringOutlierDetectorOD(
            "HBOS",
            substring_length=3,
            target_types=["numeric", "text", "categorical", "date"],
        ),
        "key_distance_outlier_feature": lambda data: KeyDistanceOutlierDetectorFeat(
            target_types=["numeric", "text", "categorical", "date"]
        ),
        "key_distance_outlier_distribution": lambda data: KeyDistanceOutlierDetectorDist(
            target_types=(
                ["categorical", "date"]
                if FAST
                else ["numeric", "text", "categorical", "date"]
            )
        ),  # ["numeric", "text", "categorical", "date"]
        "key_distance_outlier_MAD": lambda data: KeyDistanceOutlierDetectorOD(
            "MAD", target_types=["numeric", "text", "categorical", "date"]
        ),
        "key_distance_outlier_hbos": lambda data: KeyDistanceOutlierDetectorOD(
            "HBOS", target_types=["numeric", "text", "categorical", "date"]
        ),
        "capital_letter_outlier_distribution": lambda data: CapitalLetterOutlierDetectorDist(
            target_types=(
                ["numeric", "date"]
                if FAST
                else ["numeric", "text", "categorical", "date"]
            )
        ),
        "non_alphanumerical_outlier_distribution": lambda data: NonAlphanumericalOutlierDetectorDist(
            target_types=(
                ["numeric"] if FAST else ["numeric", "text", "categorical", "date"]
            )
        ),
        "sign_outlier_feature": lambda data: SignOutlierDetectorFeat(
            target_types=["numeric"]
        ),
        "semantic_outlier_10": lambda data: SemanticOutlierDetector(
            LLM, target_types=["text", "categorical"], num_neighbors=10
        ),
        "semantic_outlier_25": lambda data: SemanticOutlierDetector(
            LLM, target_types=["text", "categorical"], num_neighbors=25
        ),
        "semantic_outlier_100": lambda data: SemanticOutlierDetector(
            LLM, target_types=["text", "categorical"], num_neighbors=100
        ),
        "semantic_outlier_3_new": lambda data: SemanticOutlierDetectorNew(
            LLM, target_types=["text", "categorical"], num_neighbors=3
        ),
        "semantic_outlier_10_new": lambda data: SemanticOutlierDetectorNew(
            LLM, target_types=["text", "categorical"], num_neighbors=10
        ),
        "semantic_outlier_25_new": lambda data: SemanticOutlierDetectorNew(
            LLM, target_types=["text", "categorical"], num_neighbors=25
        ),
        "semantic_outlier_100_new": lambda data: SemanticOutlierDetectorNew(
            LLM, target_types=["text", "categorical"], num_neighbors=100
        ),
        "multi_semantic_outlier_new": lambda data: MultiSemanticOutlierDetectorNew(
            LLM,
            target_types=["categorical"] if FAST else ["text", "categorical"],
            num_neighbors_list=[3, 10, 25, 100],
        ),
        "multi_semantic_outlier_new_dub": lambda data: MultiSemanticOutlierDetectorNew(
            LLM,
            target_types=["categorical"] if FAST else ["text", "categorical"],
            num_neighbors_list=[3, 10, 25, 100],
            remove_duplicates=True,
        ),
        "semantic_outlier_3_new_dub": lambda data: SemanticOutlierDetectorNewDub(
            LLM,
            target_types=["categorical"] if FAST else ["text", "categorical"],
            num_neighbors=3,
        ),
        "semantic_outlier_10_new_dub": lambda data: SemanticOutlierDetectorNewDub(
            LLM,
            target_types=["categorical"] if FAST else ["text", "categorical"],
            num_neighbors=10,
        ),
        "semantic_outlier_25_new_dub": lambda data: SemanticOutlierDetectorNewDub(
            LLM,
            target_types=["categorical"] if FAST else ["text", "categorical"],
            num_neighbors=25,
        ),
        "semantic_outlier_100_new_dub": lambda data: SemanticOutlierDetectorNewDub(
            LLM,
            target_types=["categorical"] if FAST else ["text", "categorical"],
            num_neighbors=100,
        ),
        "type_feature": lambda data: TypeOutlierDetector(
            target_types=["numeric", "text", "categorical", "date"]
        ),
        "nan_outlier": lambda data: NanDetector(
            target_types=["numeric", "text", "categorical", "date"]
        ),
        "quantile": lambda data: QuantileDetector(target_types=["numeric", "date"]),
        "approximate_similar_samples_5": lambda data: FAISSSimilarSamplesDetector(
            num_neighbors=5, target_types=["numeric", "text", "categorical", "date"]
        ),
        "approximate_similar_samples_corr_5": lambda data: FAISSSimilarSamplesDetector(
            num_neighbors=5,
            use_correlations=True,
            target_types=["numeric", "text", "categorical", "date"],
        ),
        "approximate_similar_samples_10": lambda data: FAISSSimilarSamplesDetector(
            num_neighbors=10, target_types=["numeric", "text", "categorical", "date"]
        ),
        "approximate_similar_samples_corr_10": lambda data: FAISSSimilarSamplesDetector(
            num_neighbors=10,
            use_correlations=True,
            target_types=["numeric", "text", "categorical", "date"],
        ),
        "approximate_similar_samples_25": lambda data: FAISSSimilarSamplesDetector(
            num_neighbors=25, target_types=["numeric", "text", "categorical", "date"]
        ),
        "approximate_similar_samples_corr_25": lambda data: FAISSSimilarSamplesDetector(
            num_neighbors=25,
            use_correlations=True,
            target_types=["numeric", "text", "categorical", "date"],
        ),
        "no_duplicate_similar_samples_5": lambda data: FAISSNoDubSimilarSamplesDetector(
            num_neighbors=5, target_types=["numeric", "text", "categorical", "date"]
        ),
        "no_duplicate_similar_samples_10": lambda data: FAISSNoDubSimilarSamplesDetector(
            num_neighbors=10, target_types=["numeric", "text", "categorical", "date"]
        ),
        "no_duplicate_similar_samples_25": lambda data: FAISSNoDubSimilarSamplesDetector(
            num_neighbors=25, target_types=["numeric", "text", "categorical", "date"]
        ),
        "multi_similar_samples": lambda data: MultiSimilarSamplesDetector(
            num_neighbors_list=[25],
            include_correlations=True,
            include_no_duplicates=True,
            target_types=(
                ["numeric", "date"]
                if FAST
                else ["numeric", "text", "categorical", "date"]
            ),
        ),
        "syntactic_outlier": lambda data: SyntacticDetector(
            target_types=["numeric", "categorical", "date"]
        ),
        "llm_classifier": lambda data: LLMClassifierDetector(
            LLM, target_types=["numeric", "categorical", "date", "text"]
        ),
    }

    for dmv_type in list(next(iter(example_DMVs.values()), {}).keys()):
        targets = {col: value[dmv_type] for col, value in example_DMVs.items()}
        # Bind `targets` and `dmv_type` as default arguments to avoid late binding in the lambda
        detector_mapper[f"semantic_{dmv_type}"] = (  # type: ignore
            lambda t=targets, dt=dmv_type: (
                lambda data: SemanticDetector(
                    LLM,
                    targets=t,
                    target_types=["text", "categorical"],
                    invert=dmv_type == "valid",
                )
            )
        )()

    results = {}
    assessed_columns_per_detector = {}
    all_times = {}

    import os

    import psutil

    process = psutil.Process(os.getpid())

    for name in detectors:
        if name not in detector_mapper:
            print(f"Detector {name} not recognized. Skipping.")
            continue
        detector_instance = detector_mapper[name](polluted_dataset)
        detector_result = detector_instance(
            polluted_dataset, column_types, target_columns, embeddings
        )

        # Check if detector returns multiple results (dict) or single result (tuple)
        if isinstance(detector_result, dict):
            # Multi-result detector
            print(f"Detector {name} returned {len(detector_result)} results")
            for sub_name, sub_result in detector_result.items():
                results[sub_name] = sub_result[:2]  # (df_score, df_predict)
                all_times[sub_name] = sub_result[2]  # times
                assessed_columns_per_detector[sub_name] = sub_result[3]  # assessed
        else:
            # Single-result detector
            results[name] = detector_result[:2]
            all_times[name] = detector_result[2]
            assessed_columns_per_detector[name] = detector_result[3]

        print(
            f"[MEMORY] Current memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB"
        )

    return results, all_times, assessed_columns_per_detector
