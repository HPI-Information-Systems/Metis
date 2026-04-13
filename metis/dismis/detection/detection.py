import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

from metis.dismis.detection.detectors.frequent_value import FrequentValuesDetector, FrequentValuesDetector2
from metis.dismis.detection.detectors.bucketknn_square import BucketKNN as BucketKNNSquare
from metis.dismis.detection.detectors.distribution import DistributionFitDetector, BucketPDFGoF
from metis.dismis.detection.detectors.pyod import (
    PyODDetector,
    PyODDetector2,
    LengthOutlierDetectorFeat,
    LengthOutlierDetectorDist,
    LengthOutlierDetectorOD,
    RepeatedSubstringOutlierDetectorFeat,
    RepeatedSubstringOutlierDetectorDist,
    RepeatedSubstringOutlierDetectorOD,
    KeyDistanceOutlierDetectorFeat,
    KeyDistanceOutlierDetectorDist,
    KeyDistanceOutlierDetectorOD,
    SemanticDetector,
    SemanticOutlierDetector,
    SemanticOutlierDetectorNew,
    SemanticOutlierDetectorNewDub,
    MultiSemanticOutlierDetectorNew,
    TypeOutlierDetector,
    TypeOutlierDetector2,
    FrequencyOutlierDetector,
    CapitalLetterOutlierDetectorDist,
    NonAlphanumericalOutlierDetectorDist,
    SignOutlierDetectorFeat
)
from metis.dismis.detection.detectors.dummy import NanDetector, QuantileDetector
from metis.dismis.detection.detectors.similar_samples import FAISSSimilarSamplesDetector, FAISSNoDubSimilarSamplesDetector, MultiSimilarSamplesDetector
from metis.dismis.detection.detectors.raha_detector import RAHADetector
from metis.dismis.detection.detectors.fahes_detector import FAHESDetector
from metis.dismis.detection.detectors.syntactic import SyntacticDetector
from metis.dismis.detection.detectors.llm_classifier import LLMClassifierDetector

FAST = False

def run_detector(detector_name, polluted_dataset, detector_instance, types, target_columns, embeddings):
    print(f"Running detector: {detector_name}")
    scores, predictions, times, assessed_columns = detector_instance(polluted_dataset, types, target_columns, embeddings)
    return detector_name, (scores, predictions), times, assessed_columns

def run_detection_algorithms(polluted_dataset: pd.DataFrame, clean_dataset: pd.DataFrame, detectors: List[str], types: Dict[str, str], target_columns: List[str], example_DMVs: Dict, embeddings: Dict[str, np.ndarray], LLM=None) -> Tuple[Dict, Dict]:
    """
    Run detection algorithms on the polluted dataset in parallel.

    Args:
        polluted_dataset (pd.DataFrame): The dataset with introduced errors.
        clean_dataset (pd.DataFrame): The dataset without introduced errors.
        detectors (List[str]): List of detector names to run.
        types (Dict[str, str]): Dictionary mapping column names to their types. Expects either "numeric" or "categorical" as of now.
        target_columns (List[str]): List of columns to target for detection.
        LLM: Optional LLM instance for semantic detector.

    Returns:
        dict: A dictionary containing the results of each detector.
    """

    example_DMV_types = []
    if len(example_DMVs) > 0:
        example_DMV_types = list(example_DMVs[list(example_DMVs.keys())[0]].keys())

    detector_mapper = {
        "frequent_values_1": lambda data: FrequentValuesDetector(num_neighbors=1, target_types=["numeric", "date"]), #checked #checked
        "frequent_values_3": lambda data: FrequentValuesDetector(num_neighbors=3, target_types=["numeric", "date"]),  #checked #checked
        "frequent_values_10": lambda data: FrequentValuesDetector(num_neighbors=10, target_types=["numeric", "date"]),  #checked #checked
        "frequent_values_25": lambda data: FrequentValuesDetector(num_neighbors=25, target_types=["numeric", "date"]),  #checked #checked
        "bucket_knn_square": lambda data: BucketKNNSquare(k=10, target_types=["numeric", "date"]), #checked #checked
        "distribution": lambda data: DistributionFitDetector(target_types=["numeric", "date"]), #checked #checked
        "BucketPDFGoF": lambda data: BucketPDFGoF(target_types=["numeric", "date"]),  #checked #checked

        "pyod_knn": lambda data: PyODDetector("KNN", target_types=["numeric", "date"]), #checked #checked
        "pyod_lof": lambda data: PyODDetector("LOF", target_types=["numeric", "date"]), #checked #checked
        "pyod_iforest": lambda data: PyODDetector("IForest", target_types=["numeric", "date"]), #checked #checked
        "pyod_loda": lambda data: PyODDetector("LODA", target_types=["numeric", "date"]), #checked #checked
        "pyod_hbos": lambda data: PyODDetector("HBOS", target_types=["numeric", "text", "date"]), #checked #checked
        "pyod_cblof": lambda data: PyODDetector("CBLOF", target_types=["numeric", "date"]), #checked #checked
        "pyod_lof": lambda data: PyODDetector("LOF", target_types=["numeric", "date"]), #checked #checked
        "pyod_cof": lambda data: PyODDetector("COF", target_types=["numeric", "date"]), #checked #checked
        "pyod_gmm": lambda data: PyODDetector("GMM", target_types=["numeric", "date"]), #checked #checked
        "pyod_mad": lambda data: PyODDetector("MAD", target_types=["numeric", "date"]), #checked #checked
        "RobustZ": lambda data: PyODDetector("RobustZ", target_types=["numeric", "date"]), #checked #checked
        "Quantile": lambda data: PyODDetector("Quantile", target_types=["numeric", "date"]), #checked #checked
        "ESD": lambda data: PyODDetector("ESD", target_types=["numeric", "date"]), #checked #checked

        "pyod_mad2": lambda data: PyODDetector2("MAD", target_types=["numeric", "date"]), #checked #checked
        "pyod_mad3": lambda data: PyODDetector2("MAD", target_types=["numeric", "date"], nanvalue=1.0), #checked #checked
        "frequent_values_1_2": lambda data: FrequentValuesDetector2(num_neighbors=1, target_types=["numeric", "date"]), #checked #checked
        "frequent_values_1_3": lambda data: FrequentValuesDetector2(num_neighbors=1, target_types=["numeric", "date"], nanvalue=1.0), #checked #checked
        "type_feature_2": lambda data: TypeOutlierDetector2(target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "frequency_outlier": lambda data: FrequencyOutlierDetector(target_types=["categorical", "text"]), #checked #checked

        "length_outlier_feature": lambda data: LengthOutlierDetectorFeat(target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "length_outlier_distribution": lambda data: LengthOutlierDetectorDist(target_types=["numeric", "categorical", "date"]), #checked #checked
        "length_outlier_MAD": lambda data: LengthOutlierDetectorOD("MAD", target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "length_outlier_hbos": lambda data: LengthOutlierDetectorOD("HBOS", target_types=["numeric", "text", "categorical", "date"]), #checked #checked

        "repeated_substring1_outlier_feature": lambda data: RepeatedSubstringOutlierDetectorFeat(substring_length=1, target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "repeated_substring1_outlier_distribution": lambda data: RepeatedSubstringOutlierDetectorDist(substring_length=1, target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "repeated_substring1_outlier_MAD": lambda data: RepeatedSubstringOutlierDetectorOD("MAD", substring_length=1, target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "repeated_substring1_outlier_hbos": lambda data: RepeatedSubstringOutlierDetectorOD("HBOS", substring_length=1, target_types=["numeric", "text", "categorical", "date"]), #checked #checked

        "repeated_substring2_outlier_feature": lambda data: RepeatedSubstringOutlierDetectorFeat(substring_length=2, target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "repeated_substring2_outlier_distribution": lambda data: RepeatedSubstringOutlierDetectorDist(substring_length=2, target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "repeated_substring2_outlier_MAD": lambda data: RepeatedSubstringOutlierDetectorOD("MAD", substring_length=2, target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "repeated_substring2_outlier_hbos": lambda data: RepeatedSubstringOutlierDetectorOD("HBOS", substring_length=2, target_types=["numeric", "text", "categorical", "date"]), #checked #checked

        "repeated_substring3_outlier_feature": lambda data: RepeatedSubstringOutlierDetectorFeat(substring_length=3, target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "repeated_substring3_outlier_distribution": lambda data: RepeatedSubstringOutlierDetectorDist(substring_length=3, target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "repeated_substring3_outlier_MAD": lambda data: RepeatedSubstringOutlierDetectorOD("MAD", substring_length=3, target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "repeated_substring3_outlier_hbos": lambda data: RepeatedSubstringOutlierDetectorOD("HBOS", substring_length=3, target_types=["numeric", "text", "categorical", "date"]), #checked #checked

        "key_distance_outlier_feature": lambda data: KeyDistanceOutlierDetectorFeat(target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "key_distance_outlier_distribution": lambda data: KeyDistanceOutlierDetectorDist(target_types=["categorical", "date"] if FAST else ["numeric", "text", "categorical", "date"]), #["numeric", "text", "categorical", "date"]
        "key_distance_outlier_MAD": lambda data: KeyDistanceOutlierDetectorOD("MAD", target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "key_distance_outlier_hbos": lambda data: KeyDistanceOutlierDetectorOD("HBOS", target_types=["numeric", "text", "categorical", "date"]), #checked #checked

        "capital_letter_outlier_distribution": lambda data: CapitalLetterOutlierDetectorDist(target_types=["numeric", "date"] if FAST else ["numeric", "text", "categorical", "date"]), #checked #checked
        "non_alphanumerical_outlier_distribution": lambda data: NonAlphanumericalOutlierDetectorDist(target_types=["numeric"] if FAST else ["numeric", "text", "categorical", "date"]), #checked #checked
        "sign_outlier_feature": lambda data: SignOutlierDetectorFeat(target_types=["numeric"]), #checked #checked

        "semantic_outlier_10": lambda data: SemanticOutlierDetector(LLM, target_types=["text", "categorical"], num_neighbors=10), #checked #checked
        "semantic_outlier_25": lambda data: SemanticOutlierDetector(LLM, target_types=["text", "categorical"], num_neighbors=25), #checked #checked
        "semantic_outlier_100": lambda data: SemanticOutlierDetector(LLM, target_types=["text", "categorical"], num_neighbors=100), #checked #checked

        "semantic_outlier_3_new": lambda data: SemanticOutlierDetectorNew(LLM, target_types=["text", "categorical"], num_neighbors=3), #checked #checked
        "semantic_outlier_10_new": lambda data: SemanticOutlierDetectorNew(LLM, target_types=["text", "categorical"], num_neighbors=10), #checked #checked
        "semantic_outlier_25_new": lambda data: SemanticOutlierDetectorNew(LLM, target_types=["text", "categorical"], num_neighbors=25), #checked #checked
        "semantic_outlier_100_new": lambda data: SemanticOutlierDetectorNew(LLM, target_types=["text", "categorical"], num_neighbors=100), #checked #checked

        "multi_semantic_outlier_new": lambda data: MultiSemanticOutlierDetectorNew(LLM, target_types=["categorical"] if FAST else ["text", "categorical"], num_neighbors_list=[3, 10, 25, 100]), #checked #checked
        "multi_semantic_outlier_new_dub": lambda data: MultiSemanticOutlierDetectorNew(LLM, target_types=[ "categorical"] if FAST else ["text", "categorical"], num_neighbors_list=[3, 10, 25, 100], remove_duplicates=True), #checked #checked

        "semantic_outlier_3_new_dub": lambda data: SemanticOutlierDetectorNewDub(LLM, target_types=["categorical"] if FAST else ["text", "categorical"], num_neighbors=3), #checked #checked
        "semantic_outlier_10_new_dub": lambda data: SemanticOutlierDetectorNewDub(LLM, target_types=["categorical"] if FAST else ["text", "categorical"], num_neighbors=10), #checked #checked
        "semantic_outlier_25_new_dub": lambda data: SemanticOutlierDetectorNewDub(LLM, target_types=["categorical"] if FAST else ["text", "categorical"], num_neighbors=25), #checked #checked
        "semantic_outlier_100_new_dub": lambda data: SemanticOutlierDetectorNewDub(LLM, target_types=["categorical"] if FAST else ["text", "categorical"], num_neighbors=100), #checked #checked

        "type_feature": lambda data: TypeOutlierDetector(target_types=["numeric", "text", "categorical", "date"]), #checked #checked

        "nan_outlier": lambda data: NanDetector(target_types=["numeric", "text", "categorical", "date"]), #checked #checked
        "quantile": lambda data: QuantileDetector(target_types=["numeric", "date"]), #checked #checked

        "approximate_similar_samples_5": lambda data: FAISSSimilarSamplesDetector(num_neighbors=5, target_types=["numeric", "text", "categorical", "date"]),#checked #checked
        "approximate_similar_samples_corr_5": lambda data: FAISSSimilarSamplesDetector(num_neighbors=5, use_correlations=True, target_types=["numeric", "text", "categorical", "date"]),#checked #checked

        "approximate_similar_samples_10": lambda data: FAISSSimilarSamplesDetector(num_neighbors=10, target_types=["numeric", "text", "categorical", "date"]),#checked #checked
        "approximate_similar_samples_corr_10": lambda data: FAISSSimilarSamplesDetector(num_neighbors=10, use_correlations=True, target_types=["numeric", "text", "categorical", "date"]),#checked #checked

        "approximate_similar_samples_25": lambda data: FAISSSimilarSamplesDetector(num_neighbors=25, target_types=["numeric", "text", "categorical", "date"]),#checked #checked
        "approximate_similar_samples_corr_25": lambda data: FAISSSimilarSamplesDetector(num_neighbors=25, use_correlations=True, target_types=["numeric", "text", "categorical", "date"]),#checked #checked

        "no_duplicate_similar_samples_5": lambda data: FAISSNoDubSimilarSamplesDetector(num_neighbors=5, target_types=["numeric", "text", "categorical", "date"]),#checked #checked
        "no_duplicate_similar_samples_10": lambda data: FAISSNoDubSimilarSamplesDetector(num_neighbors=10, target_types=["numeric", "text", "categorical", "date"]),#checked #checked
        "no_duplicate_similar_samples_25": lambda data: FAISSNoDubSimilarSamplesDetector(num_neighbors=25, target_types=["numeric", "text", "categorical", "date"]),#checked #checked

        "multi_similar_samples": lambda data: MultiSimilarSamplesDetector(num_neighbors_list=[25], include_correlations=True, include_no_duplicates=True, target_types=["numeric", "date"] if FAST else ["numeric", "text", "categorical", "date"]), #["numeric", "text", "categorical", "date"]

        "syntactic_outlier": lambda data: SyntacticDetector(target_types=["numeric", "categorical", "date"]), #["numeric", "categorical", "date", "text"]

        "llm_classifier": lambda data: LLMClassifierDetector(LLM, target_types=["numeric", "categorical", "date", "text"]),

        "RAHA": lambda data: RAHADetector(clean_dataset),
        "FAHES": lambda data: FAHESDetector(),
    }

    for dmv_type in example_DMV_types:
        targets = {col: value[dmv_type] for col, value in example_DMVs.items()}
        # Bind `targets` and `dmv_type` as default arguments to avoid late binding in the lambda
        detector_mapper[f"semantic_{dmv_type}"] = (lambda t=targets, dt=dmv_type: (lambda data: SemanticDetector(LLM, targets=t, target_types=["text", "categorical"], invert=dmv_type == "valid")))()

    results = {}
    assessed_columns_per_detector = {}
    all_times = {}

    import psutil, os
    process = psutil.Process(os.getpid())


    for name in detectors:
        if name not in detector_mapper:
            print(f"Detector {name} not recognized. Skipping.")
            continue
        detector_instance = detector_mapper[name](polluted_dataset)
        detector_result = detector_instance(polluted_dataset, types, target_columns, embeddings)

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

        print(f"[MEMORY] Current memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    return results, all_times, assessed_columns_per_detector
