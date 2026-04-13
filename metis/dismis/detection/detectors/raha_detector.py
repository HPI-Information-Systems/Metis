import random
import string
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from raha import Dataset, Detection

from metis.dismis.detection.detectors.detector import DMVDetector


class MemoryDataset(Dataset):
    """
    The memory dataset class.
    """

    def __init__(self, dataset_dictionary):
        """
        The constructor creates a memory dataset.
        """
        self.name = dataset_dictionary["name"]
        self.dataframe = (
            dataset_dictionary["dirty"].copy().astype(str).map(self.value_normalizer)
        )
        self.has_ground_truth = True
        self.clean_dataframe = (
            dataset_dictionary["clean"].copy().astype(str).map(self.value_normalizer)
        )


class MemoryDetection(Detection):
    def __init__(self):
        """
        The constructor.
        """
        self.LABELING_BUDGET = 20
        self.USER_LABELING_ACCURACY = 1.0
        self.VERBOSE = False
        self.SAVE_RESULTS = False
        self.CLUSTERING_BASED_SAMPLING = True
        self.STRATEGY_FILTERING = False
        self.CLASSIFICATION_MODEL = "GBC"  # ["ABC", "DTC", "GBC", "GNB", "SGDC", "SVC"]
        self.LABEL_PROPAGATION_METHOD = "homogeneity"  # ["homogeneity", "majority"]
        self.ERROR_DETECTION_ALGORITHMS = [
            "OD",
            "PVD",
            "RVD",
            "KBVD",
        ]  # ["OD", "PVD", "RVD", "KBVD", "TFIDF"]
        self.HISTORICAL_DATASETS = []

    def initialize_dataset(self, dd):
        """
        This method initializes the dataset.
        """
        d = MemoryDataset(dd)
        # No paths will be set here since everything is in memory
        d.results_folder = "/sc/home/philipp.hildebrandt/DMV/ChameleonDetect/results"
        d.labeled_tuples = {} if not hasattr(d, "labeled_tuples") else d.labeled_tuples
        d.labeled_cells = {} if not hasattr(d, "labeled_cells") else d.labeled_cells
        d.labels_per_cluster = (
            {} if not hasattr(d, "labels_per_cluster") else d.labels_per_cluster
        )
        d.detected_cells = {} if not hasattr(d, "detected_cells") else d.detected_cells
        return d


class RAHADetector(DMVDetector):
    def __init__(self, clean_dataset: pd.DataFrame):
        """
        Initialize the SimilarSamplesDetector with a specific detector.

        Args:
            clean_dataset (pd.DataFrame): The clean dataset to use for detection.
        """
        self.app = MemoryDetection()
        self.clean_dataset = clean_dataset

    def __call__(
        self,
        dataset: pd.DataFrame,
        types: Dict[str, str],
        target_columns: List[str] = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:

        times = {}
        total_starttime = time.time()

        df_detect, df_predict = dataset.copy(), dataset.copy()
        df_predict.loc[:, :] = 0
        df_predict = df_predict.astype(int)

        random_suffix = "".join(
            random.choices(string.ascii_letters + string.digits, k=16)
        )
        dataset_dictionary = {
            "name": "Some_DisMisDataset_" + random_suffix,
            "dirty": df_detect,
            "clean": self.clean_dataset,
        }

        if len(df_detect.columns) != len(self.clean_dataset.columns):
            print(df_detect.columns)
            print(self.clean_dataset.columns)
            raise ValueError(
                "The number of columns in the dirty dataset does not match the number of columns in the clean dataset."
            )

        raha_starttime = time.time()
        detection_dictionary = self.app.run(dataset_dictionary)
        times["raha"] = time.time() - raha_starttime

        score_save_starttime = time.time()
        if len(detection_dictionary) > 0:
            positions = np.array(list(detection_dictionary.keys()))
            rows, cols = positions[:, 0], positions[:, 1]
            array = df_predict.to_numpy()
            array[rows, cols] = 1
            df_predict.iloc[:, :] = array
        df_score = df_predict.copy().astype(float)

        times["score_save"] = time.time() - score_save_starttime

        times["total"] = time.time() - total_starttime

        return df_score, df_predict.astype(int), times, df_detect.columns.tolist()
