import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from metis.dismis.detection.detectors.detector import DMVDetector
from metis.dismis.utils.types import COLUMN_TYPES


class FAHESDetector(DMVDetector):
    def __init__(
        self,
        fahes_executable: str,
    ):
        """
        Initialize the SimilarSamplesDetector with a specific detector.

        Args:
            clean_dataset (pd.DataFrame): The clean dataset to use for detection.
        """
        self.executable = fahes_executable

    def __call__(
        self,
        dataset: pd.DataFrame,
        column_types: Dict[str, COLUMN_TYPES],
        target_columns: List[str] | None = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:
        times: Dict[str, float] = {}
        total_starttime = time.time()

        df_detect = dataset.copy()
        df_predict = pd.DataFrame(0, index=dataset.index, columns=dataset.columns)
        assessed = []

        with tempfile.TemporaryDirectory() as temp_dir:
            input_csv = os.path.join(temp_dir, "input.csv")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # Save input DataFrame
            df_detect.to_csv(input_csv, index=False)

            # Run FAHES
            cmd = [self.executable, input_csv, output_dir, "4"]
            fahes_starttime = time.time()

            try:
                subprocess.run(
                    cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"FAHES failed: {e.stderr.decode()}")
            times["fahes"] = time.time() - fahes_starttime

            # Look for result file
            result_files = list(Path(output_dir).glob("*.csv"))

            # Load FAHES output
            try:
                result_df = pd.read_csv(
                    result_files[0], keep_default_na=False, na_values=[""]
                )
                result_df.columns = [
                    "Table Name",
                    "Column Name",
                    "DMV",
                    "Frequency",
                    "Tool ID",
                ]

                # Normalize DMV values to string (to match df values)
                dmvs_by_column = (
                    result_df.groupby("Column Name")["DMV"].apply(set).to_dict()
                )

                for col in df_detect.columns:
                    if col in dmvs_by_column:
                        df_predict[col] = (
                            df_detect[col]
                            .astype(str)
                            .isin(dmvs_by_column[col])
                            .astype(int)
                        )
                    assessed.append(col)
            except:
                print("No DMVs found or when processing the FAHES result file.")

        df_score = df_predict.copy().astype(float)
        times["total"] = time.time() - total_starttime

        return df_score, df_predict, times, assessed
