import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from metis.dismis.detection.detectors.detector import DMVDetector
from metis.dismis.detection.detectors.utils import force_numeric
from metis.dismis.preparation.openai_LLM import OpenAILLM


class LLMClassifierDetector(DMVDetector):
    """
    Detector that uses an LLM to classify each value in a column as either a DMV (disguised missing value)
    or a valid value. Returns binary predictions (1 for DMV, 0 for valid).
    """

    def __init__(
        self,
        LLM,
        target_types: List[str] = ["numeric", "text", "categorical", "date"],
        batch_size: int = 25,
    ):
        """
        Initialize the LLMClassifierDetector.

        Args:
            LLM: Language model instance that implements the `generate` method.
            target_types (List[str]): List of column types to target for detection.
            batch_size (int): Number of values to classify in each batch.
        """
        self.LLM = OpenAILLM(model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct")
        self.target_types = target_types
        self.batch_size = batch_size

    def _create_classification_prompt_text(
        self, column_name: str, values: List[str], frequencies: List[int]
    ) -> List[Dict[str, str]]:
        """
        Create a prompt for the LLM to classify values as DMV or valid.

        Args:
            column_name (str): Name of the column being analyzed.
            values (List[str]): Values to classify.
            sample_values (List[str]): Sample of likely valid values from the column for context.

        Returns:
            List[Dict[str, str]]: Chat messages for the LLM.
        """
        values_list = "\n".join(
            [
                f"{i+1}. {val} (Occurs {frequencies[i]} times)"
                for i, val in enumerate(values)
            ]
        )

        prompt = f"""You are a data quality analyst tasked with identifying Disguised Missing Values (DMVs) in a dataset.

DMVs are placeholder values that represent missing, unknown, or invalid data. Examples include:
- Generic placeholders: "N/A", "Unknown", "Missing", "None", "Placeholder", "?", "---", "TBD"
- Nonsense values: "asdf", "xxxx", "123test", random strings
- Comments: "See notes", "Check later", "Not sure"
- Unusual patterns: repeated characters, obvious outliers like "-999", "9999"

Column name: "{column_name}"

Now, classify each of the following values as either "DMV" (disguised missing value) or "VALID" (real data).
Consider the context of the column name.

Values to classify:
{values_list}

Respond with ONLY a comma-separated list of classifications in the same order as the values above.
Use only "DMV" or "VALID" for each value. Do not include any other text or explanations.

Example response format: VALID, DMV, VALID, VALID, DMV"""

        return [
            {
                "role": "system",
                "content": "You are a data quality analyst specializing in identifying disguised missing values in datasets.",
            },
            {"role": "user", "content": prompt},
        ]

    def _create_classification_prompt_numeric(
        self,
        column_name: str,
        values: List[str],
        frequencies: List[int],
        min_value: float,
        max_value: float,
        mean: float,
        median: float,
        p5: float,
        p95: float,
    ) -> List[Dict[str, str]]:
        """
        Create a prompt for the LLM to classify values as DMV or valid.

        Args:
            column_name (str): Name of the column being analyzed.
            values (List[str]): Values to classify.
            frequencies (List[int]): Frequency of each value in the column.

        Returns:
            List[Dict[str, str]]: Chat messages for the LLM.
        """
        values_list = "\n".join(
            [
                f"{i+1}. {val} (Occurs {frequencies[i]} times)"
                for i, val in enumerate(values)
            ]
        )
        prompt = f"""You are a data quality analyst tasked with identifying Disguised Missing Values (DMVs) in a dataset.

DMVs are placeholder values that represent missing, unknown, or invalid data. Examples include:
- Generic placeholders: "N/A", "Unknown", "Missing", "None", "Placeholder", "?", "---", "TBD"
- Nonsense values: "asdf", "xxxx", "123test", random strings
- Comments: "See notes", "Check later", "Not sure"
- Unusual patterns: repeated characters, obvious outliers like "-999", "9999"
- Numeric placeholders or outliers: "0", "-1", "9999", "-999", "12345"

Column name: "{column_name}"
Statistics for the column:
- Minimum value: {min_value}
- Maximum value: {max_value}
- Mean value: {mean}
- Median value: {median}
- 5th percentile: {p5}
- 95th percentile: {p95}

Now, classify each of the following values as either "DMV" (disguised missing value) or "VALID" (real data).
Consider the context of the column name and the numeric statistics provided.

Values to classify:
{values_list}

Respond with ONLY a comma-separated list of classifications in the same order as the values above.
Use only "DMV" or "VALID" for each value. Do not include any other text or explanations.

Example response format: VALID, DMV, VALID, VALID, DMV"""

        return [
            {
                "role": "system",
                "content": "You are a data quality analyst specializing in identifying disguised missing values in datasets.",
            },
            {"role": "user", "content": prompt},
        ]

    def _parse_llm_response(self, response: str, num_values: int) -> np.ndarray:
        """
        Parse the LLM response to extract binary predictions.

        Args:
            response (str): Raw response from the LLM.
            num_values (int): Expected number of classifications.

        Returns:
            np.ndarray: Binary array (1 for DMV, 0 for valid).
        """
        classifications = [
            c.strip() for c in response.strip().upper().split(",")[:num_values]
        ]

        predictions = np.array(
            [1 if "DMV" in c else 0 for c in classifications], dtype=np.int8
        )

        if len(predictions) < num_values:
            predictions = np.pad(
                predictions, (0, num_values - len(predictions)), constant_values=0
            )

        return predictions

    def _classify_column(
        self, column: pd.Series, column_name: str, type: str
    ) -> np.ndarray:
        """
        Classify all values in a column using the LLM.
        Only processes unique values to avoid redundancy, then maps results back to original positions.

        Args:
            column (pd.Series): Column to classify.
            column_name (str): Name of the column.

        Returns:
            np.ndarray: Binary predictions for each value (1 for DMV, 0 for valid).
        """
        all_values = column.astype(str)

        value_counts = all_values.value_counts()
        unique_values = value_counts.index.tolist()
        frequencies = value_counts.values.tolist()

        print(
            f"  Processing {len(unique_values)} unique values out of {len(all_values)} total values"
        )

        all_messages = []
        batch_info = []

        numeric_stats = None
        if type in ["numeric"]:
            numeric_col = force_numeric(column).dropna()
            numeric_stats = {
                "min_value": numeric_col.min(),
                "max_value": numeric_col.max(),
                "mean": numeric_col.mean(),
                "median": numeric_col.median(),
                "p5": numeric_col.quantile(0.05),
                "p95": numeric_col.quantile(0.95),
            }

        for start_idx in range(0, len(unique_values), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(unique_values))
            batch_values = unique_values[start_idx:end_idx]
            batch_frequencies = frequencies[start_idx:end_idx]

            if numeric_stats is not None:
                messages = self._create_classification_prompt_numeric(
                    column_name, batch_values, batch_frequencies, **numeric_stats
                )
            else:
                messages = self._create_classification_prompt_text(
                    column_name, batch_values, batch_frequencies
                )

            all_messages.append(messages)
            batch_info.append({"values": batch_values, "num_values": len(batch_values)})

        print(f"  Sending {len(all_messages)} batched requests to LLM")

        max_concurrent_requests = 512
        all_responses = []

        try:
            for i in range(0, len(all_messages), max_concurrent_requests):
                batch_end = min(i + max_concurrent_requests, len(all_messages))
                message_batch = all_messages[i:batch_end]

                print(
                    f"  Sending batch {i//max_concurrent_requests + 1}/{(len(all_messages) + max_concurrent_requests - 1)//max_concurrent_requests} ({len(message_batch)} requests)"
                )
                batch_responses = self.LLM.generate(message_batch)
                all_responses.extend(batch_responses)

            parsing_starttime = time.time()
            all_predictions = [
                self._parse_llm_response(response, info["num_values"])
                for response, info in zip(all_responses, batch_info)
            ]

            pred_values = []
            pred_labels = []
            for preds, info in zip(all_predictions, batch_info):
                pred_values.extend(info["values"])
                pred_labels.extend(preds)

            mapper = pd.Series(pred_labels, index=pred_values, dtype=np.int8)
            predictions = all_values.map(mapper).fillna(0).astype(np.int8).values
            parsing_endtime = time.time() - parsing_starttime
            print(f"  Parsed LLM responses in {parsing_endtime:.2f} seconds")

        except Exception as e:
            print(f"Error classifying batches for column {column_name}: {e}")
            predictions = np.zeros(len(all_values), dtype=np.int8)

        return predictions

    def __call__(
        self,
        dataset: pd.DataFrame,
        types: Dict[str, str],
        target_columns: List[str] = None,
        embeddings: Dict[str, pd.DataFrame] = {},
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], List[str]]:
        """
        Run LLM-based classification on the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to analyze.
            types (Dict[str, str]): Dictionary mapping column names to their types.
            target_columns (List[str], optional): Specific columns to analyze. If None, analyzes all columns.
            embeddings (Dict[str, pd.DataFrame], optional): Not used by this detector.

        Returns:
            Tuple containing:
                - df_score (pd.DataFrame): Binary predictions (same as df_predict for this detector)
                - df_predict (pd.DataFrame): Binary predictions (1 for DMV, 0 for valid)
                - times (Dict[str, float]): Runtime statistics
                - assessed (List[str]): List of columns that were analyzed
        """

        times = {
            "classification": 0,
            "total": 0,
        }

        total_starttime = time.time()
        assessed = []

        df_score = dataset.copy()
        df_score.loc[:, :] = 0
        df_score = df_score.astype(float)

        df_predict = dataset.copy()
        df_predict.loc[:, :] = 0
        df_predict = df_predict.astype(int)

        columns = dataset.columns if target_columns is None else target_columns

        for column in columns:
            if types[column] not in self.target_types:
                continue

            if len(dataset[column].dropna()) == 0:
                continue

            print(f"Classifying column: {column}")
            assessed.append(column)

            classification_starttime = time.time()

            predictions = self._classify_column(dataset[column], column, types[column])

            df_predict[column] = predictions
            df_score[column] = predictions.astype(float)

            times["classification"] += time.time() - classification_starttime

        times["total"] = time.time() - total_starttime

        return df_score, df_predict, times, assessed
