import re
import pandas as pd
from typing import List, Union
from metis.utils.result import DQResult
from metis.metric.metric import Metric

import nltk
from nltk.corpus import words as nltk_words
nltk.download("words", quiet=True)

class validity_outOfVocabulary(Metric):
    def assess(self, data: pd.DataFrame, reference: Union[pd.DataFrame, set, None] = None, metric_config: Union[str, None] = None) -> List[DQResult]:
        """
        General vocabulary check at token level.
        Any alphabetic token not in the standard vocab is OOV.
        """
        results: List[DQResult] = []

        # Build vocabulary (lowercase)
        if reference is None:
            vocab_set = {w.lower() for w in nltk_words.words()}
            ref_src = "NLTK English words"
        elif isinstance(reference, pd.DataFrame):
            if reference.shape[1] != 1:
                raise ValueError("Reference DataFrame must have exactly one column.")
            vocab_set = {str(x).strip().lower() for x in reference.iloc[:, 0].dropna().unique()}
            ref_src = "Custom vocabulary"
        elif isinstance(reference, set):
            vocab_set = {str(x).strip().lower() for x in reference}
            ref_src = "Custom vocabulary"
        else:
            raise ValueError("Reference must be a one column DataFrame, a set, or None.")

        def tokenize(text: str):
            return re.findall(r"[A-Za-z]+", text.lower())

        for column in data.columns:
            col_values = data[column].dropna().astype(str)
            total_not_null_values = len(col_values)

            if total_not_null_values == 0:
                dq_value = 0.0
                in_vocab_count = 0
            else:
                def is_valid(text: str) -> bool:
                    tokens = tokenize(text)
                    if not tokens:
                        return True  # empty or numeric-like strings are treated as valid
                    # valid if *all* tokens are in vocabulary
                    return all(token in vocab_set for token in tokens)

                in_vocab_flags = col_values.map(is_valid)
                in_vocab_count = int(in_vocab_flags.sum())
                dq_value = in_vocab_count / total_not_null_values

            annotations = {}
            if dq_value < 1.0:
                annotations = {
                    "TotalNotNullValues": total_not_null_values,
                    "InVocabValues": in_vocab_count,
                    "ReferenceSource": ref_src
                }

            result = DQResult(
                timestamp=pd.Timestamp.now(),
                DQdimension="Validity",
                DQmetric="OutOfVocabulary",
                DQgranularity="column",
                DQvalue=dq_value,
                DQexplanation=annotations,
                columnNames=[column],
            )
            results.append(result)

        return results
