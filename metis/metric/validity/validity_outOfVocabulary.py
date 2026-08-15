import re
from typing import List

import nltk
import pandas as pd
from nltk.corpus import words as nltk_words

from metis.metric.config import MetricConfig
from metis.metric.metric import Metric
from metis.metric.metric_meta import MetricMeta
from metis.metric.validity.validity_outOfVocabulary_config import (
    validity_outOfVocabulary_config,
)
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity
from metis.utils.reference_loader import load_reference_vocabulary
from metis.utils.result import DQResult


class validity_outOfVocabulary(Metric):
    meta = MetricMeta(
        label="Out of Vocabulary",
        description=(
            "Per column, the share of non-null string values whose alphabetic "
            "tokens all appear in a reference vocabulary. Defaults to NLTK's "
            "English word list when no custom reference is supplied."
        ),
        dimension=DQDimension.VALIDITY,
        granularities=frozenset({DQGranularity.COLUMN}),
    )

    def __init__(self) -> None:
        super().__init__()
        nltk.download("words", quiet=True)

    def assess(
        self,
        data: pd.DataFrame,
        *,
        metric_config: str | MetricConfig | None = None,
    ) -> List[DQResult]:
        """
        General vocabulary check at token level.
        Any alphabetic token not in the standard vocab is OOV.
        """
        results: List[DQResult] = []

        config = self.load_config(
            metric_config or validity_outOfVocabulary_config(),
            validity_outOfVocabulary_config,
        )
        vocab_set = load_reference_vocabulary(config.reference)
        if vocab_set is None:
            vocab_set = {w.lower() for w in nltk_words.words()}
            ref_src = "NLTK English word list"
        else:
            ref_src = "Custom vocabulary"

        def tokenize(text: str):
            return re.findall(r"[A-Za-z]+", text.lower())

        for column in data.columns:
            col_values = data[column].dropna().astype(str)
            total_not_null_values = len(col_values)

            # if total_not_null_values == 0:
            #     dq_value = 0.0
            #     in_vocab_count = 0
            # else:
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
                DQdimension=DQDimension.VALIDITY,
                DQmetric=self.__class__.__name__,
                DQgranularity=DQGranularity.COLUMN,
                DQvalue=dq_value,
                DQexplanation=annotations,
                columnNames=[column],
            )
            results.append(result)

        return results
