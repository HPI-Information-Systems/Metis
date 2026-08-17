from dataclasses import dataclass
from typing import Set

import pandas as pd

from metis.metric.config import MetricConfig

@dataclass
class validity_outOfVocabulary_config(MetricConfig):
    """
    Configuration class for the validity_outOfVocabulary metric.

    :param reference: Reference vocabulary to use for the out-of-vocabulary check. This can be provided as a DataFrame with a single column, a set of strings, or None to use the default NLTK English word list.
    """

    reference: pd.DataFrame | Set | None = None
