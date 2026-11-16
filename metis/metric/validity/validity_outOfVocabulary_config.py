from dataclasses import dataclass

from metis.metric.config import MetricConfig
from metis.utils.reference_loader import ReferenceSource, describe_reference_source


@dataclass
class validity_outOfVocabulary_config(MetricConfig):
    """Configuration for ``validity_outOfVocabulary``.

    :param reference: The vocabulary tokens are checked against. Accepts a
        single-column DataFrame, a CSV path, a DataConfig mapping, or an
        inline list/set of words. ``None`` falls back to the NLTK English
        word list.
    """

    reference: ReferenceSource = None

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "reference": describe_reference_source(self.reference),
        }
