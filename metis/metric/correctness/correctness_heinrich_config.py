from dataclasses import dataclass

from metis.metric.config import MetricConfig
from metis.utils.reference_loader import ReferenceSource, describe_reference_source


@dataclass
class correctness_heinrich_config(MetricConfig):
    """Configuration for ``correctness_heinrich``.

    :param reference: The clean reference data every cell is compared
        against. Must resolve to a DataFrame of the same shape as the assessed
        data. Accepts a DataFrame, a CSV path, or a DataConfig mapping.
    """

    reference: ReferenceSource = None

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "reference": describe_reference_source(self.reference),
        }

    def validate(self):
        if self.reference is None:
            raise ValueError(
                "correctness_heinrich requires a reference. Set the 'reference' "
                "field to a DataFrame, a CSV path, or a DataConfig mapping."
            )
