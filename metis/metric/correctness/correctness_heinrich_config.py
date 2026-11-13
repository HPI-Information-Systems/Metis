from dataclasses import dataclass

from metis.metric.config import MetricConfig
from metis.utils.reference_loader import (
    MissingReferenceError,
    ReferenceSource,
    describe_reference_source,
    has_reference,
)


@dataclass
class correctness_heinrich_config(MetricConfig):
    """Configuration for ``correctness_heinrich``.

    :param reference: The clean reference data every cell is compared
        against. Must resolve to a DataFrame of the same shape as the assessed
        data. Accepts a DataFrame, a CSV path, or a DataConfig mapping. When
        left at ``None``, the dataset-level reference (``reference_file_name``
        in the data loader config) is used.
    """

    reference: ReferenceSource = None

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "reference": describe_reference_source(self.reference),
        }

    def validate(self):
        if not has_reference(self.reference):
            raise MissingReferenceError(
                "correctness_heinrich requires a reference. Set the 'reference' "
                "field to a DataFrame, a CSV path, or a DataConfig mapping, or "
                "declare 'reference_file_name' in the data loader config."
            )
