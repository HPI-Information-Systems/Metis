from dataclasses import dataclass

from metis.metric.config import MetricConfig
from metis.utils.reference_loader import ReferenceSource, describe_reference_source


@dataclass
class accuracy_semanticReference_config(MetricConfig):
    """Configuration for ``accuracy_semanticReference`` (ISO/IEC 25024 Acc-I-2).

    :param reference: The gold-standard data values are compared against.
        Accepts a DataFrame, a CSV path, or a DataConfig mapping.
    :param key_column: If set, ``data`` and the reference are joined on this
        column before per-cell comparison. If ``None`` (default), rows are
        compared positionally and both must have the same length.
    """

    reference: ReferenceSource = None
    key_column: str | None = None

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "reference": describe_reference_source(self.reference),
            "key_column": self.key_column,
        }

    def validate(self):
        if self.reference is None:
            raise ValueError(
                "accuracy_semanticReference requires a reference. Set the "
                "'reference' field to a DataFrame, a CSV path, or a DataConfig "
                "mapping."
            )
        if self.key_column is not None and not isinstance(self.key_column, str):
            raise ValueError(
                f"key_column must be a string or None, got {type(self.key_column)}."
            )
