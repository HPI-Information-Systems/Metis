from enum import StrEnum


class DQDimension(StrEnum):
    """Data Quality Dimensions Enum. Primarily used for labeling DQResults inside each metric implementation with the DQ Dimension they assess."""

    CONSISTENCY = "Consistency"
    CORRECTNESS = "Correctness"
    COMPLETENESS = "Completeness"
    TIMELINESS = "Timeliness"
