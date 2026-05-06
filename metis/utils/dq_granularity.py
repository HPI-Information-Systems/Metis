from enum import StrEnum


class DQGranularity(StrEnum):
    """Data Quality Granularity Enum. Primarily used for labeling DQResults inside each metric implementation with the DQ Granularity they assess."""

    CELL = "cell"
    ROW = "row"
    COLUMN = "column"
    TABLE = "table"
