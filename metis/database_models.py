from datetime import datetime
from typing import List

from sqlalchemy import JSON, DateTime, Double, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass

def register_models(results_table_name: str):
    class DQResultModel(Base):
        __tablename__ = results_table_name
        __table_args__ = {"extend_existing": True}

        id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
        timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
        dq_dimension: Mapped[str]
        dq_metric: Mapped[str]
        dq_granularity: Mapped[str]
        dq_value: Mapped[float] = mapped_column(Double)
        dq_explanation: Mapped[dict | None] = mapped_column(JSON)
        runtime: Mapped[float | None] = mapped_column(Double)
        table_name: Mapped[str | None]
        column_names: Mapped[List[str] | None] = mapped_column(JSON)
        row_index: Mapped[int | None]
        experiment_tag: Mapped[str | None]
        dataset: Mapped[str | None]
        config_json: Mapped[dict | None] = mapped_column(JSON)

    return DQResultModel

class DataProfile(Base):
    """Stores data profiling results for caching and manual imports.

    Covers single-column statistics (null_count, distinct_count, histograms, ...),
    multi-column dependencies (FDs, UCCs, INDs, ...), and any other profiling
    result type.  The result payload is stored as JSON so the schema stays
    flexible across different task types.
    """

    __tablename__ = "data_profiles"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # --- identifiers ---
    dataset: Mapped[str]
    table_name: Mapped[str]
    column_names: Mapped[List[str]] = mapped_column(JSON)
    dp_task_name: Mapped[str]                          # e.g. "null_count", "fd", "ucc"
    task_config: Mapped[dict | None] = mapped_column(JSON)  # extra params used

    # --- category ---
    profile_type: Mapped[str] = mapped_column(default="single_column")
    # "single_column" | "multi_column" | "dependency" | "custom"

    # --- result ---
    dp_result_value: Mapped[dict | None] = mapped_column(JSON)  # {"v": <actual_value>}
    result_type: Mapped[str] = mapped_column(default="scalar")
    # "scalar" | "list" | "dict" | "series" — for deserialization hint

    # --- provenance ---
    source: Mapped[str] = mapped_column(default="computed")
    # "computed" | "imported:hyfd" | "imported:manual" | …
