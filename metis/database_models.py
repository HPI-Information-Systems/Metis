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
        mes_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
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
