from datetime import datetime
from typing import List

from sqlalchemy import JSON, Double, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass

def register_models(results_table_name: str):
    class DQResultModel(Base):
        __tablename__ = results_table_name
        __table_args__ = {"extend_existing": True}

        id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
        mes_time: Mapped[datetime] = mapped_column(server_default=func.now())
        dq_value: Mapped[float] = mapped_column(Double)
        dq_dimension: Mapped[str]
        dq_metric: Mapped[str]
        column_name: Mapped[List[str] | None] = mapped_column(JSON)
        row_index: Mapped[int | None]
        dq_annotations: Mapped[dict | None] = mapped_column(JSON)
        dataset: Mapped[str | None]
        table_name: Mapped[str | None]

    return DQResultModel
