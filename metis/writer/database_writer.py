from typing import Dict, List

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from metis.models import Base, register_models
from metis.utils.result import DQResult
from metis.writer.writer import DQResultWriter


class DatabaseWriter(DQResultWriter):
    def __init__(self, writer_config: Dict) -> None:
        self.engine = self.create_engine(writer_config)

        self.DQResultModel = register_models(writer_config.get("table_name", "dq_results"))
        Base.metadata.create_all(self.engine)

    def create_engine(self, writer_config: Dict) -> Engine:
        raise NotImplementedError("Subclasses must implement the create_engine method.")

    def write(self, results: List[DQResult]) -> None:
        with Session(self.engine) as session:
            db_entities = [
                self.DQResultModel(
                    mes_time=result.mesTime.to_pydatetime(),
                    dq_value=result.DQvalue,
                    dq_dimension=result.DQdimension,
                    dq_metric=result.DQmetric,
                    column_name=result.columnNames,
                    row_index=result.rowIndex,
                    dq_annotations=result.DQannotations,
                    dataset=result.dataset,
                    table_name=result.tableName,
                )
                for result in results
            ]
            session.add_all(db_entities)
            session.commit()
            session.add_all(db_entities)
            session.commit()
