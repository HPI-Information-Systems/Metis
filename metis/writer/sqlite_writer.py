from sqlalchemy import Engine, create_engine

from metis.writer.database_writer import DatabaseWriter


class SQLiteWriter(DatabaseWriter):
    def create_engine(self, writer_config) -> Engine:
        if not "db_name" in writer_config:
            raise ValueError("SQLite writer config must include 'db_name' field.")

        return create_engine(f"sqlite:///{writer_config['db_name']}", echo=True)
