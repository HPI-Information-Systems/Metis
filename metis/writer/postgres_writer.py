from sqlalchemy import Engine, create_engine

from metis.writer.database_writer import DatabaseWriter


class PostgresWriter(DatabaseWriter):
    def create_engine(self, writer_config) -> Engine:
        required_keys = ("db_user", "db_pass", "db_name", "db_host", "db_port")
        if not all(k in writer_config for k in required_keys):
            raise ValueError(
                "Postgres writer config must include 'db_user', 'db_pass', 'db_name', 'db_host', and 'db_port' fields."
            )

        return create_engine(
            f"postgresql://{writer_config['db_user']}:{writer_config['db_pass']}@{writer_config['db_host']}:{writer_config['db_port']}/{writer_config['db_name']}",
            echo=True,
        )
