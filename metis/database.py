from typing import Dict, Literal

from sqlalchemy import create_engine

from metis.database_models import register_models


class Database:
    """Provides a singleton reference for the database connection and models. Can be used by different modules to access the database without risking conflicts caused by multiple bases or engines."""

    _instance: Database | None = None

    def __init__(self, db_type: Literal["sqlite", "postgres"], db_config: Dict):
        if Database._instance is not None:
            raise RuntimeError(
                "Database has already been initialized. Use Database.get_instance() to access the singleton."
            )

        self.engine = self.create_engine(db_type, db_config)

        Base, self.DQResultModel, self.DataProfile = register_models(
            db_config.get("table_name", "dq_results")
        )
        Base.metadata.create_all(self.engine)

        Database._instance = self

    @classmethod
    def get_instance(cls) -> Database:
        """Return the current singleton. Raises if not initialized."""
        if cls._instance is None:
            raise RuntimeError(
                "Database has not been initialized. "
                "Call Database.initialize(engine) first."
            )
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._instance is not None

    def create_engine(self, db_type: Literal["sqlite", "postgres"], db_config: Dict):
        if db_type == "sqlite":
            return self.create_sqlite_engine(db_config)
        elif db_type == "postgres":
            return self.create_postgres_engine(db_config)
        raise ValueError(f"Unsupported database type: {db_type}")

    def create_sqlite_engine(self, db_config: Dict):
        required_keys = "db_name"
        if not all(k in db_config for k in required_keys):
            raise ValueError(
                f"SQLite database config must include the following fields: {required_keys}."
            )

        return create_engine(
            f"sqlite:///{db_config['db_name']}",
            echo=db_config.get("echo", False),
        )

    def create_postgres_engine(self, db_config: Dict):
        required_keys = ("db_user", "db_pass", "db_name", "db_host", "db_port")
        if not all(k in db_config for k in required_keys):
            raise ValueError(
                f"Postgres database config must include the following fields: {required_keys}."
            )

        return create_engine(
            f"postgresql://{db_config['db_user']}:{db_config['db_pass']}@{db_config['db_host']}:{db_config['db_port']}/{db_config['db_name']}",
            echo=db_config.get("echo", False),
        )
