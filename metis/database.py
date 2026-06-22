from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Literal

from sqlalchemy import MetaData, create_engine, inspect

from metis.database_models import register_models
from metis.utils.logging import logger


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

        if db_type == "sqlite":
            self._backup_if_schema_outdated(Base.metadata, db_config)

        Base.metadata.create_all(self.engine)

        Database._instance = self

    def _backup_if_schema_outdated(self, metadata: MetaData, db_config: Dict) -> None:
        """Back up an existing SQLite database whose schema no longer matches the
        current models, then continue with a fresh file.

        This prevents a hard crash when Metis runs against a database created by
        an older, incompatible version (a deprecated schema). The old file is
        preserved with a timestamp suffix so no data is lost, and the new
        database is created by the regular ``create_all`` call afterwards.
        """
        db_path = db_config["db_name"]
        if db_path == ":memory:" or not os.path.exists(db_path):
            return

        outdated_tables = self._outdated_tables(metadata)
        if not outdated_tables:
            return

        self.engine.dispose()
        backup_path = self._backup_sqlite_file(db_path)
        logger.warning(
            "SQLite database at '%s' uses an outdated schema (tables: %s). "
            "Backed it up to '%s' and created a fresh database.",
            db_path,
            ", ".join(sorted(outdated_tables)),
            backup_path,
        )
        self.engine = self.create_sqlite_engine(db_config)

    def _outdated_tables(self, metadata: MetaData) -> list[str]:
        """Return managed tables whose existing columns differ from the models.

        Tables that do not yet exist are ignored: ``create_all`` creates them
        without conflict.
        """
        inspector = inspect(self.engine)
        existing_tables = set(inspector.get_table_names())

        outdated = []
        for table_name, table in metadata.tables.items():
            if table_name not in existing_tables:
                continue
            expected_columns = {column.name for column in table.columns}
            actual_columns = {
                column["name"] for column in inspector.get_columns(table_name)
            }
            if expected_columns != actual_columns:
                outdated.append(table_name)
        return outdated

    @staticmethod
    def _backup_sqlite_file(db_path: str) -> str:
        """Rename the database file to a unique timestamped backup and return its path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(db_path)
        backup_path = f"{base}_{timestamp}{ext}"
        counter = 1
        while os.path.exists(backup_path):
            backup_path = f"{base}_{timestamp}_{counter}{ext}"
            counter += 1
        os.rename(db_path, backup_path)
        return backup_path

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
        required_keys = ("db_name",)
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
