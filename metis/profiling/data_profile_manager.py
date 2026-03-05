from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional

from sqlalchemy import Engine, create_engine as sa_create_engine, select
from sqlalchemy.orm import Session

from metis.database_models import Base, DataProfile


class DataProfileManager:
    """Singleton that manages data-profile caching and storage.

    Typical lifecycle:
        1. ``DQOrchestrator`` calls ``DataProfileManager.initialize(engine)``
           to hand over a SQLAlchemy engine (or a connection string).
        2. Before running metrics for a dataset the orchestrator calls
           ``set_context(dataset=..., table=...)`` so cached wrappers know
           which dataset is active.
        3. Cached profiling functions use ``get_instance()`` internally.
    """

    _instance: Optional[DataProfileManager] = None
    _lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Singleton access
    # ------------------------------------------------------------------ #
    @classmethod
    def initialize(cls, engine_or_url: Engine | str) -> DataProfileManager:
        """Create (or re-create) the singleton with the given engine."""
        with cls._lock:
            if isinstance(engine_or_url, str):
                engine = sa_create_engine(engine_or_url)
            else:
                engine = engine_or_url
            cls._instance = cls(engine)
            Base.metadata.create_all(engine)
            return cls._instance

    @classmethod
    def get_instance(cls) -> DataProfileManager:
        """Return the current singleton.  Raises if not initialized."""
        if cls._instance is None:
            raise RuntimeError(
                "DataProfileManager has not been initialized. "
                "Call DataProfileManager.initialize(engine) first."
            )
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._instance is not None

    @classmethod
    def shutdown(cls) -> None:
        """Shutdown the singleton and dispose the engine."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._engine.dispose()
                cls._instance = None

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._dataset: Optional[str] = None
        self._table: Optional[str] = None
        # In-memory cache for the current run to avoid repeated DB queries
        self._mem_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Context management  (called by DQOrchestrator)
    # ------------------------------------------------------------------ #
    def set_context(self, dataset: str, table: str) -> None:
        """Set the currently active dataset / table."""
        self._dataset = dataset
        self._table = table
        self._mem_cache.clear()

    @property
    def dataset(self) -> Optional[str]:
        return self._dataset

    @property
    def table(self) -> Optional[str]:
        return self._table

    # ------------------------------------------------------------------ #
    # Cache lookup
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cache_key(
        dataset: str,
        table: str,
        column_names: List[str],
        dp_task_name: str,
        task_config: Optional[dict] = None,
    ) -> str:
        config_str = json.dumps(task_config, sort_keys=True) if task_config else ""
        return f"{dataset}|{table}|{','.join(sorted(column_names))}|{dp_task_name}|{config_str}"

    def lookup(
        self,
        column_names: List[str],
        dp_task_name: str,
        task_config: Optional[dict] = None,
    ) -> Optional[Any]:
        """Look up a cached profiling result.  Returns ``None`` on miss."""
        if self._dataset is None or self._table is None:
            return None

        key = self._cache_key(
            self._dataset, self._table, column_names, dp_task_name, task_config
        )

        # fast path: in-memory
        if key in self._mem_cache:
            return self._mem_cache[key]

        # slow path: DB
        with Session(self._engine) as session:
            stmt = (
                select(DataProfile)
                .where(DataProfile.dataset == self._dataset)
                .where(DataProfile.table_name == self._table)
                .where(DataProfile.dp_task_name == dp_task_name)
            )
            for row in session.execute(stmt).scalars():
                if sorted(row.column_names) == sorted(column_names):
                    cfg = row.task_config or {}
                    if cfg == (task_config or {}):
                        value = self._deserialize(row.dp_result_value, row.result_type)
                        self._mem_cache[key] = value
                        return value
        return None

    # ------------------------------------------------------------------ #
    # Store results
    # ------------------------------------------------------------------ #
    def store(
        self,
        column_names: List[str],
        dp_task_name: str,
        value: Any,
        task_config: Optional[dict] = None,
        profile_type: str = "single_column",
        source: str = "computed",
        dataset: Optional[str] = None,
        table: Optional[str] = None,
    ) -> None:
        """Persist a profiling result to the database."""
        ds = dataset or self._dataset
        tbl = table or self._table
        if ds is None or tbl is None:
            raise RuntimeError(
                "Cannot store profiling result: dataset/table context not set."
            )

        serialized, result_type = self._serialize(value)

        with Session(self._engine) as session:
            # Find existing row with same logical key
            stmt = (
                select(DataProfile)
                .where(DataProfile.dataset == ds)
                .where(DataProfile.table_name == tbl)
                .where(DataProfile.dp_task_name == dp_task_name)
            )
            existing = None
            for row in session.execute(stmt).scalars():
                if sorted(row.column_names) == sorted(column_names):
                    cfg = row.task_config or {}
                    if cfg == (task_config or {}):
                        existing = row
                        break

            if existing is not None:
                existing.dp_result_value = serialized
                existing.result_type = result_type
                existing.profile_type = profile_type
                existing.source = source
            else:
                session.add(DataProfile(
                    dataset=ds,
                    table_name=tbl,
                    column_names=column_names,
                    dp_task_name=dp_task_name,
                    task_config=task_config,
                    profile_type=profile_type,
                    dp_result_value=serialized,
                    result_type=result_type,
                    source=source,
                ))
            session.commit()

        # update in-memory cache
        key = self._cache_key(ds, tbl, column_names, dp_task_name, task_config)
        self._mem_cache[key] = value

    # ------------------------------------------------------------------ #
    # Convenience helpers for dependency storage
    # ------------------------------------------------------------------ #
    def store_fd(
        self,
        lhs: List[str],
        rhs: str,
        dataset: Optional[str] = None,
        table: Optional[str] = None,
        source: str = "computed",
    ) -> None:
        """Store a functional dependency  lhs -> rhs."""
        self.store(
            column_names=sorted(lhs + [rhs]),
            dp_task_name="fd",
            value={"lhs": lhs, "rhs": rhs},
            profile_type="dependency",
            source=source,
            dataset=dataset,
            table=table,
        )

    def store_ucc(
        self,
        columns: List[str],
        dataset: Optional[str] = None,
        table: Optional[str] = None,
        source: str = "computed",
    ) -> None:
        """Store a unique column combination."""
        self.store(
            column_names=sorted(columns),
            dp_task_name="ucc",
            value={"columns": columns},
            profile_type="dependency",
            source=source,
            dataset=dataset,
            table=table,
        )

    def store_ind(
        self,
        dependent: List[str],
        referenced: List[str],
        referenced_table: Optional[str] = None,
        dataset: Optional[str] = None,
        table: Optional[str] = None,
        source: str = "computed",
    ) -> None:
        """Store an inclusion dependency."""
        self.store(
            column_names=sorted(dependent + referenced),
            dp_task_name="ind",
            value={
                "dependent": dependent,
                "referenced": referenced,
                "referenced_table": referenced_table,
            },
            profile_type="dependency",
            source=source,
            dataset=dataset,
            table=table,
        )

    def get_fds(
        self,
        dataset: Optional[str] = None,
        table: Optional[str] = None,
    ) -> List[dict]:
        """Return all stored FDs for a dataset/table."""
        return self._query_by_task("fd", dataset, table)

    def get_uccs(
        self,
        dataset: Optional[str] = None,
        table: Optional[str] = None,
    ) -> List[dict]:
        return self._query_by_task("ucc", dataset, table)

    def get_inds(
        self,
        dataset: Optional[str] = None,
        table: Optional[str] = None,
    ) -> List[dict]:
        return self._query_by_task("ind", dataset, table)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _query_by_task(
        self, dp_task_name: str, dataset: Optional[str], table: Optional[str]
    ) -> List[dict]:
        ds = dataset or self._dataset
        tbl = table or self._table
        if ds is None or tbl is None:
            return []
        with Session(self._engine) as session:
            stmt = (
                select(DataProfile)
                .where(DataProfile.dataset == ds)
                .where(DataProfile.table_name == tbl)
                .where(DataProfile.dp_task_name == dp_task_name)
            )
            return [
                self._deserialize(row.dp_result_value, row.result_type)
                for row in session.execute(stmt).scalars()
            ]

    @staticmethod
    def _serialize(value: Any) -> tuple[dict, str]:
        """Wrap *value* into a JSON-safe dict and return (payload, type_tag)."""
        import numpy as np
        import pandas as pd

        def to_json_safe(v: Any) -> Any:
            """Convert numpy types to native Python types."""
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {k: to_json_safe(val) for k, val in v.items()}
            if isinstance(v, list):
                return [to_json_safe(item) for item in v]
            return v

        if isinstance(value, pd.Series):
            return {"v": to_json_safe(value.to_dict())}, "series"

        # MinHash support (for minhash_signature results)
        from datasketch import MinHash as _MinHash

        if isinstance(value, _MinHash):
            return {
                "v": {
                    "hashvalues": value.hashvalues.tolist(),
                    "num_perm": int(value.num_perm),
                    "seed": int(value.seed),
                }
            }, "minhash"

        if isinstance(value, dict) and value and isinstance(next(iter(value.values())), _MinHash):
            return {
                "v": {
                    k: {
                        "hashvalues": v.hashvalues.tolist(),
                        "num_perm": int(v.num_perm),
                        "seed": int(v.seed),
                    }
                    for k, v in value.items()
                }
            }, "minhash_dict"

        if isinstance(value, dict):
            return {"v": to_json_safe(value)}, "dict"
        if isinstance(value, list):
            return {"v": to_json_safe(value)}, "list"
        # scalar (int, float, str, bool, None …)
        return {"v": to_json_safe(value)}, "scalar"

    @staticmethod
    def _deserialize(payload: Optional[dict], result_type: str) -> Any:
        if payload is None:
            return None
        import pandas as pd

        raw = payload.get("v")
        if result_type == "series":
            return pd.Series(raw)
        if result_type == "minhash":
            from datasketch import MinHash as _MinHash
            import numpy as np
            return _MinHash(
                num_perm=raw["num_perm"],
                seed=raw["seed"],
                hashvalues=np.array(raw["hashvalues"], dtype=np.uint64),
            )
        if result_type == "minhash_dict":
            from datasketch import MinHash as _MinHash
            import numpy as np
            return {
                k: _MinHash(
                    num_perm=v["num_perm"],
                    seed=v["seed"],
                    hashvalues=np.array(v["hashvalues"], dtype=np.uint64),
                )
                for k, v in raw.items()
            }
        return raw
