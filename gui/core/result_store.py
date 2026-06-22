"""Persistence layer.

``SQLiteResultStore`` is the primary backend used by the native Streamlit app.
``JSONResultStore`` is used in browser (stlite) mode and writes/reads JSON files
on the IDBFS-mounted filesystem.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import repeat

import numpy as np
import pandas as pd

from core.serialization import result_to_dict
from metis.utils.result import DQResult

try:
    from sqlalchemy import func, select, text
    from sqlalchemy.orm import Session

    from metis.database import Database
    _SQLALCHEMY_AVAILABLE = True
except ImportError:
    _SQLALCHEMY_AVAILABLE = False

try:
    import orjson

    def _dumps(value) -> str:
        return orjson.dumps(value).decode()
except ImportError:
    def _dumps(value) -> str:
        return json.dumps(value)


logger = logging.getLogger("metis").getChild("gui.result_store")

DEFAULT_DB_PATH: str = os.path.join(os.path.dirname(__file__), "..", "..", "dq_repository", "dq_repository.db")
_BROWSER_RESULTS_DIR: str = "/metis_results"

TABLE_COLUMN_PLACEHOLDER: str = "(table)"
UNKNOWN_COLUMN_PLACEHOLDER: str = "(unknown)"

HISTOGRAM_BIN_COUNT: int = 20
WORST_RESULTS_LIMIT: int = 200

_GRANULARITY_PREFERENCE: tuple[str, ...] = ("cell", "row", "column", "table")


# Fragments cannot receive the store as an argument because SQLAlchemy's ORM
# model classes (held inside Database) are not picklable, which causes
# Streamlit to hang when it tries to serialise fragment args into session state.
# The active store is therefore registered once from app.py and looked up via
# ``get_active_store()`` from within fragments.
_active_store: "ResultStore | None" = None


def set_active_store(store: "ResultStore") -> None:
    """
    Register the active result store for the current process.

    :param store: The result store to use as the singleton.
    :return: None.
    """
    global _active_store
    _active_store = store


def get_active_store() -> "ResultStore":
    """
    Look up the active result store registered by ``set_active_store``.

    :return: The active result store.
    :raises RuntimeError: If no active store has been registered yet.
    """
    if _active_store is None:
        raise RuntimeError("No active store — call set_active_store() from app.py first.")
    return _active_store


def _primary_granularity(granularities: set) -> str:
    """
    Pick the most granular level present from a set of granularities.

    :param granularities: Set of granularity strings.
    :return: One of ``cell``, ``row``, ``column``, ``table``.
    """
    for g in _GRANULARITY_PREFERENCE:
        if g in granularities:
            return g
    return next(iter(granularities), "table")


def _primary_column_label(column_names) -> str:
    """
    Reduce a list of column names to a single primary label for indexing.

    :param column_names: Either a list of column names or ``None``.
    :return: The first column name, or :data:`TABLE_COLUMN_PLACEHOLDER` when empty.
    """
    if not column_names:
        return TABLE_COLUMN_PLACEHOLDER
    return column_names[0]


@dataclass
class RunMetadata:
    experiment_tag: str
    dataset_name: str
    table_name: str | None = None


@dataclass
class RunSummary:
    experiment_tag: str
    dataset_name: str
    timestamp: str
    result_count: int
    metrics: list[str]


class ResultStore(ABC):
    """Abstract base for any persistence backend."""

    @abstractmethod
    def save_run(self, results: list[DQResult], metadata: RunMetadata) -> None: ...

    @abstractmethod
    def list_runs(self) -> list[RunSummary]: ...

    @abstractmethod
    def load_results(self, experiment_tag: str) -> list[dict]: ...

    @abstractmethod
    def load_results_for_metric(self, metric_name: str) -> list[dict]: ...

    @abstractmethod
    def load_temporal_data(self, metric_name: str, dataset_name: str = "") -> list[dict]: ...

    @abstractmethod
    def delete_run(self, experiment_tag: str) -> None: ...

    @abstractmethod
    def export_json(self, experiment_tag: str) -> bytes: ...

    @abstractmethod
    def list_metrics_for_run(self, tag: str) -> list[str]: ...

    @abstractmethod
    def list_columns_for_run(self, tag: str) -> list[str]: ...

    @abstractmethod
    def count_column_metrics(self, tag: str) -> int: ...

    @abstractmethod
    def get_metric_summary(self, tag: str, metric: str, granularity: str | None = None) -> dict: ...

    @abstractmethod
    def get_column_aggregates(self, tag: str, metric: str) -> list[dict]: ...

    @abstractmethod
    def get_column_results(self, tag: str, metric: str) -> list[dict]: ...

    @abstractmethod
    def get_histogram_data(self, tag: str, metric: str, granularity: str | None = None) -> list[dict]: ...

    @abstractmethod
    def get_worst_results(
        self, tag: str, metric: str, granularity: str | None = None, n: int = WORST_RESULTS_LIMIT,
    ) -> list[dict]: ...

    @abstractmethod
    def get_table_results(self, tag: str, metric: str) -> list[dict]: ...

    @abstractmethod
    def get_heatmap_data(self, tag: str) -> list[dict]: ...


class JSONResultStore(ResultStore):
    """
    Browser (stlite) persistence layer.

    Each run is stored as a separate JSON file in ``base_dir`` (IDBFS-mounted at
    ``/metis_results`` in the stlite build). The pre-computed demo file can be
    seeded by placing ``restaurant_results.json`` in the same directory before
    mount.
    """

    def __init__(self, base_dir: str = _BROWSER_RESULTS_DIR) -> None:
        self._dir = base_dir
        self._file_cache: dict[str, list[dict]] = {}
        os.makedirs(base_dir, exist_ok=True)

    def save_run(self, results: list[DQResult], metadata: RunMetadata) -> None:
        """
        Append serialized results to the run's JSON file.

        :param results: Results emitted by the metric run.
        :param metadata: Run-level metadata (tag, dataset name, optional table name).
        :return: None.
        """
        tag = metadata.experiment_tag
        self._file_cache.pop(tag, None)
        existing = self._load_file(tag)
        existing.extend([result_to_dict(r) for r in results])
        for rec in existing:
            rec.setdefault("experimentTag", tag)
            rec.setdefault("dataset", metadata.dataset_name)
        self._save_file(tag, existing)
        self._file_cache[tag] = existing

    def list_runs(self) -> list[RunSummary]:
        """
        Enumerate all runs by scanning the JSON files on disk.

        :return: A list of run summaries (one per JSON file found).
        """
        summaries = []
        for fname in sorted(os.listdir(self._dir)):
            if not fname.endswith(".json"):
                continue
            tag = fname[:-5]
            records = self._load_file(tag)
            if not records:
                continue
            metrics = sorted({r.get("DQmetric", "") for r in records})
            ts = min(
                (r.get("timestamp", "") for r in records if r.get("timestamp")),
                default="",
            )
            dataset = records[0].get("dataset", "") if records else ""
            summaries.append(RunSummary(
                experiment_tag=tag,
                dataset_name=dataset,
                timestamp=ts,
                result_count=len(records),
                metrics=metrics,
            ))
        return summaries

    def load_results(self, experiment_tag: str) -> list[dict]:
        return self._load_file(experiment_tag)

    def load_results_for_metric(self, metric_name: str) -> list[dict]:
        all_results = []
        for run in self.list_runs():
            all_results.extend(
                r for r in self._load_file(run.experiment_tag)
                if r.get("DQmetric") == metric_name
            )
        return sorted(all_results, key=lambda r: r.get("timestamp", ""))

    def load_temporal_data(self, metric_name: str, dataset_name: str = "") -> list[dict]:
        """
        Compute per-(run, column) mean DQ values for the temporal chart.

        :param metric_name: Metric whose history to load.
        :param dataset_name: Optional dataset filter (empty string disables it).
        :return: A list of ``{timestamp, tag, column, DQvalue}`` dicts ordered by timestamp.
        """
        buckets: dict[tuple, list] = defaultdict(list)
        ts_by_key: dict[tuple, str] = {}
        for run in self.list_runs():
            if dataset_name and run.dataset_name != dataset_name:
                continue
            for r in self._load_file(run.experiment_tag):
                if r.get("DQmetric") != metric_name:
                    continue
                val = r.get("DQvalue")
                if val is None:
                    continue
                granularity = r.get("DQgranularity", "")
                col_names = r.get("columnNames") or []
                col = col_names[0] if (granularity == "column" and col_names) else TABLE_COLUMN_PLACEHOLDER
                tag = r.get("experimentTag", "")
                ts = r.get("timestamp", "")
                key = (tag, col)
                buckets[key].append(val)
                if key not in ts_by_key or ts < ts_by_key[key]:
                    ts_by_key[key] = ts
        return [
            {
                "timestamp": ts_by_key[(tag, col)],
                "tag": tag,
                "column": col,
                "DQvalue": sum(vals) / len(vals),
            }
            for (tag, col), vals in sorted(
                buckets.items(), key=lambda x: ts_by_key[x[0]]
            )
        ]

    def list_metrics_for_run(self, tag: str) -> list[str]:
        records = self._load_file(tag)
        return sorted({r.get("DQmetric", "") for r in records if r.get("DQmetric")})

    def list_columns_for_run(self, tag: str) -> list[str]:
        records = self._load_file(tag)
        seen: set[str] = set()
        cols: list[str] = []
        for r in records:
            for c in (r.get("columnNames") or []):
                if c and c not in seen:
                    cols.append(c)
                    seen.add(c)
        return cols

    def count_column_metrics(self, tag: str) -> int:
        records = self._load_file(tag)
        return len({r.get("DQmetric") for r in records if r.get("DQgranularity") == "column"})

    def get_metric_summary(self, tag: str, metric: str, granularity: str | None = None) -> dict:
        records = self._load_file(tag)
        filtered = [r for r in records if r.get("DQmetric") == metric]
        if granularity:
            filtered = [r for r in filtered if r.get("DQgranularity") == granularity]
        if not filtered:
            return _empty_summary()
        granularities = {r.get("DQgranularity", "") for r in filtered}
        vals = [r["DQvalue"] for r in filtered if r.get("DQvalue") is not None]
        mean_score = sum(vals) / len(vals) if vals else None
        pct_perfect = sum(1 for v in vals if v == 1.0) / len(vals) if vals else None
        expl_keys: list[str] = []
        for r in filtered:
            expl = r.get("DQexplanation")
            if expl and isinstance(expl, dict):
                expl_keys = list(expl.keys())
                break
        return {
            "count": len(filtered),
            "granularities": granularities,
            "primary_granularity": _primary_granularity(granularities),
            "explanation_keys": expl_keys,
            "mean_score": mean_score,
            "pct_perfect": pct_perfect,
        }

    def get_column_aggregates(self, tag: str, metric: str) -> list[dict]:
        records = self._load_file(tag)
        filtered = [r for r in records if r.get("DQmetric") == metric and r.get("DQvalue") is not None]
        if not filtered:
            return []
        rows = [
            {
                "column": _primary_column_label(r.get("columnNames")) if r.get("columnNames") else UNKNOWN_COLUMN_PLACEHOLDER,
                "val": r["DQvalue"],
            }
            for r in filtered
        ]
        df = pd.DataFrame(rows)
        agg = (
            df.groupby("column")["val"]
            .agg(mean_score="mean", std_score="std", cnt="count")
            .reset_index()
        )
        agg["std_score"] = agg["std_score"].fillna(0.0)
        agg = agg.sort_values("mean_score").reset_index(drop=True)
        return [
            {
                "column": row["column"],
                "mean_score": row["mean_score"],
                "std_score": row["std_score"],
                "cnt": int(row["cnt"]),
            }
            for _, row in agg.iterrows()
        ]

    def get_column_results(self, tag: str, metric: str) -> list[dict]:
        records = self._load_file(tag)
        rows = [
            {
                "column": _primary_column_label(r.get("columnNames")),
                "DQvalue": r.get("DQvalue") if r.get("DQvalue") is not None else 0.0,
                "DQexplanation": r.get("DQexplanation") or {},
            }
            for r in records
            if r.get("DQmetric") == metric
        ]
        return sorted(rows, key=lambda x: x["DQvalue"])

    def get_histogram_data(self, tag: str, metric: str, granularity: str | None = None) -> list[dict]:
        records = self._load_file(tag)
        vals = [
            r["DQvalue"] for r in records
            if r.get("DQmetric") == metric
            and r.get("DQvalue") is not None
            and (granularity is None or r.get("DQgranularity") == granularity)
        ]
        if not vals:
            return []
        max_bin = HISTOGRAM_BIN_COUNT - 1
        counts: Counter = Counter(min(int(v * HISTOGRAM_BIN_COUNT), max_bin) for v in vals)
        return [{"bin_idx": idx, "count": cnt} for idx, cnt in sorted(counts.items())]

    def get_worst_results(
        self, tag: str, metric: str, granularity: str | None = None, n: int = WORST_RESULTS_LIMIT,
    ) -> list[dict]:
        records = self._load_file(tag)
        rows = []
        for r in records:
            if r.get("DQmetric") != metric:
                continue
            if r.get("DQvalue") is None:
                continue
            if granularity and r.get("DQgranularity") != granularity:
                continue
            col_names = r.get("columnNames") or []
            rows.append({
                "column": col_names[0] if col_names else UNKNOWN_COLUMN_PLACEHOLDER,
                "row_index": r.get("rowIndex"),
                "dq_value": r["DQvalue"],
            })
        rows.sort(key=lambda x: x["dq_value"])
        return rows[:n]

    def get_table_results(self, tag: str, metric: str) -> list[dict]:
        records = self._load_file(tag)
        return [
            {
                "DQvalue": r.get("DQvalue"),
                "DQexplanation": r.get("DQexplanation") or {},
                "columnNames": r.get("columnNames") or [],
            }
            for r in records
            if r.get("DQmetric") == metric
        ]

    def get_heatmap_data(self, tag: str) -> list[dict]:
        records = self._load_file(tag)
        filtered = [
            r for r in records
            if r.get("DQgranularity") in ("column", "cell")
            and r.get("DQvalue") is not None
        ]
        if not filtered:
            return []
        rows = [
            {
                "dq_metric": r.get("DQmetric", ""),
                "column": _primary_column_label(r.get("columnNames")),
                "val": r["DQvalue"],
            }
            for r in filtered
        ]
        df = pd.DataFrame(rows)
        agg = df.groupby(["dq_metric", "column"])["val"].mean().reset_index()
        agg = agg.rename(columns={"val": "mean_score"})
        return agg.sort_values(["dq_metric", "column"]).to_dict("records")

    def delete_run(self, experiment_tag: str) -> None:
        self._file_cache.pop(experiment_tag, None)
        path = self._path(experiment_tag)
        if os.path.exists(path):
            os.unlink(path)

    def export_json(self, experiment_tag: str) -> bytes:
        return json.dumps(self._load_file(experiment_tag), default=str, indent=2).encode()

    def _path(self, tag: str) -> str:
        safe = tag.replace("/", "_").replace("\\", "_")
        return os.path.join(self._dir, f"{safe}.json")

    def _load_file(self, tag: str) -> list[dict]:
        if tag in self._file_cache:
            return self._file_cache[tag]
        path = self._path(tag)
        if not os.path.exists(path):
            return []
        with open(path) as f:
            payload = json.load(f)
        if isinstance(payload, dict) and "results" in payload:
            records = payload["results"]
        else:
            records = payload if isinstance(payload, list) else []
        self._file_cache[tag] = records
        return records

    def _save_file(self, tag: str, records: list[dict]) -> None:
        with open(self._path(tag), "w") as f:
            json.dump(records, f, default=str, indent=2)


class SQLiteResultStore(ResultStore):
    """SQLite-backed result store with vectorised inserts and aggregate side-tables."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        if not _SQLALCHEMY_AVAILABLE:
            raise RuntimeError(
                "SQLAlchemy / metis.database not available. "
                "Use JSONResultStore in browser mode."
            )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if not Database.is_initialized():
            Database("sqlite", {"db_name": db_path})
        self.db = Database.get_instance()
        self._table = self.db.DQResultModel.__tablename__
        self._summary_ready = False
        self._agg_ready = False
        self._insert_event: threading.Event | None = None
        self._insert_thread: threading.Thread | None = None
        self._aggregates_built: set[str] = set()
        self._aggregates_lock = threading.Lock()
        self._ensure_indexes()

    def save_run(self, results: list[DQResult], metadata: RunMetadata) -> None:
        """
        Bulk-insert results and precompute all aggregate side-tables.

        The main N-row insert runs in a background thread so the caller (the
        Streamlit UI) gets control back as soon as the side-tables and run
        summary are written (~15s for 2.4M rows). The background thread finishes
        the main-table insert in parallel; ``export_json`` joins it on demand if
        called before it completes.

        :param results: Results to persist.
        :param metadata: Run-level metadata (tag, dataset, optional table).
        :return: None.
        """
        n_rows = len(results)
        tag = metadata.experiment_tag

        fields = _extract_result_fields(results)
        ts_list, min_ts = _vectorise_timestamps(fields["timestamps"], n_rows)
        dq_vals = _nan_to_none(fields["values"])

        col_jsons, col_labels = _encode_column_lists(fields["column_names"])
        cfg_jsons = _encode_configs(fields["configs"])
        expl_jsons = [_dumps(e) if e is not None else None for e in fields["explanations"]]

        agg_frames = _compute_aggregates_pd(
            fields["metrics"], fields["granularities"], dq_vals,
            col_labels, fields["row_indices"], expl_jsons,
        )

        self._write_aggregates_pd(tag, agg_frames)
        self._update_run_summary_fast(tag, metadata, set(fields["metrics"]), min_ts, n_rows)
        self._kick_off_main_insert(tag, metadata, fields, ts_list, dq_vals, col_jsons, cfg_jsons, expl_jsons)

    def _kick_off_main_insert(
        self,
        tag: str,
        metadata: RunMetadata,
        fields: dict,
        ts_list: list[str],
        dq_vals: list,
        col_jsons: list,
        cfg_jsons: list,
        expl_jsons: list,
    ) -> None:
        """
        Start the background daemon thread that bulk-inserts into the main table.

        The main-table insert is only needed for ``export_json()`` and
        ``load_results()`` (rare operations); ``_insert_thread`` and
        ``_insert_event`` let those callers join it when needed.

        :param tag: The experiment tag.
        :param metadata: Run metadata (for table name and dataset).
        :param fields: The output of :func:`_extract_result_fields`.
        :param ts_list: Pre-formatted timestamp strings.
        :param dq_vals: NaN-cleaned DQ values.
        :param col_jsons: Pre-encoded column-name JSON strings.
        :param cfg_jsons: Pre-encoded config JSON strings.
        :param expl_jsons: Pre-encoded explanation JSON strings.
        :return: None.
        """
        insert_sql = (
            f"INSERT INTO {self._table}"
            " (timestamp, dq_dimension, dq_metric, dq_granularity, dq_value,"
            "  dq_explanation, runtime, table_name, column_names,"
            "  row_index, experiment_tag, dataset, config_json)"
            " VALUES (?,?,?,?,?, ?,?,?,?, ?,?,?,?)"
        )
        rows = list(zip(
            ts_list, fields["dimensions"], fields["metrics"], fields["granularities"], dq_vals,
            expl_jsons, fields["runtimes"], repeat(metadata.table_name), col_jsons,
            fields["row_indices"], repeat(tag), repeat(metadata.dataset_name), cfg_jsons,
        ))

        def _bg_insert() -> None:
            raw = self.db.engine.raw_connection()
            try:
                cur = raw.cursor()
                cur.execute("PRAGMA synchronous = OFF")
                cur.executemany(insert_sql, rows)
                raw.commit()
                cur.execute("PRAGMA synchronous = NORMAL")
                raw.commit()
            finally:
                raw.close()
            self._insert_event.set()

        self._insert_event = threading.Event()
        self._insert_thread = threading.Thread(target=_bg_insert, daemon=True)
        self._insert_thread.start()

    def _write_aggregates_pd(self, tag: str, frames: dict) -> None:
        """
        Write all five aggregate side-tables using raw sqlite3 ``executemany``.

        Uses ``raw_connection()`` plus ``executemany`` with positional ``?``
        placeholders and ``list(zip(...))`` over numpy arrays — avoids the
        overhead of SQLAlchemy named-param dicts and ``DataFrame.to_dict``.

        :param tag: The experiment tag.
        :param frames: Output of :func:`_compute_aggregates_pd`.
        :return: None.
        """
        metric_df = frames["metric_df"]
        gran_df = frames["gran_df"]
        col_df = frames["col_df"]
        hist_df = frames["hist_df"]
        worst_df = frames["worst_df"]

        raw = self.db.engine.raw_connection()
        try:
            cur = raw.cursor()

            for tbl in ("dq_metric_summary", "dq_granularity_summary",
                        "dq_column_agg", "dq_histogram", "dq_worst"):
                cur.execute(f"DELETE FROM {tbl} WHERE experiment_tag = ?", (tag,))

            if not metric_df.empty:
                cur.executemany(
                    "INSERT INTO dq_metric_summary"
                    " (experiment_tag, dq_metric, result_count, granularities_csv,"
                    "  mean_score, pct_perfect, explanation_json)"
                    " VALUES (?,?,?,?,?,?,?)",
                    list(zip(
                        [tag] * len(metric_df),
                        metric_df["metric"].tolist(),
                        metric_df["count"].tolist(),
                        metric_df["grans_csv"].tolist(),
                        metric_df["mean_score"].tolist(),
                        metric_df["pct_perfect"].tolist(),
                        metric_df["expl_json"].tolist(),
                    )),
                )

            if not gran_df.empty:
                cur.executemany(
                    "INSERT INTO dq_granularity_summary"
                    " (experiment_tag, dq_metric, granularity, result_count, mean_score, pct_perfect)"
                    " VALUES (?,?,?,?,?,?)",
                    list(zip(
                        [tag] * len(gran_df),
                        gran_df["metric"].tolist(),
                        gran_df["gran"].tolist(),
                        gran_df["count"].tolist(),
                        gran_df["mean_score"].tolist(),
                        gran_df["pct_perfect"].tolist(),
                    )),
                )

            if not col_df.empty:
                cur.executemany(
                    "INSERT INTO dq_column_agg"
                    " (experiment_tag, dq_metric, column_name, mean_score, std_score, cnt)"
                    " VALUES (?,?,?,?,?,?)",
                    list(zip(
                        [tag] * len(col_df),
                        col_df["metric"].tolist(),
                        col_df["col"].tolist(),
                        col_df["mean_score"].tolist(),
                        col_df["std_score"].tolist(),
                        col_df["count"].tolist(),
                    )),
                )

            if not hist_df.empty:
                cur.executemany(
                    "INSERT INTO dq_histogram"
                    " (experiment_tag, dq_metric, granularity, bin_idx, count)"
                    " VALUES (?,?,?,?,?)",
                    list(zip(
                        [tag] * len(hist_df),
                        hist_df["metric"].tolist(),
                        hist_df["gran"].tolist(),
                        hist_df["bin_idx"].tolist(),
                        hist_df["count"].tolist(),
                    )),
                )

            if not worst_df.empty:
                cur.executemany(
                    "INSERT INTO dq_worst"
                    " (experiment_tag, dq_metric, granularity, column_name, row_index, dq_value)"
                    " VALUES (?,?,?,?,?,?)",
                    list(zip(
                        [tag] * len(worst_df),
                        worst_df["metric"].tolist(),
                        worst_df["gran"].tolist(),
                        worst_df["col"].tolist(),
                        worst_df["ridx"].tolist(),
                        worst_df["val"].tolist(),
                    )),
                )

            raw.commit()
        finally:
            raw.close()

        self._agg_ready = True

    def _update_run_summary_fast(
        self,
        tag: str,
        metadata: RunMetadata,
        metrics_set: set[str],
        min_ts: str,
        new_count: int,
    ) -> None:
        """
        Upsert ``dq_run_summary`` from in-memory values without touching the main table.

        :param tag: The experiment tag.
        :param metadata: Run metadata (for dataset name).
        :param metrics_set: Set of metric names appearing in this batch.
        :param min_ts: Minimum timestamp string in this batch.
        :param new_count: Number of new results added.
        :return: None.
        """
        with self.db.engine.connect() as conn:
            existing = conn.execute(text(
                "SELECT result_count, metrics_csv, created_at FROM dq_run_summary WHERE experiment_tag = :tag"
            ), {"tag": tag}).fetchone()

            if existing:
                old_metrics = set(existing[1].split(",")) if existing[1] else set()
                merged_metrics = ",".join(sorted(old_metrics | metrics_set))
                merged_count = existing[0] + new_count
                merged_ts = min(existing[2], min_ts) if existing[2] else min_ts
            else:
                merged_metrics = ",".join(sorted(metrics_set))
                merged_count = new_count
                merged_ts = min_ts

            conn.execute(text("""
                INSERT INTO dq_run_summary (experiment_tag, dataset_name, created_at, result_count, metrics_csv)
                VALUES (:tag, :dataset, :ts, :cnt, :metrics)
                ON CONFLICT(experiment_tag) DO UPDATE SET
                    dataset_name = excluded.dataset_name,
                    created_at   = excluded.created_at,
                    result_count = excluded.result_count,
                    metrics_csv  = excluded.metrics_csv
            """), {
                "tag": tag,
                "dataset": metadata.dataset_name,
                "ts": merged_ts,
                "cnt": merged_count,
                "metrics": merged_metrics,
            })
            conn.commit()

        self._summary_ready = True

    def list_runs(self) -> list[RunSummary]:
        """
        Look up all runs from ``dq_run_summary`` (fast O(N_runs)).

        Falls back to a live ``GROUP BY`` while the one-time background migration
        is still running (first startup after ``dq_run_summary`` was introduced).

        :return: A list of run summaries ordered most-recent first.
        """
        if not self._summary_ready:
            return self._list_runs_live()
        with self.db.engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT experiment_tag, dataset_name, created_at, result_count, metrics_csv
                FROM dq_run_summary
                ORDER BY created_at DESC
            """)).fetchall()
        return self._rows_to_summaries(rows)

    def _list_runs_live(self) -> list[RunSummary]:
        """Fallback ``GROUP BY`` query used while the background migration is pending."""
        t = self._table
        with self.db.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT
                    COALESCE(experiment_tag, '')     AS experiment_tag,
                    dataset,
                    MIN(timestamp)                   AS min_ts,
                    COUNT(*)                         AS cnt,
                    GROUP_CONCAT(DISTINCT dq_metric) AS metrics_csv
                FROM {t}
                GROUP BY COALESCE(experiment_tag, ''), dataset
                ORDER BY MIN(timestamp) DESC
            """)).fetchall()
        return self._rows_to_summaries(rows)

    @staticmethod
    def _rows_to_summaries(rows) -> list[RunSummary]:
        summaries = []
        for row in rows:
            metrics_csv = row[4] or ""
            metrics = sorted(m for m in metrics_csv.split(",") if m)
            summaries.append(RunSummary(
                experiment_tag=str(row[0] or ""),
                dataset_name=str(row[1] or UNKNOWN_COLUMN_PLACEHOLDER),
                timestamp=str(row[2]) if row[2] else "",
                result_count=int(row[3]),
                metrics=metrics,
            ))
        return summaries

    def load_results_for_metric(self, metric_name: str) -> list[dict]:
        m = self.db.DQResultModel
        stmt = (
            select(
                m.timestamp, m.dq_dimension, m.dq_metric, m.dq_granularity,
                m.dq_value, m.dq_explanation, m.runtime, m.table_name,
                m.column_names, m.row_index, m.experiment_tag, m.dataset,
                m.config_json,
            )
            .where(m.dq_metric == metric_name)
            .order_by(m.timestamp.asc())
        )
        with self.db.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [_row_to_dict(r) for r in rows]

    def load_temporal_data(self, metric_name: str, dataset_name: str = "") -> list[dict]:
        """
        Per-(run, column) mean for the temporal chart.

        Fast path: join ``dq_column_agg`` with ``dq_run_summary`` — two tiny
        tables. Live fallback: ``GROUP BY`` plus ``json_extract`` over the main
        table.

        :param metric_name: Metric whose history to load.
        :param dataset_name: Optional dataset filter (empty string disables it).
        :return: A list of ``{timestamp, tag, column, DQvalue}`` dicts.
        """
        if self._agg_ready and self._summary_ready:
            with self.db.engine.connect() as conn:
                if dataset_name:
                    rows = conn.execute(text("""
                        SELECT s.created_at, ca.experiment_tag, ca.column_name, ca.mean_score
                        FROM dq_column_agg ca
                        JOIN dq_run_summary s ON s.experiment_tag = ca.experiment_tag
                        WHERE ca.dq_metric = :metric
                          AND s.dataset_name = :dataset
                        ORDER BY s.created_at ASC
                    """), {"metric": metric_name, "dataset": dataset_name}).fetchall()
                else:
                    rows = conn.execute(text("""
                        SELECT s.created_at, ca.experiment_tag, ca.column_name, ca.mean_score
                        FROM dq_column_agg ca
                        JOIN dq_run_summary s ON s.experiment_tag = ca.experiment_tag
                        WHERE ca.dq_metric = :metric
                        ORDER BY s.created_at ASC
                    """), {"metric": metric_name}).fetchall()
            if rows:
                return [
                    {
                        "timestamp": str(r[0]) if r[0] else "",
                        "tag": r[1] or "",
                        "column": r[2] or TABLE_COLUMN_PLACEHOLDER,
                        "DQvalue": r[3],
                    }
                    for r in rows
                ]
        return self._load_temporal_live(metric_name, dataset_name)

    def _load_temporal_live(self, metric_name: str, dataset_name: str = "") -> list[dict]:
        t = self._table
        dataset_filter = "AND dataset = :dataset" if dataset_name else ""
        params: dict = {"metric": metric_name}
        if dataset_name:
            params["dataset"] = dataset_name
        with self.db.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT
                    MIN(timestamp)  AS timestamp,
                    experiment_tag,
                    COALESCE(
                        CASE dq_granularity
                            WHEN 'column' THEN json_extract(column_names, '$[0]')
                        END,
                        '{TABLE_COLUMN_PLACEHOLDER}'
                    )               AS column_label,
                    AVG(dq_value)   AS mean_dqvalue
                FROM {t}
                WHERE dq_metric = :metric
                  AND dq_value   IS NOT NULL
                  {dataset_filter}
                GROUP BY experiment_tag, column_label
                ORDER BY MIN(timestamp) ASC
            """), params).fetchall()
        return [
            {
                "timestamp": str(r.timestamp) if r.timestamp else "",
                "tag": r.experiment_tag or "",
                "column": r.column_label or TABLE_COLUMN_PLACEHOLDER,
                "DQvalue": r.mean_dqvalue,
            }
            for r in rows
        ]

    def list_metrics_for_run(self, tag: str) -> list[str]:
        self._ensure_aggregates(tag)
        if self._agg_ready:
            with self.db.engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT dq_metric FROM dq_metric_summary"
                    " WHERE experiment_tag = :tag ORDER BY dq_metric"
                ), {"tag": tag}).fetchall()
            if rows:
                return [r[0] for r in rows if r[0]]
        t = self._table
        with self.db.engine.connect() as conn:
            rows = conn.execute(text(
                f"SELECT DISTINCT dq_metric FROM {t}"
                f" WHERE experiment_tag = :tag ORDER BY dq_metric"
            ), {"tag": tag}).fetchall()
        return [r[0] for r in rows if r[0]]

    def list_columns_for_run(self, tag: str) -> list[str]:
        self._ensure_aggregates(tag)
        if self._agg_ready:
            with self.db.engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT DISTINCT column_name FROM dq_column_agg"
                    " WHERE experiment_tag = :tag AND column_name != :placeholder"
                    " ORDER BY column_name"
                ), {"tag": tag, "placeholder": TABLE_COLUMN_PLACEHOLDER}).fetchall()
            if rows:
                return [r[0] for r in rows if r[0]]
        t = self._table
        with self.db.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT DISTINCT json_extract(column_names, '$[0]') AS col
                FROM {t}
                WHERE experiment_tag = :tag
                  AND json_extract(column_names, '$[0]') IS NOT NULL
            """), {"tag": tag}).fetchall()
        return [r[0] for r in rows if r[0]]

    def count_column_metrics(self, tag: str) -> int:
        self._ensure_aggregates(tag)
        if self._agg_ready:
            with self.db.engine.connect() as conn:
                row = conn.execute(text("""
                    SELECT COUNT(*) FROM dq_metric_summary
                    WHERE experiment_tag = :tag
                      AND (granularities_csv = 'column'
                           OR granularities_csv LIKE 'column,%'
                           OR granularities_csv LIKE '%,column,%'
                           OR granularities_csv LIKE '%,column')
                """), {"tag": tag}).fetchone()
            if row is not None:
                return int(row[0])
        t = self._table
        with self.db.engine.connect() as conn:
            row = conn.execute(text(f"""
                SELECT COUNT(DISTINCT dq_metric) FROM {t}
                WHERE experiment_tag = :tag AND dq_granularity = 'column'
            """), {"tag": tag}).fetchone()
        return int(row[0]) if row else 0

    def get_metric_summary(self, tag: str, metric: str, granularity: str | None = None) -> dict:
        self._ensure_aggregates(tag)
        if self._agg_ready:
            if granularity is None:
                with self.db.engine.connect() as conn:
                    row = conn.execute(text("""
                        SELECT result_count, granularities_csv, mean_score,
                               pct_perfect, explanation_json
                        FROM dq_metric_summary
                        WHERE experiment_tag = :tag AND dq_metric = :metric
                    """), {"tag": tag, "metric": metric}).fetchone()
                if row:
                    granularities = set(row[1].split(",")) if row[1] else set()
                    expl_keys: list[str] = []
                    if row[4]:
                        try:
                            expl_keys = list(json.loads(row[4]).keys())
                        except (json.JSONDecodeError, AttributeError):
                            pass
                    return {
                        "count": int(row[0]),
                        "granularities": granularities,
                        "primary_granularity": _primary_granularity(granularities),
                        "explanation_keys": expl_keys,
                        "mean_score": row[2],
                        "pct_perfect": row[3],
                    }
            else:
                with self.db.engine.connect() as conn:
                    row = conn.execute(text("""
                        SELECT result_count, mean_score, pct_perfect
                        FROM dq_granularity_summary
                        WHERE experiment_tag = :tag
                          AND dq_metric = :metric
                          AND granularity = :gran
                    """), {"tag": tag, "metric": metric, "gran": granularity}).fetchone()
                if row:
                    gran_set = {granularity}
                    return {
                        "count": int(row[0]),
                        "granularities": gran_set,
                        "primary_granularity": granularity,
                        "explanation_keys": [],
                        "mean_score": row[1],
                        "pct_perfect": row[2],
                    }
        return self._get_metric_summary_live(tag, metric, granularity)

    def _get_metric_summary_live(
        self, tag: str, metric: str, granularity: str | None = None
    ) -> dict:
        t = self._table
        gran_filter = "AND dq_granularity = :granularity" if granularity else ""
        params: dict = {"tag": tag, "metric": metric}
        if granularity:
            params["granularity"] = granularity
        with self.db.engine.connect() as conn:
            row = conn.execute(text(f"""
                SELECT
                    COUNT(*)                                        AS cnt,
                    GROUP_CONCAT(DISTINCT dq_granularity)           AS granularities,
                    AVG(dq_value)                                   AS mean_score,
                    CAST(SUM(CASE WHEN dq_value = 1.0 THEN 1 ELSE 0 END) AS REAL)
                        / NULLIF(COUNT(*), 0)                       AS pct_perfect
                FROM {t}
                WHERE experiment_tag = :tag AND dq_metric = :metric
                {gran_filter}
            """), params).fetchone()
            expl_row = conn.execute(text(f"""
                SELECT dq_explanation FROM {t}
                WHERE experiment_tag = :tag AND dq_metric = :metric
                  AND dq_explanation IS NOT NULL
                {gran_filter}
                LIMIT 1
            """), params).fetchone()
        if not row or not row[0]:
            return _empty_summary()
        granularities = set(row[1].split(",")) if row[1] else set()
        expl_keys: list[str] = []
        if expl_row and expl_row[0]:
            try:
                expl_keys = list(json.loads(expl_row[0]).keys())
            except (json.JSONDecodeError, AttributeError):
                pass
        return {
            "count": int(row[0]),
            "granularities": granularities,
            "primary_granularity": _primary_granularity(granularities),
            "explanation_keys": expl_keys,
            "mean_score": row[2],
            "pct_perfect": row[3],
        }

    def get_column_aggregates(self, tag: str, metric: str) -> list[dict]:
        self._ensure_aggregates(tag)
        if self._agg_ready:
            with self.db.engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT column_name, mean_score, std_score, cnt
                    FROM dq_column_agg
                    WHERE experiment_tag = :tag AND dq_metric = :metric
                    ORDER BY mean_score ASC
                """), {"tag": tag, "metric": metric}).fetchall()
            if rows:
                return [
                    {
                        "column": r[0] or UNKNOWN_COLUMN_PLACEHOLDER,
                        "mean_score": r[1],
                        "std_score": r[2] or 0.0,
                        "cnt": int(r[3]),
                    }
                    for r in rows
                ]
        t = self._table
        with self.db.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT
                    json_extract(column_names, '$[0]')   AS column,
                    AVG(dq_value)                        AS mean_score,
                    SQRT(MAX(0, AVG(dq_value * dq_value) - AVG(dq_value) * AVG(dq_value)))
                                                         AS std_score,
                    COUNT(*)                             AS cnt
                FROM {t}
                WHERE experiment_tag = :tag AND dq_metric = :metric
                GROUP BY json_extract(column_names, '$[0]')
                ORDER BY mean_score ASC
            """), {"tag": tag, "metric": metric}).fetchall()
        return [
            {
                "column": r[0] or UNKNOWN_COLUMN_PLACEHOLDER,
                "mean_score": r[1],
                "std_score": r[2] or 0.0,
                "cnt": int(r[3]),
            }
            for r in rows
        ]

    def get_column_results(self, tag: str, metric: str) -> list[dict]:
        """
        Fetch column-granularity results with explanations.

        Only called for column-level metrics (~N_cols rows per metric) so the
        ``json_extract`` over the already-indexed ``(tag, metric)`` subset is
        fast. No side-table needed here.

        :param tag: The experiment tag.
        :param metric: Metric name.
        :return: A list of ``{column, DQvalue, DQexplanation}`` dicts ordered ascending by DQvalue.
        """
        t = self._table
        with self.db.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT
                    json_extract(column_names, '$[0]') AS column,
                    dq_value,
                    dq_explanation
                FROM {t}
                WHERE COALESCE(experiment_tag, '') = :tag AND dq_metric = :metric
                ORDER BY dq_value ASC
            """), {"tag": tag, "metric": metric}).fetchall()
        result = []
        for r in rows:
            try:
                expl = json.loads(r[2]) if r[2] else {}
            except (json.JSONDecodeError, TypeError):
                expl = {}
            result.append({
                "column": r[0] or TABLE_COLUMN_PLACEHOLDER,
                "DQvalue": r[1] if r[1] is not None else 0.0,
                "DQexplanation": expl,
            })
        return result

    def get_histogram_data(
        self, tag: str, metric: str, granularity: str | None = None
    ) -> list[dict]:
        self._ensure_aggregates(tag)
        if self._agg_ready:
            params: dict = {"tag": tag, "metric": metric}
            gran_filter = "AND granularity = :gran" if granularity else ""
            if granularity:
                params["gran"] = granularity
            with self.db.engine.connect() as conn:
                rows = conn.execute(text(f"""
                    SELECT bin_idx, SUM(count)
                    FROM dq_histogram
                    WHERE experiment_tag = :tag AND dq_metric = :metric
                    {gran_filter}
                    GROUP BY bin_idx
                    ORDER BY bin_idx
                """), params).fetchall()
            if rows:
                return [{"bin_idx": int(r[0]), "count": int(r[1])} for r in rows]
        t = self._table
        gran_filter = "AND dq_granularity = :granularity" if granularity else ""
        params = {"tag": tag, "metric": metric}
        if granularity:
            params["granularity"] = granularity
        max_bin = HISTOGRAM_BIN_COUNT - 1
        with self.db.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT
                    MIN(CAST(dq_value * {HISTOGRAM_BIN_COUNT} AS INTEGER), {max_bin}) AS bin_idx,
                    COUNT(*) AS count
                FROM {t}
                WHERE experiment_tag = :tag AND dq_metric = :metric
                  AND dq_value IS NOT NULL
                {gran_filter}
                GROUP BY bin_idx
                ORDER BY bin_idx
            """), params).fetchall()
        return [{"bin_idx": int(r[0]), "count": int(r[1])} for r in rows]

    def get_worst_results(
        self, tag: str, metric: str, granularity: str | None = None, n: int = WORST_RESULTS_LIMIT,
    ) -> list[dict]:
        self._ensure_aggregates(tag)
        if self._agg_ready:
            params: dict = {"tag": tag, "metric": metric, "n": n}
            gran_filter = "AND granularity = :gran" if granularity else ""
            if granularity:
                params["gran"] = granularity
            with self.db.engine.connect() as conn:
                rows = conn.execute(text(f"""
                    SELECT column_name, row_index, dq_value
                    FROM dq_worst
                    WHERE experiment_tag = :tag AND dq_metric = :metric
                    {gran_filter}
                    ORDER BY dq_value ASC
                    LIMIT :n
                """), params).fetchall()
            if rows:
                return [
                    {"column": r[0] or UNKNOWN_COLUMN_PLACEHOLDER, "row_index": r[1], "dq_value": r[2]}
                    for r in rows
                ]
        t = self._table
        gran_filter = "AND dq_granularity = :granularity" if granularity else ""
        params = {"tag": tag, "metric": metric, "n": n}
        if granularity:
            params["granularity"] = granularity
        with self.db.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT
                    json_extract(column_names, '$[0]') AS column,
                    row_index,
                    dq_value
                FROM {t}
                WHERE experiment_tag = :tag AND dq_metric = :metric
                  AND dq_value IS NOT NULL
                {gran_filter}
                ORDER BY dq_value ASC
                LIMIT :n
            """), params).fetchall()
        return [
            {"column": r[0] or UNKNOWN_COLUMN_PLACEHOLDER, "row_index": r[1], "dq_value": r[2]}
            for r in rows
        ]

    def get_table_results(self, tag: str, metric: str) -> list[dict]:
        t = self._table
        with self.db.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT dq_value, dq_explanation, column_names
                FROM {t}
                WHERE COALESCE(experiment_tag, '') = :tag AND dq_metric = :metric
            """), {"tag": tag, "metric": metric}).fetchall()
        result = []
        for r in rows:
            try:
                expl = json.loads(r[1]) if r[1] else {}
            except (json.JSONDecodeError, TypeError):
                expl = {}
            try:
                col_names = json.loads(r[2]) if r[2] else []
            except (json.JSONDecodeError, TypeError):
                col_names = []
            result.append({"DQvalue": r[0], "DQexplanation": expl, "columnNames": col_names})
        return result

    def get_heatmap_data(self, tag: str) -> list[dict]:
        self._ensure_aggregates(tag)
        if self._agg_ready:
            with self.db.engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT ca.dq_metric, ca.column_name, ca.mean_score
                    FROM dq_column_agg ca
                    JOIN dq_metric_summary ms
                      ON ms.experiment_tag = ca.experiment_tag
                     AND ms.dq_metric      = ca.dq_metric
                    WHERE ca.experiment_tag = :tag
                      AND ca.column_name   != :placeholder
                      AND (ms.granularities_csv = 'cell'
                           OR ms.granularities_csv LIKE 'cell,%'
                           OR ms.granularities_csv LIKE '%,cell,%'
                           OR ms.granularities_csv LIKE '%,cell'
                           OR ms.granularities_csv = 'column'
                           OR ms.granularities_csv LIKE 'column,%'
                           OR ms.granularities_csv LIKE '%,column,%'
                           OR ms.granularities_csv LIKE '%,column')
                    ORDER BY ca.dq_metric, ca.column_name
                """), {"tag": tag, "placeholder": TABLE_COLUMN_PLACEHOLDER}).fetchall()
            if rows:
                return [
                    {"dq_metric": r[0], "column": r[1] or TABLE_COLUMN_PLACEHOLDER, "mean_score": r[2]}
                    for r in rows
                ]
        t = self._table
        with self.db.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT
                    dq_metric,
                    json_extract(column_names, '$[0]') AS column,
                    AVG(dq_value)                      AS mean_score
                FROM {t}
                WHERE experiment_tag = :tag
                  AND dq_granularity IN ('column', 'cell')
                  AND dq_value IS NOT NULL
                GROUP BY dq_metric, column
                ORDER BY dq_metric, column
            """), {"tag": tag}).fetchall()
        return [
            {"dq_metric": r[0], "column": r[1] or TABLE_COLUMN_PLACEHOLDER, "mean_score": r[2]}
            for r in rows
        ]

    def delete_run(self, experiment_tag: str) -> None:
        m = self.db.DQResultModel
        with Session(self.db.engine) as session:
            rows = session.execute(
                select(m).where(func.coalesce(m.experiment_tag, "") == experiment_tag)
            ).scalars().all()
            for row in rows:
                session.delete(row)
            session.commit()
        with self.db.engine.connect() as conn:
            for tbl in ("dq_run_summary", "dq_metric_summary", "dq_granularity_summary",
                        "dq_column_agg", "dq_histogram", "dq_worst"):
                conn.execute(
                    text(f"DELETE FROM {tbl} WHERE experiment_tag = :tag"),
                    {"tag": experiment_tag},
                )
            conn.commit()

    def _wait_for_insert(self) -> None:
        """Block until the background main-table insert thread completes, if running."""
        if self._insert_event is not None and not self._insert_event.is_set():
            self._insert_event.wait()

    def load_results(self, experiment_tag: str) -> list[dict]:
        self._wait_for_insert()
        m = self.db.DQResultModel
        stmt = (
            select(
                m.timestamp, m.dq_dimension, m.dq_metric, m.dq_granularity,
                m.dq_value, m.dq_explanation, m.runtime, m.table_name,
                m.column_names, m.row_index, m.experiment_tag, m.dataset,
                m.config_json,
            )
            .where(func.coalesce(m.experiment_tag, "") == experiment_tag)
        )
        with self.db.engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()
        return [_row_to_dict(r) for r in rows]

    def export_json(self, experiment_tag: str) -> bytes:
        self._wait_for_insert()
        results = self.load_results(experiment_tag)
        return json.dumps(results, default=str, indent=2).encode("utf-8")

    def _ensure_indexes(self) -> None:
        """
        Create indexes and side-tables idempotently at startup.

        Index strategy: two composite covering indexes replace the previous five
        single/dual-column indexes, halving B-tree update cost per insert.

        PRAGMAs run via ``raw_connection()`` because SQLAlchemy 2.0 wraps every
        ``conn.execute()`` in an implicit transaction and SQLite forbids
        changing ``synchronous`` / ``journal_mode`` inside a transaction.

        Background migration is started for legacy DBs that predate
        ``dq_run_summary``. ``_agg_ready`` is set if the aggregate tables
        already contain data (new-format DB or previously migrated).

        :return: None.
        """
        t = self._table

        with self.db.engine.connect() as conn:
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS idx_gui_tag_metric_val"
                f" ON {t}(experiment_tag, dq_metric, dq_value)"
            ))
            conn.execute(text(
                f"CREATE INDEX IF NOT EXISTS idx_gui_metric_tag_val"
                f" ON {t}(dq_metric, experiment_tag, dq_value)"
            ))
            for old in ("idx_gui_exp_tag", "idx_gui_metric", "idx_gui_tag_metric"):
                conn.execute(text(f"DROP INDEX IF EXISTS {old}"))

            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dq_run_summary (
                    experiment_tag  TEXT PRIMARY KEY,
                    dataset_name    TEXT,
                    created_at      TEXT,
                    result_count    INTEGER,
                    metrics_csv     TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dq_metric_summary (
                    experiment_tag   TEXT NOT NULL,
                    dq_metric        TEXT NOT NULL,
                    result_count     INTEGER,
                    granularities_csv TEXT,
                    mean_score       REAL,
                    pct_perfect      REAL,
                    explanation_json TEXT,
                    PRIMARY KEY (experiment_tag, dq_metric)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dq_granularity_summary (
                    experiment_tag  TEXT NOT NULL,
                    dq_metric       TEXT NOT NULL,
                    granularity     TEXT NOT NULL,
                    result_count    INTEGER,
                    mean_score      REAL,
                    pct_perfect     REAL,
                    PRIMARY KEY (experiment_tag, dq_metric, granularity)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dq_column_agg (
                    experiment_tag  TEXT NOT NULL,
                    dq_metric       TEXT NOT NULL,
                    column_name     TEXT NOT NULL,
                    mean_score      REAL,
                    std_score       REAL,
                    cnt             INTEGER,
                    PRIMARY KEY (experiment_tag, dq_metric, column_name)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dq_histogram (
                    experiment_tag  TEXT NOT NULL,
                    dq_metric       TEXT NOT NULL,
                    granularity     TEXT NOT NULL,
                    bin_idx         INTEGER NOT NULL,
                    count           INTEGER,
                    PRIMARY KEY (experiment_tag, dq_metric, granularity, bin_idx)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS dq_worst (
                    experiment_tag  TEXT,
                    dq_metric       TEXT,
                    granularity     TEXT,
                    column_name     TEXT,
                    row_index       INTEGER,
                    dq_value        REAL
                )
            """))
            conn.execute(text(
                "CREATE INDEX IF NOT EXISTS idx_dq_worst_lookup"
                " ON dq_worst(experiment_tag, dq_metric, granularity)"
            ))

            summary_count = conn.execute(
                text("SELECT COUNT(*) FROM dq_run_summary")
            ).scalar()
            agg_count = conn.execute(
                text("SELECT COUNT(*) FROM dq_metric_summary")
            ).scalar()
            conn.commit()

        raw = self.db.engine.raw_connection()
        try:
            cur = raw.cursor()
            cur.execute("PRAGMA journal_mode = WAL")
            cur.execute("PRAGMA synchronous = NORMAL")
            cur.execute("PRAGMA cache_size = -262144")
            cur.execute("PRAGMA temp_store = MEMORY")
        finally:
            raw.close()

        if agg_count and agg_count > 0:
            self._agg_ready = True
        if summary_count and summary_count > 0:
            self._summary_ready = True

        with self.db.engine.connect() as conn:
            has_missing = conn.execute(text(f"""
                SELECT 1 FROM {t}
                WHERE COALESCE(experiment_tag, '') NOT IN (
                    SELECT experiment_tag FROM dq_metric_summary
                )
                   OR COALESCE(experiment_tag, '') NOT IN (
                    SELECT experiment_tag FROM dq_run_summary
                )
                LIMIT 1
            """)).fetchone()
        if has_missing:
            threading.Thread(
                target=self._backfill_background,
                daemon=True,
            ).start()
        else:
            self._summary_ready = True
            self._agg_ready = True

    def _ensure_aggregates(self, tag: str) -> None:
        """
        Make aggregate side-tables and ``dq_run_summary`` consistent for ``tag``.

        Cheap PK lookup on subsequent calls. The first call for a tag that has
        rows in the main table but no aggregates (the CLI-write case) builds
        them synchronously, then caches the result. Thread-safe via
        ``_aggregates_lock`` so concurrent reads don't double-build.

        :param tag: The experiment tag (``""`` for untagged runs).
        :return: None.
        """
        if tag in self._aggregates_built:
            return
        with self._aggregates_lock:
            if tag in self._aggregates_built:
                return
            with self.db.engine.connect() as conn:
                agg_exists = conn.execute(text(
                    "SELECT 1 FROM dq_metric_summary"
                    " WHERE experiment_tag = :tag LIMIT 1"
                ), {"tag": tag}).fetchone()
                summary_exists = conn.execute(text(
                    "SELECT 1 FROM dq_run_summary"
                    " WHERE experiment_tag = :tag LIMIT 1"
                ), {"tag": tag}).fetchone()
            if not agg_exists or not summary_exists:
                self._build_aggregates_for_tag(tag)
            self._aggregates_built.add(tag)
            self._agg_ready = True
            self._summary_ready = True

    def _build_aggregates_for_tag(self, tag: str) -> None:
        """
        Aggregate ``dq_results`` rows for ``tag`` into the five side-tables and
        ``dq_run_summary``. Idempotent: ``_write_aggregates_pd`` uses
        ``DELETE``-then-``INSERT`` per tag, and ``dq_run_summary`` is written
        with ``INSERT OR REPLACE``.

        Reads with ``COALESCE(experiment_tag, '')`` so untagged runs match
        whether the writer stored ``NULL`` or ``""``.

        :param tag: The experiment tag.
        :return: None.
        """
        t = self._table
        with self.db.engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT dq_metric, dq_granularity, dq_value,
                       COALESCE(json_extract(column_names, '$[0]'), :placeholder) AS col_label,
                       row_index, dq_explanation, timestamp, dataset
                FROM {t}
                WHERE COALESCE(experiment_tag, '') = :tag
            """), {"tag": tag, "placeholder": TABLE_COLUMN_PLACEHOLDER}).fetchall()
        if not rows:
            return

        metrics    = [r[0] for r in rows]
        grans      = [r[1] for r in rows]
        dq_vals    = [r[2] for r in rows]
        col_labels = [r[3] for r in rows]
        ridxs      = [r[4] for r in rows]
        expl_jsons = [r[5] for r in rows]

        frames = _compute_aggregates_pd(
            metrics, grans, dq_vals, col_labels, ridxs, expl_jsons
        )
        self._write_aggregates_pd(tag, frames)

        min_ts = min((r[6] for r in rows if r[6]), default="")
        dataset = next((r[7] for r in rows if r[7]), "")
        metrics_csv = ",".join(sorted(set(metrics)))
        raw = self.db.engine.raw_connection()
        try:
            cur = raw.cursor()
            cur.execute(
                "INSERT OR REPLACE INTO dq_run_summary"
                " (experiment_tag, dataset_name, created_at, result_count, metrics_csv)"
                " VALUES (?, ?, ?, ?, ?)",
                (tag, dataset, str(min_ts), len(rows), metrics_csv),
            )
            raw.commit()
        finally:
            raw.close()

    def _backfill_background(self) -> None:
        """
        Background thread: ensure every run in ``dq_results`` has aggregates.

        One pass: discover the set of distinct tags in the main table, then
        call ``_ensure_aggregates`` for each. Cheap when there's nothing to do
        because ``_ensure_aggregates`` short-circuits on its in-memory cache /
        PK lookup. Sets ``_summary_ready`` / ``_agg_ready`` at the end so any
        method gated on them switches to the fast path.

        :return: None.
        """
        t = self._table
        try:
            with self.db.engine.connect() as conn:
                rows = conn.execute(text(
                    f"SELECT DISTINCT COALESCE(experiment_tag, '') FROM {t}"
                )).fetchall()
            tags = [r[0] for r in rows]
            for tag in tags:
                self._ensure_aggregates(tag)
        except Exception as exc:
            logger.warning(f"Background aggregate backfill failed: {exc}")
        finally:
            self._summary_ready = True
            self._agg_ready = True


def _empty_summary() -> dict:
    """Return the canonical empty summary dict used when a metric has no rows."""
    return {
        "count": 0,
        "granularities": set(),
        "primary_granularity": "table",
        "explanation_keys": [],
        "mean_score": None,
        "pct_perfect": None,
    }


def _extract_result_fields(results: list[DQResult]) -> dict[str, list]:
    """
    Pull each :class:`DQResult` field into its own list for vectorised processing.

    :param results: The raw result objects.
    :return: A dict of equal-length lists keyed by field name.
    """
    return {
        "timestamps":    [r.timestamp       for r in results],
        "metrics":       [r.DQmetric        for r in results],
        "dimensions":    [str(r.DQdimension)  for r in results],
        "granularities": [str(r.DQgranularity) for r in results],
        "values":        [r.DQvalue         for r in results],
        "explanations":  [r.DQexplanation   for r in results],
        "column_names":  [r.columnNames     for r in results],
        "row_indices":   [r.rowIndex        for r in results],
        "configs":       [r.configJson      for r in results],
        "runtimes":      [r.runtime         for r in results],
    }


def _vectorise_timestamps(timestamps: list, n_rows: int) -> tuple[list[str], str]:
    """
    Convert a list of pandas Timestamps to ISO strings using a single C pass.

    Uses ``numpy.datetime_as_string`` (~0.3 µs/element) over a microsecond view
    of the nanosecond epoch ints in ``Timestamp.value`` — no per-element Python
    object allocation.

    :param timestamps: Pandas Timestamp objects.
    :param n_rows: Length of ``timestamps`` (passed to avoid a duplicate ``len`` call).
    :return: ``(iso_strings, min_iso_string)``.
    """
    ts_ns = np.array([t.value for t in timestamps], dtype="int64")
    ts_us = ts_ns.view("datetime64[ns]").astype("datetime64[us]")
    ts_arr = np.datetime_as_string(ts_us, unit="us")
    ts_list = ts_arr.tolist()
    min_ts = str(np.datetime_as_string(ts_us.min(), unit="us")) if n_rows > 0 else ""
    return ts_list, min_ts


def _nan_to_none(values: list) -> list:
    """
    Replace ``NaN`` floats with ``None`` for SQL-friendly value lists.

    :param values: Raw DQ values (may contain ``NaN``).
    :return: A new list with ``NaN`` slots replaced by ``None``.
    """
    arr = np.asarray(values, dtype=float)
    nan_idx = np.where(np.isnan(arr))[0]
    out = arr.tolist()
    for i in nan_idx:
        out[i] = None
    return out


def _encode_column_lists(column_names_lists: list) -> tuple[list, list]:
    """
    JSON-encode each result's column-name list with caching, also produce primary labels.

    :param column_names_lists: Per-result column-name lists (some may be ``None``).
    :return: ``(json_strings_or_none, primary_labels)``.
    """
    cache: dict[tuple, str] = {}
    json_list: list = []
    label_list: list = []
    for cols in column_names_lists:
        if cols is None:
            json_list.append(None)
            label_list.append(TABLE_COLUMN_PLACEHOLDER)
            continue
        key = tuple(cols)
        encoded = cache.get(key)
        if encoded is None:
            cache[key] = encoded = _dumps(cols)
        json_list.append(encoded)
        label_list.append(cols[0] if cols else TABLE_COLUMN_PLACEHOLDER)
    return json_list, label_list


def _encode_configs(configs: list) -> list:
    """
    JSON-encode each result's config dict, deduplicating by Python identity.

    :param configs: Per-result config payloads (some may be ``None``).
    :return: A list of JSON strings or ``None``.
    """
    cache: dict[int, str] = {}
    out: list = []
    for cfg in configs:
        if cfg is None:
            out.append(None)
            continue
        key = id(cfg)
        encoded = cache.get(key)
        if encoded is None:
            cache[key] = encoded = _dumps(cfg)
        out.append(encoded)
    return out


def _compute_aggregates_pd(
    metrics: list,
    grans: list,
    dq_vals: list,
    col_labels: list,
    ridxs: list,
    expl_jsons: list,
) -> dict:
    """
    Compute all aggregate DataFrames from pre-extracted field lists.

    Uses only named pandas aggregations (no Python lambdas in groupby) so every
    operation runs at C-level speed. Returns a dict of small DataFrames that
    :meth:`SQLiteResultStore._write_aggregates_pd` inserts into the five
    side-tables.

    :param metrics: Per-row metric names.
    :param grans: Per-row granularity strings.
    :param dq_vals: Per-row DQ values (None for missing).
    :param col_labels: Per-row primary column labels.
    :param ridxs: Per-row indices.
    :param expl_jsons: Per-row JSON-encoded explanations.
    :return: A dict with keys ``metric_df``, ``gran_df``, ``col_df``, ``hist_df``, ``worst_df``.
    """
    df = pd.DataFrame({
        "metric": metrics,
        "gran":   grans,
        "val":    dq_vals,
        "col":    col_labels,
        "ridx":   ridxs,
        "expl":   expl_jsons,
    })
    df["val"] = pd.to_numeric(df["val"], errors="coerce")

    valid = df[df["val"].notna()].copy()
    valid["val_sq"]     = valid["val"] ** 2
    valid["is_perfect"] = (valid["val"] == 1.0).astype("int8")
    max_bin = HISTOGRAM_BIN_COUNT - 1
    valid["bin_idx"]    = (valid["val"] * HISTOGRAM_BIN_COUNT).astype(int).clip(0, max_bin)

    all_cnts = df.groupby("metric", sort=False).size().rename("count")
    num_agg = valid.groupby("metric", sort=False).agg(
        val_sum    =("val",        "sum"),
        val_sq_sum =("val_sq",     "sum"),
        perfect    =("is_perfect", "sum"),
    )
    gran_unique = df.groupby(["metric", "gran"], sort=False).size().reset_index()[["metric", "gran"]]
    grans_csv = gran_unique.groupby("metric")["gran"].apply(
        lambda x: ",".join(sorted(x))
    ).rename("grans_csv")
    expl_sample = (
        df[df["expl"].notna()]
        .groupby("metric", sort=False)["expl"]
        .first()
        .rename("expl_json")
    )
    metric_df = (
        all_cnts.to_frame()
        .join(num_agg,     how="left")
        .join(grans_csv,   how="left")
        .join(expl_sample, how="left")
        .reset_index()
    )
    metric_df["mean_score"] = metric_df["val_sum"] / metric_df["count"]
    metric_df["pct_perfect"] = metric_df["perfect"] / metric_df["count"]

    gran_df = valid.groupby(["metric", "gran"], sort=False).agg(
        count   =("val",        "count"),
        val_sum =("val",        "sum"),
        perfect =("is_perfect", "sum"),
    ).reset_index()
    gran_df["mean_score"] = gran_df["val_sum"] / gran_df["count"]
    gran_df["pct_perfect"] = gran_df["perfect"] / gran_df["count"]

    col_df = valid.groupby(["metric", "col"], sort=False).agg(
        count      =("val",    "count"),
        val_sum    =("val",    "sum"),
        val_sq_sum =("val_sq", "sum"),
    ).reset_index()
    col_df["mean_score"] = col_df["val_sum"] / col_df["count"]
    variance = (col_df["val_sq_sum"] / col_df["count"] - col_df["mean_score"] ** 2).clip(lower=0.0)
    col_df["std_score"] = variance ** 0.5

    hist_df = (
        valid.groupby(["metric", "gran", "bin_idx"], sort=False)
        .size()
        .reset_index(name="count")
    )

    sorted_valid = valid.sort_values("val", kind="mergesort")
    worst_df = (
        sorted_valid
        .groupby(["metric", "gran"], sort=False, group_keys=False)
        .head(WORST_RESULTS_LIMIT)[["metric", "gran", "col", "ridx", "val"]]
        .reset_index(drop=True)
    )
    ridx_series = worst_df["ridx"]
    worst_df["ridx"] = pd.array(
        [None if pd.isna(x) else int(x) for x in ridx_series],
        dtype=object,
    )

    return {
        "metric_df": metric_df,
        "gran_df":   gran_df,
        "col_df":    col_df,
        "hist_df":   hist_df,
        "worst_df":  worst_df,
    }


def _row_to_dict(r) -> dict:
    """
    Convert a SQLAlchemy Core row to the canonical result-dict format.

    :param r: A row returned by ``conn.execute``.
    :return: A dict compatible with :func:`core.serialization.dict_to_result`.
    """
    return {
        "timestamp": r.timestamp.isoformat() if r.timestamp else None,
        "DQdimension": r.dq_dimension,
        "DQmetric": r.dq_metric,
        "DQgranularity": r.dq_granularity,
        "DQvalue": r.dq_value,
        "DQexplanation": r.dq_explanation,
        "runtime": r.runtime,
        "tableName": r.table_name,
        "columnNames": r.column_names,
        "rowIndex": r.row_index,
        "experimentTag": r.experiment_tag,
        "dataset": r.dataset,
        "configJson": r.config_json,
    }
