from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


DB_PATH = "dq_repository/dq_repository.db"
OUT_BASE = Path("demo/readability/runs")


def _json_load_maybe(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    s = str(x)
    s = s.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return s


def main() -> None:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM dqresults", con)
    finally:
        con.close()

    # normalize json-like columns if present
    for col in ["dq_explanation", "column_names", "row_index", "config_json"]:
        if col in df.columns:
            df[col] = df[col].apply(_json_load_maybe)

    # filter readability only
    rdf = df[df["dq_dimension"] == "Readability"].copy()

    # create run folder
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = OUT_BASE / f"{ts}_db_export"
    run_dir.mkdir(parents=True, exist_ok=True)

    # write raw exports
    rdf.to_csv(run_dir / "readability_results.csv", index=False)
    (run_dir / "readability_results.json").write_text(
        json.dumps(rdf.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # small meta
    meta = {
        "created_at": ts,
        "db_path": DB_PATH,
        "total_rows_db": int(len(df)),
        "total_rows_readability": int(len(rdf)),
        "metrics_grouped": (
            rdf.groupby(["dq_metric", "dq_granularity"]).size().reset_index(name="count").to_dict(orient="records")
        ),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("OK - wrote:", str(run_dir / "readability_results.csv"))
    print("OK - wrote:", str(run_dir / "readability_results.json"))
    print("OK - wrote:", str(run_dir / "meta.json"))
    print("ROWS readability:", len(rdf))
    print("GROUPED:\n", rdf.groupby(["dq_metric", "dq_granularity"]).size())


if __name__ == "__main__":
    main()
