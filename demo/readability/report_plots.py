# demo/readability/report_plots.py
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from metis.dq_orchestrator import DQOrchestrator

tmp_path = Path(__file__).resolve().parent / "tmp_path.json"

# ----------------------------
# Helpers: IO / Paths
# ----------------------------

def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_dataset(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = p.suffix.lower()
    if suffix in [".csv"]:
        return pd.read_csv(p)
    if suffix in [".json"]:
        # countries-capitals is often JSON array -> pandas handles it
        return pd.read_json(p)
    if suffix in [".jsonl"]:
        return pd.read_json(p, lines=True)

    raise ValueError(f"Unsupported dataset type: {suffix}")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Experiment: Pollution
# ----------------------------

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _random_token(rng: random.Random, length: int = 8) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(length))


def _pollute_text(s: Any, rng: random.Random, degree: float) -> Any:
    """
    Replace some word tokens with random strings to simulate 'pollution'.
    degree in [0,1].
    """
    if s is None:
        return s
    txt = str(s)
    if not txt.strip():
        return s

    # simple tokenization (keep spaces punctuation mostly)
    parts = txt.split()
    if not parts:
        return s

    out = []
    for tok in parts:
        if rng.random() < degree:
            out.append(_random_token(rng, length=max(4, min(12, len(tok)))))
        else:
            out.append(tok)
    return " ".join(out)


def pollute_dataframe(df: pd.DataFrame, rng: random.Random, degree: float, columns: Optional[List[str]] = None) -> pd.DataFrame:
    out = df.copy()
    cols = columns or [c for c in out.columns]
    for c in cols:
        # only pollute object/string columns
        dt = str(out[c].dtype)
        if dt != "object" and not dt.startswith("string"):
            continue
        out[c] = out[c].apply(lambda x: _pollute_text(x, rng, degree))
    return out


# ----------------------------
# METIS Runner (WordNet vs LLM)
# ----------------------------

@dataclass
class ReadabilityRunResult:
    label: str
    table_wordnet: float
    table_llm: float
    col_wordnet: Dict[str, float]
    col_llm: Dict[str, float]
    meta_wordnet: Dict[str, Any]
    meta_llm: Dict[str, Any]
    runtime_wordnet_s: float
    runtime_llm_s: float


def _extract_results(results: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    From orchestrator output (list of DQResult dicts), pick:
    - wordnet table result
    - llm table result
    - column results for each
    """
    wn_table = None
    llm_table = None
    col = {
        "wn": {},
        "llm": {},
        "wn_meta": {},
        "llm_meta": {},
    }

    for r in results:
        dim = r.get("DQdimension")
        if dim != "Readability":
            continue
        metric = r.get("DQmetric")
        gran = r.get("DQgranularity")

        # WordNet-only metric names (your current implementation)
        if metric == "readability_wordnet_content" and gran == "table":
            wn_table = r
        if metric == "readability_wordnet_content_column" and gran == "column":
            cn = r.get("columnNames") or []
            if cn:
                col["wn"][cn[0]] = float(r.get("DQvalue", 0.0))
                col["wn_meta"][cn[0]] = r.get("DQexplanation") or {}

        # LLM hybrid metric names (your current implementation)
        if metric == "readability_llm_content_wordnetFirst_fallback" and gran == "table":
            llm_table = r
        if metric == "readability_llm_content_column" and gran == "column":
            cn = r.get("columnNames") or []
            if cn:
                col["llm"][cn[0]] = float(r.get("DQvalue", 0.0))
                col["llm_meta"][cn[0]] = r.get("DQexplanation") or {}

    return wn_table, llm_table, col


def run_readability_on_df(
    df: pd.DataFrame,
    dataset_label: str,
    readability_config_path: str,
    include_schema: bool = True,
) -> ReadabilityRunResult:
    """
    Run readability_wordnet and readability_llm on an in-memory dataframe.
    Uses METIS orchestrator, but loads data from a temp CSV.
    """
    # Save to temp to reuse existing METIS loader path logic
    # Save CSV
    tmp_dir = Path("demo/readability/_tmp")
    _ensure_dir(tmp_dir)

    csv_path = tmp_dir / f"_tmp_{dataset_label}.csv"
    df.to_csv(csv_path, index=False)

    # Create loader config JSON
    loader_config = {
        "type": "csv",
        "path": str(csv_path),
        "separator": ",",
        "header": True
    }

    json_path = tmp_dir / f"_tmp_{dataset_label}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(loader_config, f, indent=2)

    # Now load via JSON config
    orchestrator = DQOrchestrator()
    orchestrator.load(data_loader_configs=[str(json_path)])


    # WordNet-only run
    t0 = time.time()
    results_wn = orchestrator.assess(
        metrics=["readability_wordnet"],
        metric_configs=[readability_config_path],
    )
    t1 = time.time()

    # LLM hybrid run
    t2 = time.time()
    results_llm = orchestrator.assess(
        metrics=["readability_llm"],
        metric_configs=[readability_config_path],
    )
    t3 = time.time()

    # Orchestrator may print + return None depending on your version.
    # In your setup, it prints results but ALSO should return list.
    # If it returns None, we fail loudly with a clear message.
    if results_wn is None or results_llm is None:
        raise RuntimeError(
            "orchestrator.assess returned None. "
            "In your METIS version it may only write via writer and print. "
            "Fix: make assess() return the DQResults list, or read from SQLite/jsonl instead."
        )

    wn_table, _, cols_wn = _extract_results(results_wn)
    _, llm_table, cols_llm = _extract_results(results_llm)

    if wn_table is None:
        raise RuntimeError("Could not find WordNet table result (readability_wordnet_content).")
    if llm_table is None:
        raise RuntimeError("Could not find LLM table result (readability_llm_content_wordnetFirst_fallback).")

    return ReadabilityRunResult(
        label=dataset_label,
        table_wordnet=float(wn_table["DQvalue"]),
        table_llm=float(llm_table["DQvalue"]),
        col_wordnet=cols_wn["wn"],
        col_llm=cols_llm["llm"],
        meta_wordnet=wn_table.get("DQexplanation") or {},
        meta_llm=llm_table.get("DQexplanation") or {},
        runtime_wordnet_s=(t1 - t0),
        runtime_llm_s=(t3 - t2),
    )


# ----------------------------
# Plotting (Paper-like)
# ----------------------------

def plot_pipeline_comparison(run: ReadabilityRunResult, out_png: Path) -> None:
    # Table-level comparison (WordNet vs LLM)
    labels = ["WordNet", "WordNet+LLM(Fallback)"]
    values = [run.table_wordnet, run.table_llm]

    plt.figure()
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Readability score (table)")
    plt.title("Readability Pipeline Comparison (Table)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_column_comparison(run: ReadabilityRunResult, out_png: Path) -> None:
    # Column-level comparison
    cols = sorted(set(run.col_wordnet.keys()) | set(run.col_llm.keys()))
    wn_vals = [run.col_wordnet.get(c, 0.0) for c in cols]
    llm_vals = [run.col_llm.get(c, 0.0) for c in cols]

    x = list(range(len(cols)))
    width = 0.4

    plt.figure()
    plt.bar([i - width/2 for i in x], wn_vals, width=width, label="WordNet")
    plt.bar([i + width/2 for i in x], llm_vals, width=width, label="WordNet+LLM(Fallback)")
    plt.ylim(0, 1)
    plt.xticks(x, cols, rotation=30, ha="right")
    plt.ylabel("Readability score (column)")
    plt.title("Readability Pipeline Comparison (Column)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_pollution_curve(pollution_points: List[Tuple[float, float, float]], out_png: Path) -> None:
    """
    pollution_points: list of (degree, wordnet_score, llm_score)
    """
    deg = [p[0] for p in pollution_points]
    wn = [p[1] for p in pollution_points]
    llm = [p[2] for p in pollution_points]

    plt.figure()
    plt.plot(deg, wn, marker="o", label="WordNet")
    plt.plot(deg, llm, marker="o", label="WordNet+LLM(Fallback)")
    plt.ylim(0, 1)
    plt.xlabel("Pollution degree")
    plt.ylabel("Readability score (table)")
    plt.title("Pollution Experiment (Table Readability)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_runtime_curve(runtime_points: List[Tuple[int, float, float]], out_png: Path) -> None:
    """
    runtime_points: list of (sample_size, runtime_wordnet, runtime_llm)
    """
    ss = [p[0] for p in runtime_points]
    wn = [p[1] for p in runtime_points]
    llm = [p[2] for p in runtime_points]

    plt.figure()
    plt.plot(ss, wn, marker="o", label="WordNet")
    plt.plot(ss, llm, marker="o", label="WordNet+LLM(Fallback)")
    plt.xlabel("Sample size")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Sample Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/countries-capitals.json", help="Path to dataset (csv/json/jsonl)")
    ap.add_argument("--cfg", default="configs/metric/readability.json", help="Readability metric config JSON path")
    ap.add_argument("--out-root", default="demo/readability/runs", help="Root output directory")
    ap.add_argument("--seed", type=int, default=13, help="Seed for pollution experiment")
    ap.add_argument("--do-pollution", action="store_true", help="Run pollution experiment (like paper)")
    ap.add_argument("--pollution", default="0,0.1,0.2,0.5", help="Comma list of pollution degrees")
    ap.add_argument("--do-runtime", action="store_true", help="Run runtime experiment over sample sizes")
    ap.add_argument("--sample-sizes", default="50,100,200,500", help="Comma list sample sizes for runtime experiment")
    args = ap.parse_args()

    df = _load_dataset(args.dataset)
    dataset_label = Path(args.dataset).stem

    run_dir = Path(args.out_root) / f"{_now_stamp()}_{dataset_label}"
    plots_dir = run_dir / "plots"
    _ensure_dir(plots_dir)

    # 1) Base run (no pollution)
    base = run_readability_on_df(
        df=df,
        dataset_label=dataset_label,
        readability_config_path=args.cfg,
    )

    # Save base results summary
    summary = {
        "dataset": args.dataset,
        "cfg": args.cfg,
        "table_wordnet": base.table_wordnet,
        "table_llm": base.table_llm,
        "runtime_wordnet_s": base.runtime_wordnet_s,
        "runtime_llm_s": base.runtime_llm_s,
        "meta_wordnet": base.meta_wordnet,
        "meta_llm": base.meta_llm,
        "col_wordnet": base.col_wordnet,
        "col_llm": base.col_llm,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 2) Plots: pipeline comparison (paper-like)
    plot_pipeline_comparison(base, plots_dir / "pipeline_table_comparison.png")
    plot_column_comparison(base, plots_dir / "pipeline_column_comparison.png")

    # 3) Pollution experiment (paper-like)
    if args.do_pollution:
        rng = random.Random(args.seed)
        degrees = [float(x.strip()) for x in args.pollution.split(",") if x.strip() != ""]
        points: List[Tuple[float, float, float]] = []

        for d in degrees:
            d = max(0.0, min(1.0, d))
            df_p = pollute_dataframe(df, rng=rng, degree=d)
            rr = run_readability_on_df(
                df=df_p,
                dataset_label=f"{dataset_label}_pollution_{int(d*100)}",
                readability_config_path=args.cfg,
            )
            points.append((d, rr.table_wordnet, rr.table_llm))

        # save raw points
        (run_dir / "pollution_points.json").write_text(json.dumps(points, indent=2), encoding="utf-8")
        plot_pollution_curve(points, plots_dir / "pollution_curve.png")

    # 4) Runtime experiment over sample sizes
    if args.do_runtime:
        # We overwrite only sample_size inside cfg by creating a temporary JSON per run.
        cfg_obj = json.loads(Path(args.cfg).read_text(encoding="utf-8"))
        sizes = [int(x.strip()) for x in args.sample_sizes.split(",") if x.strip() != ""]
        points_rt: List[Tuple[int, float, float]] = []

        tmp_cfg_dir = run_dir / "_tmp_cfg"
        _ensure_dir(tmp_cfg_dir)

        for n in sizes:
            cfg_obj["sample_size"] = int(n)
            tmp_cfg = tmp_cfg_dir / f"readability_sample_{n}.json"
            tmp_cfg.write_text(json.dumps(cfg_obj, indent=2), encoding="utf-8")

            rr = run_readability_on_df(
                df=df,
                dataset_label=f"{dataset_label}_sample_{n}",
                readability_config_path=str(tmp_cfg),
            )
            points_rt.append((n, rr.runtime_wordnet_s, rr.runtime_llm_s))

        (run_dir / "runtime_points.json").write_text(json.dumps(points_rt, indent=2), encoding="utf-8")
        plot_runtime_curve(points_rt, plots_dir / "runtime_curve.png")

    print(f"OK - wrote: {run_dir}")
    print(f"OK - plots: {plots_dir}")


if __name__ == "__main__":
    main()
