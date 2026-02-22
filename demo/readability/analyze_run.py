from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt


def _find_latest_db_export_run(base: Path) -> Path:
    runs = sorted(base.glob("*_db_export"), reverse=True)
    if not runs:
        raise FileNotFoundError(f"No *_db_export run found in: {base}")
    return runs[0]


def _safe_get(d: Any, key: str, default=None):
    if isinstance(d, dict):
        return d.get(key, default)
    return default


def _normalize_column_name(x: Any) -> Optional[str]:
    # column_names can be None, "null", ["country"], '["country"]', etc.
    if x is None:
        return None
    if isinstance(x, list):
        return str(x[0]) if x else None
    s = str(x).strip()
    if s.lower() in ("null", "none", ""):
        return None
    try:
        j = json.loads(s)
        if isinstance(j, list) and j:
            return str(j[0])
    except Exception:
        pass
    return s


def main() -> None:
    base = Path("demo/readability/runs")
    run_dir = _find_latest_db_export_run(base)
    csv_path = run_dir / "readability_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    # parse dq_explanation if it was written as JSON-ish string
    def parse_expl(x: Any) -> Any:
        if pd.isna(x):
            return {}
        if isinstance(x, dict):
            return x
        s = str(x)
        try:
            return json.loads(s)
        except Exception:
            return {}

    df["dq_explanation_parsed"] = df.get("dq_explanation", "{}").apply(parse_expl)
    df["column_name_norm"] = df.get("column_names", None).apply(_normalize_column_name)

    # ---- Build TABLE summary: WordNet vs LLM fallback
    table_rows = df[df["dq_granularity"] == "table"].copy()

    # pick content + schema table scores
    wn_content = table_rows[table_rows["dq_metric"] == "readability_wordnet_content"]
    llm_content = table_rows[table_rows["dq_metric"] == "readability_llm_content_wordnetFirst_fallback"]

    wn_schema = table_rows[table_rows["dq_metric"] == "readability_wordnet_schema"]
    llm_schema = table_rows[table_rows["dq_metric"] == "readability_llm_schema_wordnetFirst_fallback"]

    # helper to extract one scalar (if multiple, we keep mean and also list)
    def summarize_metric(sub: pd.DataFrame, label: str) -> Dict[str, Any]:
        vals = sub["dq_value"].astype(float).tolist()
        return {
            f"{label}_n": len(vals),
            f"{label}_mean": float(pd.Series(vals).mean()) if vals else None,
            f"{label}_values": vals,
        }

    summary: Dict[str, Any] = {}
    summary.update(summarize_metric(wn_content, "wordnet_content"))
    summary.update(summarize_metric(llm_content, "llm_content"))
    summary.update(summarize_metric(wn_schema, "wordnet_schema"))
    summary.update(summarize_metric(llm_schema, "llm_schema"))

    # uplift (means)
    def uplift(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(a - b)

    summary["uplift_content_mean"] = uplift(summary["llm_content_mean"], summary["wordnet_content_mean"])
    summary["uplift_schema_mean"] = uplift(summary["llm_schema_mean"], summary["wordnet_schema_mean"])

    # extract token diagnostics from llm_content explanations (if present)
    # (these live in dq_explanation of readability_llm_content_wordnetFirst_fallback)
    if not llm_content.empty:
        ex = llm_content.iloc[-1]["dq_explanation_parsed"]
        summary["llm_tokens_count_total"] = _safe_get(ex, "llm_tokens_count_total")
        summary["unique_tokens_count_total"] = _safe_get(ex, "unique_tokens_count_total")
        summary["llm_tokens_share_total"] = _safe_get(ex, "llm_tokens_share_total")
        summary["hf_model_id"] = _safe_get(ex, "hf_model_id")
        summary["llm_mode"] = _safe_get(ex, "llm_mode")
        summary["use_llm_fallback"] = _safe_get(ex, "use_llm_fallback")

    # write summary json
    (run_dir / "summary_table.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---- Column-level comparison table
    col_rows = df[df["dq_granularity"] == "column"].copy()
    wn_cols = col_rows[col_rows["dq_metric"] == "readability_wordnet_content_column"].copy()
    llm_cols = col_rows[col_rows["dq_metric"] == "readability_llm_content_column"].copy()

    wn_cols = wn_cols[["table_name", "column_name_norm", "dq_value", "dq_explanation_parsed"]].rename(
        columns={"dq_value": "wordnet_score", "dq_explanation_parsed": "wordnet_expl"}
    )
    llm_cols = llm_cols[["table_name", "column_name_norm", "dq_value", "dq_explanation_parsed"]].rename(
        columns={"dq_value": "llm_score", "dq_explanation_parsed": "llm_expl"}
    )

    merged = pd.merge(
        wn_cols,
        llm_cols,
        on=["table_name", "column_name_norm"],
        how="outer",
    )

    merged["uplift"] = merged["llm_score"].astype(float) - merged["wordnet_score"].astype(float)

    # token diagnostics per column (if present)
    merged["llm_tokens_count"] = merged["llm_expl"].apply(lambda d: _safe_get(d, "llm_tokens_count"))
    merged["unique_tokens_count"] = merged["llm_expl"].apply(lambda d: _safe_get(d, "unique_tokens_count"))
    merged["llm_tokens_share"] = merged["llm_expl"].apply(lambda d: _safe_get(d, "llm_tokens_share"))

    merged = merged.sort_values(["uplift"], ascending=False)

    merged.to_csv(run_dir / "columns_comparison.csv", index=False)

    # ---- Plots folder
    figs = run_dir / "figs"
    figs.mkdir(exist_ok=True)

    # Plot 1: table scores (wordnet vs llm)
    labels = []
    wn_vals = []
    llm_vals = []

    if summary["wordnet_content_mean"] is not None and summary["llm_content_mean"] is not None:
        labels.append("content")
        wn_vals.append(summary["wordnet_content_mean"])
        llm_vals.append(summary["llm_content_mean"])
    if summary["wordnet_schema_mean"] is not None and summary["llm_schema_mean"] is not None:
        labels.append("schema")
        wn_vals.append(summary["wordnet_schema_mean"])
        llm_vals.append(summary["llm_schema_mean"])

    if labels:
        x = range(len(labels))
        width = 0.35
        plt.figure()
        plt.bar([i - width / 2 for i in x], wn_vals, width=width, label="WordNet")
        plt.bar([i + width / 2 for i in x], llm_vals, width=width, label="WordNet+LLM fallback")
        plt.xticks(list(x), labels)
        plt.ylim(0, 1)
        plt.title("Readability table scores")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs / "table_scores.png", dpi=160)
        plt.close()

    # Plot 2: column uplift bar (top N)
    topn = merged.head(20).copy()
    if not topn.empty:
        plt.figure(figsize=(10, 5))
        names = topn["column_name_norm"].fillna("<?>").astype(str).tolist()
        vals = topn["uplift"].astype(float).tolist()
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), names, rotation=45, ha="right")
        plt.title("Top column uplift (LLM - WordNet)")
        plt.tight_layout()
        plt.savefig(figs / "top_column_uplift.png", dpi=160)
        plt.close()

    # Plot 3: token share per column (if present)
    tok = merged.dropna(subset=["llm_tokens_share"]).copy()
    if not tok.empty:
        tok = tok.sort_values("llm_tokens_share", ascending=False).head(20)
        plt.figure(figsize=(10, 5))
        names = tok["column_name_norm"].fillna("<?>").astype(str).tolist()
        vals = tok["llm_tokens_share"].astype(float).tolist()
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), names, rotation=45, ha="right")
        plt.ylim(0, 1)
        plt.title("Top LLM token share per column")
        plt.tight_layout()
        plt.savefig(figs / "top_llm_token_share.png", dpi=160)
        plt.close()

    print("OK - run:", run_dir)
    print("OK - wrote:", run_dir / "summary_table.json")
    print("OK - wrote:", run_dir / "columns_comparison.csv")
    print("OK - figs:", figs)


if __name__ == "__main__":
    main()
