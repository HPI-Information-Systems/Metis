# demo/readability/readability_experiment.py
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from zoneinfo import ZoneInfo

from metis.dq_orchestrator import DQOrchestrator
from metis.utils.result import DQResult

# PDF (Report)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
    _HAS_REPORTLAB = True
except Exception:
    _HAS_REPORTLAB = False

# ----------------------------
# Inputs (minimal)
# ----------------------------

@dataclass
class Inputs:
    experiment_name: str
    data_loader_configs: List[str]
    readability_config_path: str
    writer_config_path: str = "configs/writer/sqlite.json"
    run_wordnet: bool = True
    run_llm: bool = True


# ----------------------------
# Experiment folder
# ----------------------------

def _safe_name(s: str) -> str:
    s = s.strip()
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

def make_experiment_dir(experiment_name: str) -> Path:
    ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H%M%S")
    root = Path("demo/readability/experiments")
    out = root / f"{ts}__{_safe_name(experiment_name)}"
    (out / "figs").mkdir(parents=True, exist_ok=True)
    return out


# ----------------------------
# DQResult -> DataFrame (minimal, robust)
# ----------------------------

def dqresults_to_df(results: List[DQResult]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "mesTime": getattr(r, "mesTime", None),
                "DQdimension": getattr(r, "DQdimension", None),
                "DQmetric": getattr(r, "DQmetric", None),
                "DQgranularity": getattr(r, "DQgranularity", None),
                "DQvalue": getattr(r, "DQvalue", None),
                "DQexplanation": getattr(r, "DQexplanation", None),
                "tableName": getattr(r, "tableName", None),
                "columnNames": getattr(r, "columnNames", None),
                "dataset": getattr(r, "dataset", None),
            }
        )

    df = pd.DataFrame(rows)

    # types
    df["DQvalue"] = pd.to_numeric(df["DQvalue"], errors="coerce")
    df["mesTime"] = pd.to_datetime(df["mesTime"], errors="coerce")

    # normalize explanation to dict
    def safe_json(x: Any) -> Dict[str, Any]:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return {}
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return {}
            try:
                return json.loads(s)
            except Exception:
                return {}
        return {}

    df["explain"] = df["DQexplanation"].apply(safe_json)

    # normalize first column name
    def normalize_column(x: Any) -> Optional[str]:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, list):
            return str(x[0]) if x else None
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return None
            if s.startswith("["):
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list) and arr:
                        return str(arr[0])
                except Exception:
                    return None
            return s
        return None

    df["column"] = df["columnNames"].apply(normalize_column)

    # metric family: only "wordnet" or "llm"
    def metric_family(m: Any) -> Optional[str]:
        ms = str(m).strip().lower()
        if ms in ("wordnet", "llm"):
            return ms
        if "wordnet" in ms and "llm" not in ms:
            return "wordnet"
        if "llm" in ms:
            return "llm"
        return None

    df["metric_family"] = df["DQmetric"].apply(metric_family)

    # keep only readability
    df = df[df["DQdimension"].astype(str).str.lower() == "readability"].copy()
    return df


# ----------------------------
# Extractors
# ----------------------------

def pick_latest(df: pd.DataFrame, group_cols: Tuple[str, ...]) -> pd.DataFrame:
    d = df.copy()
    if d["mesTime"].notna().any():
        d = d.sort_values("mesTime")
    else:
        d = d.reset_index(drop=True)
    return d.groupby(list(group_cols), as_index=False).tail(1)

def extract_table_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["DQgranularity"].astype(str).str.lower() == "table") & (df["metric_family"].notna())].copy()
    d = pick_latest(d, ("dataset", "tableName", "metric_family", "DQgranularity"))
    return d[["dataset", "tableName", "metric_family", "DQvalue", "explain"]].copy()

def extract_schema_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["DQgranularity"].astype(str).str.lower() == "schema") & (df["metric_family"].notna())].copy()
    d = pick_latest(d, ("dataset", "tableName", "metric_family", "DQgranularity"))
    return d[["dataset", "tableName", "metric_family", "DQvalue", "explain"]].copy()

def extract_column_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["DQgranularity"].astype(str).str.lower() == "column") & (df["metric_family"].notna())].copy()
    d = pick_latest(d, ("dataset", "tableName", "column", "metric_family", "DQgranularity"))
    return d[["dataset", "tableName", "column", "metric_family", "DQvalue", "explain"]].copy()

def extract_llm_token_share_table(df: pd.DataFrame) -> pd.DataFrame:
    t = df[(df["metric_family"] == "llm") & (df["DQgranularity"].astype(str).str.lower() == "table")].copy()
    t = pick_latest(t, ("dataset", "tableName", "metric_family", "DQgranularity"))
    t["llm_share"] = t["explain"].apply(lambda d: d.get("llm_tokens_share_total", None))
    out = t[["dataset", "tableName", "llm_share"]].copy()
    out["llm_share"] = pd.to_numeric(out["llm_share"], errors="coerce")
    return out.dropna(subset=["llm_share"])


# ----------------------------
# Plot helpers
# ----------------------------

def save_fig(path: Path) -> None:
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()

def plot_table_wordnet_vs_llm(table_scores: pd.DataFrame, out_dir: Path) -> None:
    if table_scores.empty:
        return

    for (ds, tn), g in table_scores.groupby(["dataset", "tableName"]):
        piv = g.pivot_table(index=["dataset", "tableName"], columns="metric_family", values="DQvalue", aggfunc="first")
        if "wordnet" not in piv.columns or "llm" not in piv.columns:
            continue

        w = float(piv.loc[(ds, tn), "wordnet"])
        l = float(piv.loc[(ds, tn), "llm"])
        uplift = l - w

        plt.figure()
        plt.bar(["wordnet", "llm"], [w, l])
        plt.ylim(0, 1)
        plt.title(f"Table Content Readability: WordNet vs Hybrid\n{tn}")
        plt.ylabel("Readability score")
        plt.text(0.5, max(w, l), f"uplift={uplift:.3f}", ha="center", va="bottom")

        save_fig(out_dir / f"01_table_wordnet_vs_llm__{_safe_name(tn)}.png")

def plot_column_wordnet_vs_llm(col_scores: pd.DataFrame, out_dir: Path) -> None:
    """
    Column-Level Content Readability: WordNet vs Hybrid (LLM fallback).
    Uplift (Hybrid - WordNet) is annotated above each column.
    """
    if col_scores.empty:
        return

    for (ds, tn), g in col_scores.groupby(["dataset", "tableName"]):
        piv = g.pivot_table(index="column", columns="metric_family", values="DQvalue", aggfunc="first").sort_index()
        if "wordnet" not in piv.columns or "llm" not in piv.columns:
            continue

        piv["uplift"] = piv["llm"] - piv["wordnet"]
        piv = piv.sort_values("uplift", ascending=False)

        x = list(range(len(piv)))
        width = 0.38

        plt.figure(figsize=(max(7, len(piv) * 1.0), 4.2))
        plt.bar([i - width/2 for i in x], piv["wordnet"].values, width=width, label="WordNet (content)")
        plt.bar([i + width/2 for i in x], piv["llm"].values, width=width, label="Hybrid (WordNet + LLM fallback)")
        plt.ylim(0, 1)
        plt.xticks(x, piv.index.tolist(), rotation=45, ha="right")
        plt.ylabel("Readability score")
        plt.title(
            "Column-Level Content Readability: WordNet vs Hybrid (LLM fallback)\n"
            f"{tn}\n"
            "Annotated Δ values show uplift (Hybrid − WordNet)."
        )
        plt.legend()

        for i, (w, l, u) in enumerate(zip(piv["wordnet"].values, piv["llm"].values, piv["uplift"].values)):
            plt.text(i, max(w, l) + 0.02, f"Δ={u:.3f}", ha="center", va="bottom", fontsize=9)

        save_fig(out_dir / f"02_columns_wordnet_vs_llm__{_safe_name(tn)}.png")

def plot_llm_token_share_pie(llm_share: pd.DataFrame, out_dir: Path) -> None:
    """
    Token Coverage (Table): WordNet-known vs LLM-evaluated.
    """
    if llm_share.empty:
        return

    for (ds, tn), g in llm_share.groupby(["dataset", "tableName"]):
        share = float(g["llm_share"].iloc[0])
        share = max(0.0, min(1.0, share))
        known = 1.0 - share

        plt.figure()
        plt.pie([known, share], labels=["WordNet-known", "LLM-evaluated"], autopct="%.1f%%", startangle=90)
        plt.title(f"Token Coverage (Table): WordNet-known vs LLM-evaluated\n{tn}")
        save_fig(out_dir / f"03_llm_token_share_table__{_safe_name(tn)}.png")

def plot_schema_vs_content(schema_scores: pd.DataFrame, table_scores: pd.DataFrame, out_dir: Path) -> None:
    if schema_scores.empty or table_scores.empty:
        return

    merged = schema_scores.merge(
        table_scores,
        on=["dataset", "tableName", "metric_family"],
        suffixes=("_schema", "_table"),
        how="inner",
    )
    if merged.empty:
        return

    for (ds, tn), g in merged.groupby(["dataset", "tableName"]):
        order = [x for x in ["wordnet", "llm"] if x in g["metric_family"].tolist()]
        g = g.set_index("metric_family").reindex(order).reset_index()

        labels = g["metric_family"].astype(str).tolist()
        schema_vals = g["DQvalue_schema"].astype(float).tolist()
        table_vals = g["DQvalue_table"].astype(float).tolist()

        x = range(len(labels))
        width = 0.4

        plt.figure()
        plt.bar([i - width/2 for i in x], schema_vals, width=width, label="schema")
        plt.bar([i + width/2 for i in x], table_vals, width=width, label="content(table)")
        plt.ylim(0, 1)
        plt.xticks(list(x), labels)
        plt.title(f"Schema vs Content Readability\n{tn}")
        plt.ylabel("Readability score")
        plt.legend()
        save_fig(out_dir / f"04_schema_vs_content__{_safe_name(tn)}.png")


# ----------------------------
# Report (PDF) from PNGs
# ----------------------------

def build_pdf_report(
    out_pdf: Path,
    experiment_name: str,
    run_ts: str,
    dataset_configs: List[str],
    readability_config_path: str,
    summary: Dict[str, Any],
    figs_dir: Path,
) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        leftMargin=2*cm,
        rightMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
    )

    story: List[Any] = []
    story.append(Paragraph("<b>Readability Experiment Report</b>", styles["Title"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"<b>Experiment:</b> {experiment_name}", styles["Normal"]))
    story.append(Paragraph(f"<b>Timestamp:</b> {run_ts} (Europe/Berlin)", styles["Normal"]))
    story.append(Paragraph(f"<b>Data configs:</b> {', '.join(dataset_configs)}", styles["Normal"]))
    story.append(Paragraph(f"<b>Readability config:</b> {readability_config_path}", styles["Normal"]))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
    for k, v in summary.items():
        story.append(Paragraph(f"{k}: {v}", styles["Normal"]))
    story.append(Spacer(1, 0.6*cm))

    story.append(Paragraph("<b>Figures</b>", styles["Heading2"]))
    story.append(Spacer(1, 0.2*cm))

    # include only our 4 core figures by prefix
    keep_prefixes = ("01_", "02_", "03_", "04_")
    pngs = sorted([p for p in figs_dir.glob("*.png") if p.name.startswith(keep_prefixes)])

    if not pngs:
        story.append(Paragraph("No figures were generated.", styles["Normal"]))
        doc.build(story)
        return

    caption_map = {
        "01_table_wordnet_vs_llm": (
            "Table-Level Content Readability (WordNet vs Hybrid). "
            "Shows aggregated table-level content readability and the uplift gained by LLM fallback."
        ),
        "02_columns_wordnet_vs_llm": (
            "Column-Level Content Readability (WordNet vs Hybrid). "
            "Bars compare WordNet-only vs Hybrid (WordNet+LLM fallback) per column. "
            "Annotated Δ values represent uplift (Hybrid − WordNet) per column."
        ),
        "03_llm_token_share_table": (
            "Token Coverage at Table Level. "
            "Pie chart shows the share of tokens handled by WordNet vs tokens requiring LLM evaluation."
        ),
        "04_schema_vs_content": (
            "Schema vs Content Readability. "
            "Compares label readability (schema granularity) against content readability (table granularity)."
        ),
    }

    max_w = A4[0] - (doc.leftMargin + doc.rightMargin)
    max_h = A4[1] - (doc.topMargin + doc.bottomMargin) - 4*cm

    for i, p in enumerate(pngs, start=1):
        key = p.stem.split("__", 1)[0]
        caption = caption_map.get(key, p.stem.replace("__", " - "))

        story.append(Paragraph(f"<b>Figure {i}.</b> {caption}", styles["Normal"]))
        story.append(Spacer(1, 0.2*cm))

        img = RLImage(str(p))
        iw, ih = img.imageWidth, img.imageHeight
        scale = min(max_w / iw, max_h / ih)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        story.append(img)
        story.append(Spacer(1, 0.6*cm))

        if i < len(pngs):
            story.append(PageBreak())

    doc.build(story)



# ----------------------------
# Save artifacts + summary
# ----------------------------

def save_artifacts(out_dir: Path, df: pd.DataFrame, config_snapshot: Dict[str, Any]) -> None:
    df.to_csv(out_dir / "dqresults.csv", index=False, encoding="utf-8")
    with open(out_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(config_snapshot, f, ensure_ascii=False, indent=2)

def compute_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    t = extract_table_scores(df)
    s = extract_schema_scores(df)

    def get_val(d: pd.DataFrame, fam: str) -> Optional[float]:
        x = d[d["metric_family"] == fam]
        if x.empty:
            return None
        return float(x["DQvalue"].iloc[0])

    summary["table_wordnet"] = get_val(t, "wordnet")
    summary["table_llm"] = get_val(t, "llm")
    if summary["table_wordnet"] is not None and summary["table_llm"] is not None:
        summary["table_uplift"] = float(summary["table_llm"] - summary["table_wordnet"])

    summary["schema_wordnet"] = get_val(s, "wordnet")
    summary["schema_llm"] = get_val(s, "llm")
    if summary["schema_wordnet"] is not None and summary["schema_llm"] is not None:
        summary["schema_uplift"] = float(summary["schema_llm"] - summary["schema_wordnet"])

    share = extract_llm_token_share_table(df)
    if not share.empty:
        summary["llm_tokens_share_total"] = float(share["llm_share"].iloc[0])

    return summary


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--data-configs", nargs="+", required=True)
    ap.add_argument("--readability-config", default="configs/metric/readability.json")
    ap.add_argument("--writer-config", default="configs/writer/sqlite.json")
    ap.add_argument("--no-wordnet", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    inp = Inputs(
        experiment_name=args.experiment_name,
        data_loader_configs=args.data_configs,
        readability_config_path=args.readability_config,
        writer_config_path=args.writer_config,
        run_wordnet=not args.no_wordnet,
        run_llm=not args.no_llm,
    )

    if not inp.run_wordnet and not inp.run_llm:
        raise ValueError("Nothing to run: enable wordnet and/or llm.")

    out_dir = make_experiment_dir(inp.experiment_name)
    figs_dir = out_dir / "figs"

    config_snapshot = {
        "experiment_name": inp.experiment_name,
        "data_loader_configs": inp.data_loader_configs,
        "readability_config": inp.readability_config_path,
        "writer_config": inp.writer_config_path,
        "run_wordnet": inp.run_wordnet,
        "run_llm": inp.run_llm,
    }

    orchestrator = DQOrchestrator(writer_config_path=inp.writer_config_path)
    orchestrator.load(data_loader_configs=inp.data_loader_configs)

    metrics: List[str] = []
    metric_configs: List[str] = []

    if inp.run_wordnet:
        metrics.append("readability_wordnet")
        metric_configs.append(inp.readability_config_path)

    if inp.run_llm:
        metrics.append("readability_llm")
        metric_configs.append(inp.readability_config_path)

    results = orchestrator.assess(metrics=metrics, metric_configs=metric_configs)
    df = dqresults_to_df(results)

    save_artifacts(out_dir, df, config_snapshot)

    summary = compute_summary(df)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    table_scores = extract_table_scores(df)
    schema_scores = extract_schema_scores(df)
    col_scores = extract_column_scores(df)
    llm_share = extract_llm_token_share_table(df)

    # 4 core figures
    if inp.run_wordnet and inp.run_llm:
        plot_table_wordnet_vs_llm(table_scores, figs_dir)
        plot_column_wordnet_vs_llm(col_scores, figs_dir)

    if inp.run_llm:
        plot_llm_token_share_pie(llm_share, figs_dir)

    plot_schema_vs_content(schema_scores, table_scores, figs_dir)

    # PDF report (PNGs + captions)
    run_ts = out_dir.name.split("__", 1)[0]
    pdf_path = out_dir / "report.pdf"

    if _HAS_REPORTLAB:
        build_pdf_report(
            out_pdf=pdf_path,
            experiment_name=inp.experiment_name,
            run_ts=run_ts,
            dataset_configs=inp.data_loader_configs,
            readability_config_path=inp.readability_config_path,
            summary=summary,
            figs_dir=figs_dir,
        )
        print(f"[OK] PDF report: {pdf_path}")
    else:
        print("[WARN] reportlab not available in this python environment -> skipping PDF report generation")

    print(f"[OK] Experiment saved to: {out_dir}")


if __name__ == "__main__":
    main()