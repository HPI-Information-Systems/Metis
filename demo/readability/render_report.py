from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _find_latest_db_export_run(base: Path) -> Path:
    runs = sorted(base.glob("*_db_export"), reverse=True)
    if not runs:
        raise FileNotFoundError(f"No *_db_export run found in: {base}")
    return runs[0]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if len(df) > max_rows:
        df = df.head(max_rows).copy()
    return df.to_html(index=False, escape=False)


def main() -> None:
    base = Path("demo/readability/runs")
    run_dir = _find_latest_db_export_run(base)

    summary_path = run_dir / "summary_table.json"
    cols_path = run_dir / "columns_comparison.csv"
    figs_dir = run_dir / "figs"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not cols_path.exists():
        raise FileNotFoundError(cols_path)

    summary = _load_json(summary_path)
    cols = pd.read_csv(cols_path)

    # Useful derived values
    content_wn = summary.get("wordnet_content_mean")
    content_llm = summary.get("llm_content_mean")
    schema_wn = summary.get("wordnet_schema_mean")
    schema_llm = summary.get("llm_schema_mean")

    uplift_content = summary.get("uplift_content_mean")
    uplift_schema = summary.get("uplift_schema_mean")

    llm_share_total = summary.get("llm_tokens_share_total")
    llm_tokens_total = summary.get("llm_tokens_count_total")
    unique_tokens_total = summary.get("unique_tokens_count_total")

    model_id = summary.get("hf_model_id")
    llm_mode = summary.get("llm_mode")
    use_llm_fallback = summary.get("use_llm_fallback")

    # Top tables
    cols_sorted_uplift = cols.sort_values("uplift", ascending=False).copy()
    cols_sorted_share = cols.dropna(subset=["llm_tokens_share"]).sort_values("llm_tokens_share", ascending=False).copy()

    # Figures (if exist)
    fig_table_scores = figs_dir / "table_scores.png"
    fig_uplift = figs_dir / "top_column_uplift.png"
    fig_share = figs_dir / "top_llm_token_share.png"

    def img_tag(p: Path, title: str) -> str:
        if not p.exists():
            return f"<p><i>Missing figure: {p.name}</i></p>"
        return f"<h3>{title}</h3><img src='figs/{p.name}' style='max-width:100%; border:1px solid #ddd; border-radius:8px;'/>"

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Readability Report (DQ4AI-style)</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.4; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 18px; }}
    h1,h2 {{ margin: 0 0 10px 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 14px; }}
    th {{ background: #f5f5f5; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }}
  </style>
</head>
<body>

<h1>Readability Report (DQ4AI-style)</h1>
<p class="mono">Run folder: {run_dir.as_posix()}</p>

<div class="card">
  <h2>Configuration</h2>
  <ul>
    <li><b>use_llm_fallback</b>: {use_llm_fallback}</li>
    <li><b>llm_mode</b>: {llm_mode}</li>
    <li><b>hf_model_id</b>: {model_id}</li>
  </ul>
</div>

<div class="card">
  <h2>Table-level Summary</h2>
  <div class="grid">
    <div>
      <h3>Content</h3>
      <ul>
        <li><b>WordNet</b>: {content_wn}</li>
        <li><b>WordNet+LLM fallback</b>: {content_llm}</li>
        <li><b>Uplift</b>: {uplift_content}</li>
      </ul>
    </div>
    <div>
      <h3>Schema</h3>
      <ul>
        <li><b>WordNet</b>: {schema_wn}</li>
        <li><b>WordNet+LLM fallback</b>: {schema_llm}</li>
        <li><b>Uplift</b>: {uplift_schema}</li>
      </ul>
    </div>
  </div>

  <h3>LLM Usage</h3>
  <ul>
    <li><b>llm_tokens_count_total</b>: {llm_tokens_total}</li>
    <li><b>unique_tokens_count_total</b>: {unique_tokens_total}</li>
    <li><b>llm_tokens_share_total</b>: {llm_share_total}</li>
  </ul>
</div>

<div class="card">
  <h2>Plots</h2>
  {img_tag(fig_table_scores, "Table scores (WordNet vs Hybrid)")}
  {img_tag(fig_uplift, "Top column uplift (Hybrid - WordNet)")}
  {img_tag(fig_share, "Top LLM token share per column")}
</div>

<div class="card">
  <h2>Column-level Comparison (Top uplift)</h2>
  {_df_to_html_table(cols_sorted_uplift[["table_name","column_name_norm","wordnet_score","llm_score","uplift","llm_tokens_share"]], max_rows=30)}
</div>

<div class="card">
  <h2>Column-level LLM Token Share (Top)</h2>
  {_df_to_html_table(cols_sorted_share[["table_name","column_name_norm","llm_tokens_count","unique_tokens_count","llm_tokens_share"]], max_rows=30)}
</div>

</body>
</html>
"""

    out_html = run_dir / "report.html"
    out_html.write_text(html, encoding="utf-8")
    print("OK - wrote:", out_html)

    # Optional PDF (only if reportlab available)
        # Optional PDF (only if reportlab available)
    try:
        from .pdf_report import create_pdf  # type: ignore

        summary_text = (
            f"Content WordNet: {content_wn}\n"
            f"Content Hybrid: {content_llm}\n"
            f"Content Uplift: {uplift_content}\n\n"
            f"Schema WordNet: {schema_wn}\n"
            f"Schema Hybrid: {schema_llm}\n"
            f"Schema Uplift: {uplift_schema}\n\n"
            f"LLM share total: {llm_share_total}\n"
            f"LLM tokens total: {llm_tokens_total}\n"
            f"Unique tokens total: {unique_tokens_total}\n"
            f"Model: {model_id} | Mode: {llm_mode} | use_llm_fallback: {use_llm_fallback}\n"
        )

        out_pdf = run_dir / "report.pdf"
        create_pdf(run_dir, out_pdf, summary_text)
        print("OK - wrote:", out_pdf)
    except Exception as e:
        print("SKIP - PDF not generated:", repr(e))



if __name__ == "__main__":
    main()
