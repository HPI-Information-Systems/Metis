from datetime import datetime
import os

from .db_reader import load_readability_results
from .plots import plot_wordnet_vs_llm, plot_llm_token_share
from .pdf_report import create_pdf

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = f"demo/readability/runs/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    df = load_readability_results()

    plot1 = plot_wordnet_vs_llm(df, run_dir)
    plot2 = plot_llm_token_share(df, run_dir)

    summary = f"""
    Total Readability Results: {len(df)}
    """

    pdf_path = os.path.join(run_dir, "readability_report.pdf")
    create_pdf(pdf_path, [plot1, plot2], summary)

    print("Report created:", pdf_path)

if __name__ == "__main__":
    main()
