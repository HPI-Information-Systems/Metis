import matplotlib.pyplot as plt
import os

def plot_wordnet_vs_llm(df, output_dir):
    table_df = df[df["dq_granularity"] == "table"]

    wn = table_df[table_df["dq_metric"].str.contains("wordnet_only", na=False)]
    hybrid = table_df[table_df["dq_metric"].str.contains("wordnetFirst", na=False)]

    if wn.empty or hybrid.empty:
        return None

    wn_val = wn["dq_value"].iloc[-1]
    hybrid_val = hybrid["dq_value"].iloc[-1]

    plt.figure()
    plt.bar(["WordNet Only", "WordNet + LLM"], [wn_val, hybrid_val])
    plt.title("Content Readability Comparison")
    plt.ylabel("Readability Score")

    path = os.path.join(output_dir, "comparison_wordnet_vs_llm.png")
    plt.savefig(path)
    plt.close()

    return path


def plot_llm_token_share(df, output_dir):
    table_df = df[df["dq_metric"].str.contains("wordnetFirst", na=False)]

    if table_df.empty:
        return None

    explanation = table_df.iloc[-1]["dq_explanation"]

    try:
        import json
        explanation = json.loads(explanation)
        share = explanation.get("llm_tokens_share_total", 0)
    except Exception:
        return None

    plt.figure()
    plt.bar(["LLM Token Share"], [share])
    plt.ylim(0, 1)
    plt.title("LLM Token Usage Share")

    path = os.path.join(output_dir, "llm_token_share.png")
    plt.savefig(path)
    plt.close()

    return path
