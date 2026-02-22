Readability Metric (METIS)

1. Purpose of the metric
The readability metric evaluates the readability of tabular data on two levels:
    Schema readability – readability of column labels
    Data Readability – Readability of the actual cell contents
The implementation combines:
    WordNet-based lexical evaluation
    Optional LLM support (fallback or strict)
Hierarchical aggregation: