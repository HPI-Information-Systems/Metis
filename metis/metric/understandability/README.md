# Understandability Metric

The metric assesses non-empty textual table-cell values in their original
surface form. Numeric columns and schema-level assessment are excluded.

## Shared preprocessing

Tokenization is performed once before pipeline selection. The resulting exact
word occurrences, including capitalization, spelling, punctuation, digits, and
hyphens, are reused by `resource_based`, `llm`, and `hybrid`. Each occurrence has a
deterministic ID based on row, column, and token position.

## Resource-based pipeline

The resource-based estimator calculates four intrinsic criteria:

1. lexical recognizability
2. semantic ambiguity
3. notational clarity
4. lexical processing ease

Princeton WordNet and OdeNet are queried directly for every word-like token.
Automatic language detection is not used. A criterion that is not applicable or
cannot be evaluated remains marked as unavailable in the diagnostics and enters
the fixed four-criterion numerical aggregation as zero.

Cell-level row and column embeddings are used only as conservative contextual
evidence. Cosine similarities from `-0.40` through `0.60` are neutral. With
`context_lambda = 0.10`, context can change the intrinsic score by at most
`±0.05`.

## LLM pipeline

The configured model is `Qwen/Qwen3-4B-Instruct-2507`. The LLM receives the exact
preselected word occurrences together with the unchanged cell value and the
observable row and column context. The prompt describes the assessment criteria
in full and does not request abbreviated feature values.

The model returns one score per supplied word occurrence and one independent
holistic score for the complete cell. The cell score is not derived from the
word scores. The active model output is a minimal flat JSON object containing
only `cell_id`, `cell_score`, and `word_scores` with `word_id` and `score`.

## Hybrid pipeline

The hybrid pipeline calculates the resource-based score first. Under globally fixed
uncertainty conditions, the complete cell is reassessed by the LLM and the
resource-based score is replaced by the independent holistic LLM cell score.
The LLM word scores remain available for analysis but are not aggregated into
the replacement score.

## Validation and missing results

A model response is used only when all expected word IDs occur exactly once and
in the original order, all scores are within `[0, 1]`, and a valid independent
cell score is present. At most one structured retry is issued.

A failed LLM response is represented as unavailable, not as a score of zero. A
hybrid replacement that fails is also unavailable and does not silently revert
to the resource-based score. Table scores are calculated from successfully
scored assessable cells and must be reported together with cell assessment
coverage.

## Table aggregation

The table score is the direct arithmetic mean over successfully scored
assessable cell scores. Each cell contributes at most once. Row and column
aggregates are diagnostic views only.
