# Readability Metric

## 1. Purpose of the metric  
The readability metric evaluates the readability of tabular data on two levels:  
- Schema readability – readability of column labels  
- Data Readability – Readability of the actual cell contents  

The implementation combines:  
- WordNet-based lexical evaluation  
- Optional LLM support (fallback or strict)  
- Hierarchical aggregation:  
- Word → Cell → Column → Table  

## 2. Granularities  
The metric generates results (DQResult) with the following DQgranularity values:  

|Granularity|Meaning|  
|--------|--------|
|schema|Aggregated readability of all column names|  
|table|Aggregated readability of all text columns|
|column|Readability per text column|
|cell|Readability of individual cells (optional)|

**Please note: The schema is calculated without weighting synonyms and homonyms. Correct weights can be found in the paper [2019_Ehrlinger](https://personales.upv.es/thinkmind/dl/conferences/dbkda/dbkda_2019/dbkda_2019_1_30_50036.pdf).**  

## 3. Pipeline modes  
The metric supports two implementations:  

### 3.1 WordNet-only
Metric:         readability_wordnet  
Properties:     Evaluates exclusively via WordNet  

### 3.2 WordNet + LLM fallback (hybrid)
Metric:         readability_llm  
Configuration:  “llm_mode”: “fallback”  
Properties:     WordNet is used first, LLM is only used for unknown/difficult tokens, Minimal LLM usage  

### 3.3 LLM Strict
Metric:         readability_llm  
Configuration:  “llm_mode”: “strict”  
Properties:     LLM dominates the evaluation, Higher computational effort  

## Experiments
