from metis.dq_orchestrator import DQOrchestrator

orchestrator = DQOrchestrator()

orchestrator.load(data_loader_configs=["data/countries-capitals.json"])

# Only the metric Consistency needs a config file
orchestrator.assess(
    metrics=["completeness_nullRatio", "consistency_countFDViolations", "readability_wordnet", "readability_llm"],
    metric_configs=["", "configs/metric/consistency.json", "configs/metric/readability.json",  "configs/metric/readability.json"]
)