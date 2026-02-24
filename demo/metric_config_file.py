from metis.dq_orchestrator import DQOrchestrator

orchestrator = DQOrchestrator("configs/writer/sqlite.json")

orchestrator.load(data_loader_configs=["data/countries-capitals.json"])

# Only the metric Consistency needs a config file
orchestrator.assess(
    metrics=["readability_llm", "readability_wordnet"],
    metric_configs=["configs/metric/readability.json", "configs/metric/readability.json"],
)