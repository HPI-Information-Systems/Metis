from metis.dq_orchestrator import DQOrchestrator

# No config file means default to console writer
orchestrator = DQOrchestrator(writer_config_path="configs/writer/sqlite.json")

orchestrator.load(data_loader_configs=["data/restaurant.json"])

orchestrator.assess(metrics=["minimality_clustering"], metric_configs=["configs/metric/minimality.json"])