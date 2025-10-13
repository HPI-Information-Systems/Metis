from metis.dq_orchestrator import DQOrchestrator

# No config file means default to console writer
orchestrator = DQOrchestrator(writer_config="configs/writer/sqlite.json")

orchestrator.load(data_loader_configs=["data/adult.json"])

orchestrator.assess(metrics=["Completeness"], metric_configs=[None])
orchestrator.assess(metrics=["AttributeUniqueness"], metric_configs=[None])