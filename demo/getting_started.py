from metis.dq_orchestrator import DQOrchestrator

# No config file means default to console writer
orchestrator = DQOrchestrator(writer_config_path="configs/writer/sqlite.json")

orchestrator.load(data_loader_configs=["data/adult.json"])

orchestrator.assess(metrics=["completeness_nullRatio"], metric_configs=['{"measure_runtime": true}'])
orchestrator.assess(metrics=["minimality_duplicateCount"], metric_configs=[None])
orchestrator.assess(metrics=["validity_outOfVocabulary"], metric_configs=['{"use_nltk": true, "lowercase": true}'])