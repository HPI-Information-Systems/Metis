from metis.dq_orchestrator import DQOrchestrator

orchestrator = DQOrchestrator(db_name="dq_repository/dq_repository.db", result_table_name="dq_results")

orchestrator.load(data_loader_configs=["data/adult.json"])

orchestrator.assess(metrics=["Completeness"], metric_configs=[None])
