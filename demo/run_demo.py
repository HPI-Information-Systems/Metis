from metis.dq_orchestrator import DQOrchestrator
from metis.metric import Metric

_METRIC_CONFIGS = {
    "consistency_countFDViolations": "demo/configs/consistency.json",
}

orchestrator = DQOrchestrator(writer_config_path="demo/configs/sqlite.json")
orchestrator.load(data_loader_configs=["data/restaurants.json"])

for metric_name in Metric.registry:
    try:
        orchestrator.assess(
            metrics=[metric_name],
            metric_configs=[_METRIC_CONFIGS.get(metric_name)],
        )
    except Exception as exc:
        print(f"Metric {metric_name} failed: {exc}")
