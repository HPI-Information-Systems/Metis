import json
from metis.dq_orchestrator import DQOrchestrator

orchestrator = DQOrchestrator(writer_config_path="configs/writer/sqlite.json")

orchestrator.load(data_loader_configs=[
    "data/duplicate_detection/2MASS.json",
    "data/duplicate_detection/drives.json",
    "data/duplicate_detection/notebooks1.json",
    "data/duplicate_detection/notebooks2.json",
    "data/duplicate_detection/restaurant.json"
])

similarity_threshold = 0.05

while similarity_threshold < 1:
    similarity_threshold += 0.05
    for use_semhash in [False, True]:
        config = json.dumps({
            "similarity_threshold": similarity_threshold,
            "use_semhash": use_semhash
        })
        orchestrator.assess(
            metrics=["minimality_clustering"],
            metric_configs=[config]
        )
