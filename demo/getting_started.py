from metis.dq_orchestrator import DQOrchestrator
from metis.metric.accuracy.accuracy_syntacticDomain_config import (
    accuracy_syntacticDomain_config,
)

# No config file means default to console writer
orchestrator = DQOrchestrator(writer_config_path="configs/writer/sqlite.json")

orchestrator.load(data_loader_configs=["data/adult.json"])

orchestrator.assess(metrics=["completeness_nullRatio"], metric_configs=[""])
orchestrator.assess(metrics=["minimality_duplicateCount"], metric_configs=[None])
orchestrator.assess(
    metrics=["validity_outOfVocabulary"],
    metric_configs=['{"use_nltk": true, "lowercase": true}'],
)

orchestrator.assess(
    metrics=["accuracy_dataRange"],
    metric_configs=['{"intervals": {"age": [17, 90]}, "fallback": "profiling"}'],
)

orchestrator.assess(
    metrics=["accuracy_outlierRisk"],
    metric_configs=[None],
)

orchestrator.assess(
    metrics=["accuracy_syntacticDomain"],
    # adult.csv preserves leading spaces in categorical values
    metric_configs=[accuracy_syntacticDomain_config(
        method="exact_match",
        domains={
            "workclass": [
                " Private", " Self-emp-not-inc", " Self-emp-inc",
                " Federal-gov", " Local-gov", " State-gov",
                " Without-pay", " Never-worked",
            ],
        },
    )],
)

# Second orchestrator uses ConsoleWriter to avoid re-initialising the Database singleton
ref_orchestrator = DQOrchestrator()
ref_orchestrator.load(data_loader_configs=["demo/configs/adult_with_reference.json"])
ref_orchestrator.assess(
    metrics=["accuracy_semanticReference"],
    metric_configs=[None],
)
