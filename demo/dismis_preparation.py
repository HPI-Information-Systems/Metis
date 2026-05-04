from pathlib import Path

from metis.dismis.preparation.generate_example_dmvs import generate_example_dmvs
from metis.dismis.preparation.precompute_detection_example_embeddings import (
    precompute_detection_example_embeddings,
)
from metis.dismis.preparation.precompute_value_embeddings import (
    precompute_value_embeddings,
)

generate_example_dmvs(
    Path("data/weather.csv"),
    "qwen3:8b",
    "http://localhost:11434/v1/",
    "placeholder",
    "weather_example_dmvs_detection",
)

precompute_detection_example_embeddings(
    model_name="qwen3-embedding:8b",
    llm_base_url="http://localhost:11434/v1/",
    llm_api_key="placeholder",
    json_files="data/weather_example_dmvs_detection.json",
)

precompute_value_embeddings(
    model_name="qwen3-embedding:8b",
    llm_base_url="http://localhost:11434/v1/",
    llm_api_key="placeholder",
    datasets_and_types=("data/weather.csv", "data/weather_types.json"),
)
