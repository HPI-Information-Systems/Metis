import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from metis.dismis.preparation.openai_LLM import OpenAIEmbedding
from metis.utils.logging import logger as main_logger

logger = main_logger.getChild(__name__)


def precompute_value_embeddings(
    model_name: str,
    llm_base_url: str,
    llm_api_key: str,
    datasets_and_types: Tuple[str, str] | List[Tuple[str, str]],
):
    trunc = 512
    model = OpenAIEmbedding(
        model_name=model_name, base_url=llm_base_url, api_key=llm_api_key
    )

    if isinstance(datasets_and_types, tuple):
        datasets_and_types = [datasets_and_types]

    for dataset_idx, (dataset, types) in enumerate(datasets_and_types, 1):
        logger.info(
            f"Processing dataset {dataset_idx}/{len(datasets_and_types)}: {dataset}"
        )
        dataset_file = Path(dataset)
        types_file = Path(types)

        if dataset_file.suffix != ".csv":
            raise ValueError(
                f"Expected CSV file, got {dataset_file.suffix} for {dataset_file}"
            )

        if not types_file.exists():
            raise FileNotFoundError(
                f"Types file not found. Expected column types at {types_file}."
            )
        with open(types_file, "r") as f:
            types = json.load(f)

        text_columns = [col for col in types if types[col] in ["text", "categorical"]]
        if len(text_columns) == 0:
            logger.debug(f"No text or categorical columns found in {dataset}")
            continue

        logger.info(f"Text columns: {text_columns}")
        unique_values = {col: set() for col in text_columns}

        df = pd.read_csv(dataset_file, keep_default_na=False, na_values=[""])
        for col in text_columns:
            unique_column_values = [
                str(val) for val in df[col].dropna().unique().tolist()
            ]

            unique_values[col].update(unique_column_values)

        for col in unique_values.keys():
            logger.debug(f"Column '{col}' has {len(unique_values[col])} unique values.")

        embeddings = {col: {} for col in text_columns}
        for col in unique_values:
            outputs = model.embed(list(unique_values[col]))
            embeddings[col] = {
                val: o[:trunc] for val, o in zip(unique_values[col], outputs)
            }

        with open(
            dataset_file.parent / f"{dataset_file.stem}_value_embeddings.json", "w"
        ) as f:
            json.dump(embeddings, f, indent=4)
            logger.info(
                f"Saved embeddings to {dataset_file.parent / f'{dataset_file.stem}_value_embeddings.json'}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute embeddings for example DMVs"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Embedding-8B",
        help="Model name for embedding",
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default="http://localhost:11434/v1/",
        help="Base URL for the LLM API",
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default="placeholder",
        help="API key for the LLM (if required)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="List of dataset csv files to process",
    )
    args = parser.parse_args()

    precompute_value_embeddings(
        args.model,
        args.llm_base_url,
        args.llm_api_key,
        [
            (dataset, str(Path(dataset).parent / f"{Path(dataset).stem}_types.json"))
            for dataset in args.datasets
        ],
    )
