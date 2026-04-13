import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Literal

import pandas as pd

from metis.dismis.preparation.openai_LLM import OpenAILLM
from metis.dismis.preparation.pollution.errors.error import DMV
from metis.dismis.preparation.pollution.errors.LLMplaceholder import (
    LLMCommentDMV2,
    LLMPlaceholderDMV2,
    LLMUnsureDMV2,
    LLMValidDMV2,
)
from metis.dismis.utils.pathutils import require_exists
from metis.utils.logging import logger as main_logger

logger = main_logger.getChild(__name__)

EXAMPLE_DMV_CATEGORIES = Literal["placeholder", "comments", "unsure", "valid"]


def generate_example_dmvs(
    dataset_file: Path,
    llm_name: str,
    llm_base_url: str,
    llm_api_key: str,
    output_file_name: str,
):
    llm = OpenAILLM(model_name=llm_name, base_url=llm_base_url, api_key=llm_api_key)

    dataset_name = dataset_file.stem
    DMV_types: Dict[EXAMPLE_DMV_CATEGORIES, DMV] = {
        "placeholder": LLMPlaceholderDMV2(llm, table_name=dataset_name),
        "comments": LLMCommentDMV2(llm, table_name=dataset_name),
        "unsure": LLMUnsureDMV2(llm, table_name=dataset_name),
        "valid": LLMValidDMV2(llm, table_name=dataset_name),
    }

    types_file = dataset_file.parent / f"{dataset_name}_types.json"

    with require_exists(types_file, "Column types").open("r") as f:
        types = json.load(f)

    unique_columns = [
        col for col in types.keys() if types[col] in ["categorical", "text"]
    ]
    dataset = pd.read_csv(dataset_file, keep_default_na=False, na_values=[""])

    example_values = {}
    for col in unique_columns:
        unique_values = dataset[col][:10000].dropna().astype(str).tolist()
        sampled_values = list(
            set(random.sample(unique_values, min(20, len(unique_values))))
        )
        selected_values = sampled_values[: min(5, len(sampled_values))]
        example_values[col] = selected_values
    generated_dmvs = {col: {} for col in unique_columns}

    for dmv_idx, (dmv_type, dmv_generator) in enumerate(DMV_types.items(), 1):
        logger.info(f"Generating {dmv_type} DMVs ({dmv_idx}/{len(DMV_types)})")
        all_placeholders, _, _ = dmv_generator.get_column_placeholders(
            unique_columns, example_values=example_values
        )
        for col in unique_columns:
            generated_dmvs[col][dmv_type] = all_placeholders[col]

    with open(os.path.join(dataset_file.parent, f"{output_file_name}.json"), "w") as f:
        json.dump(generated_dmvs, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMV detection benchmark.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset CSV file",
    )
    parser.add_argument(
        "--llm_name", type=str, required=True, help="LLM model name or path"
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
        "--example_file_name", type=str, required=True, help="Name of the output file"
    )
    args = parser.parse_args()

    generate_example_dmvs(
        Path(args.dataset),
        args.llm_name,
        args.llm_base_url,
        args.llm_api_key,
        args.example_file_name,
    )
