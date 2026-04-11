import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd
from openai_LLM import OpenAILLM as LLM
from pollution.errors.LLMplaceholder import (
    LLMCommentDMV2,
    LLMPlaceholderDMV2,
    LLMUnsureDMV2,
    LLMValidDMV2,
)


def generate_example_dmvs(
    dataset_file: Path,
    llm_name: str,
    llm_base_url: str,
    llm_api_key: str,
    output_file_name: str,
):
    llm = LLM(model_name=llm_name, base_url=llm_base_url, api_key=llm_api_key)

    dataset_name = dataset_file.stem
    DMV_types = {
        "placeholder": LLMPlaceholderDMV2(llm, table_name=dataset_name),
        "comments": LLMCommentDMV2(llm, table_name=dataset_name),
        "unsure": LLMUnsureDMV2(llm, table_name=dataset_name),
        "valid": LLMValidDMV2(llm, table_name=dataset_name),
    }

    types_file = dataset_file.parent / f"{dataset_name}_types.json"

    if not types_file.exists():
        raise FileNotFoundError(
            f"Types file not found. Expected column types at {types_file}."
        )
    with open(types_file, "r") as f:
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

    for dmv_type, dmv_generator in DMV_types.items():
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
