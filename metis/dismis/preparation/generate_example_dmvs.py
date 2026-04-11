import random
import pandas as pd
import json
import os
import argparse
import pandas as pd
from pathlib import Path

from openai_LLM import OpenAILLM as LLM
from pollution.errors.LLMplaceholder import LLMPlaceholderDMV2, LLMCommentDMV2, LLMUnsureDMV2, LLMValidDMV2


def get_csv_files(path_str):
    path = Path(path_str)

    if path.is_file() and path.suffix.lower() == ".csv" and not path.name.endswith("labels.csv"):
        return [path]

    elif path.is_dir():
        return list(path.rglob("*.csv"))

    else:
        return []

def collect_unique_columns(path_str):
    csv_files = get_csv_files(path_str)
    all_columns = set()

    for f in csv_files:
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        all_columns.update(cols)

    return sorted(all_columns)

def main(dataset_path, llm_name, dataset_name, output_file_name):
    llm = LLM(model_name=llm_name, port=8000)

    DMV_types = {
        "placeholder": LLMPlaceholderDMV2(llm, table_name=dataset_name),
        "comments": LLMCommentDMV2(llm, table_name=dataset_name),
        "unsure": LLMUnsureDMV2(llm, table_name=dataset_name),
        "valid": LLMValidDMV2(llm, table_name=dataset_name),
    }
    dataset_identifier = dataset_path.strip("/").split("/")[-1]

    with open(os.path.join(dataset_path, f"{dataset_identifier}_types.json"), "r") as f:
        types = json.load(f)

    unique_columns = [col for col in types.keys() if types[col] in ["categorical", "text"]]
    dataset = pd.read_csv(os.path.join(dataset_path, f"{dataset_identifier}.csv"), keep_default_na=False, na_values=[""])

    columns = dataset.columns.to_list()
    example_values = {}
    for col in columns:
        unique_values = dataset[col][:10000].dropna().astype(str).tolist()
        sampled_values = list(set(random.sample(unique_values, min(20, len(unique_values)))))
        selected_values = sampled_values[:min(5, len(sampled_values))]
        example_values[col] = selected_values
    generated_dmvs = {col: {} for col in unique_columns}

    for dmv_type, dmv_generator in DMV_types.items():
        all_placeholders, valid_values_dict, invalid_values_dict = dmv_generator.get_column_placeholders(unique_columns, example_values=example_values)
        for col in unique_columns:
            placeholders = all_placeholders[col]
            generated_dmvs[col][dmv_type] = placeholders

    with open(os.path.join(dataset_path, f"{output_file_name}.json"), 'w') as f:
        json.dump(generated_dmvs, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMV detection benchmark.")
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset directory or single CSV file')
    parser.add_argument('--llm_name', type=str, required=True, help='LLM model name or path')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--example_file_name', type=str, required=True, help='Name of the output file')
    args = parser.parse_args()

    main(args.dataset, args.llm_name, args.dataset_name, args.example_file_name)
