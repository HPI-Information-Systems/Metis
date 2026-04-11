from vllm import LLM
import argparse
import os
import json
import pandas as pd
import glob

"data/polluted/*/*"
def main(model_name, datasets):
    model = LLM(model=model_name, task="embed")
    trunc=512

    if isinstance(datasets, str):
        datasets = [datasets]

    datasets = [dir for dir in datasets if os.path.isdir(dir)]

    for dataset in datasets:
        dataset_name = dataset.split("/")[-1]

        print("Dataset:", dataset)
        with open(dataset+f"/{dataset_name}_types.json", 'r') as f:
            types = json.load(f)

        text_columns = [col for col in types if types[col] in ["text", "categorical"]]
        if len(text_columns) == 0:
            continue

        print("Text columns:", text_columns)
        unique_values = {col: set() for col in text_columns}

        df = pd.read_csv(dataset + f"/{dataset_name}.csv", keep_default_na=False, na_values=[""])
        for col in text_columns:
            unique_column_values = [str(val) for val in df[col].dropna().unique().tolist()]

            unique_values[col].update(unique_column_values)

        for col in unique_values.keys():
            print(f"Column '{col}' has {len(unique_values[col])} unique values.")

        embeddings = {col: {} for col in text_columns}
        for col in unique_values:
            outputs = model.embed(list(unique_values[col]))
            embeddings[col] = {val: o.outputs.embedding[:trunc] for val, o in zip(unique_values[col], outputs)}

        with open(dataset+f"/{dataset_name}_value_embeddings.json", 'w') as f:
            json.dump(embeddings, f, indent=4)
            print(f"Saved embeddings to {dataset+f'/{dataset_name}_value_embeddings.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute embeddings for example DMVs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-8B",
                            help="Model name for embedding")
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    args = parser.parse_args()

    main(args.model, args.datasets)
