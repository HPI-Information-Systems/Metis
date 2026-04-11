
import json
from vllm import LLM
import argparse

def main(model_name, json_files):
    model = LLM(model=model_name, task="embed")
    trunc = 512

    if isinstance(json_files, str):
        json_files = [json_files]

    # Collect all unique texts from all lists in all files
    for file in json_files:
        all_texts = set()
        with open(file, "r") as f:
            data = json.load(f)
            for col, lists in data.items():
                for key, values in lists.items():
                    all_texts.update([str(v) for v in values])

        all_texts = sorted(all_texts)
        print(f"Total unique texts to embed: {len(all_texts)}")

        # Compute embeddings in batches (if needed)
        embeddings = {}
        outputs = model.embed(all_texts)
        for text, output in zip(all_texts, outputs):
            embeddings[text] = output.outputs.embedding[:trunc]

        # Save as precomputed_example_embeddings.json
        with open(file.replace("example_dmvs_detection.json", f"precomputed_example_embeddings.json"), "w") as f:
            json.dump(embeddings, f, indent=2)

        print(f"Saved embeddings for {len(embeddings)} texts to {file.replace('example_dmvs_detection.json', f'precomputed_example_embeddings.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute embeddings for example DMVs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-8B",
                            help="Model name for embedding")
    parser.add_argument("--json_files", type=str, nargs="+", required=True)
    args = parser.parse_args()

    main(args.model, args.json_files)
