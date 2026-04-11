import argparse
import json
from pathlib import Path
from typing import List

from metis.dismis.preparation.openai_LLM import OpenAIEmbedding
from metis.utils.logging import logger as main_logger

logger = main_logger.getChild(__name__)


def precompute_detection_example_embeddings(
    model_name: str, llm_base_url: str, llm_api_key: str, json_files: str | List[str]
):
    trunc = 512
    model = OpenAIEmbedding(
        model_name=model_name, base_url=llm_base_url, api_key=llm_api_key
    )

    if isinstance(json_files, str):
        json_files = [json_files]

    # Collect all unique texts from all lists in all files
    for file_idx, file in enumerate(json_files, 1):
        logger.info(f"Processing file {file_idx}/{len(json_files)}: {file}")
        all_texts = set()
        with open(file, "r") as f:
            data = json.load(f)
            for lists in data.values():
                for values in lists.values():
                    all_texts.update([str(v) for v in values])

        all_texts = sorted(all_texts)
        logger.info(f"Total unique texts to embed: {len(all_texts)}")

        # Compute embeddings in batches (if needed)
        embeddings = {}
        outputs = model.embed(all_texts)
        for text, output in zip(all_texts, outputs):
            embeddings[text] = output[:trunc]

        # Save as precomputed_example_embeddings.json
        output_path = Path(file).parent / "precomputed_example_embeddings.json"
        with open(output_path, "w") as f:
            json.dump(embeddings, f, indent=2)

        logger.info(f"Saved embeddings for {len(embeddings)} texts to {output_path}")


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
    parser.add_argument("--json_files", type=str, nargs="+", required=True)
    args = parser.parse_args()

    precompute_detection_example_embeddings(
        args.model, args.llm_base_url, args.llm_api_key, args.json_files
    )
