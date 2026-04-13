from functools import lru_cache

import torch
from vllm import LLM


class VLLMModel:
    def __init__(
        self,
        model_name: str = "/sc/home/philipp.hildebrandt/models/Qwen/Qwen3-Embedding-0.6B",
    ):
        """
        Initialize the VLLMModel with a specific model name.

        Args:
            model_name (str): The name of the model to use.
        """
        self.model = LLM(model=model_name, task="embed", max_seq_len_to_capture=256)

    @lru_cache(maxsize=128)
    def embed(self, texts: tuple) -> torch.Tensor:
        """
        Embed a list of texts using the VLLM model.

        Args:
            texts (list): List of texts to embed.

        Returns:
            torch.Tensor: The embeddings of the input texts.
        """
        outputs = self.model.embed(texts)
        return torch.tensor([o.outputs.embedding for o in outputs]).cpu()
