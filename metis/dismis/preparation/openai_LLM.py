from openai import AsyncOpenAI, OpenAI
from typing import List, Dict
import asyncio
import numpy as np
from collections import OrderedDict

class LRUCache(OrderedDict):
    def __init__(self, capacity: int = 10000):
        super().__init__()
        self.capacity = capacity

    def __getitem__(self, key):
        value = super().__getitem__(key)
        # Only move to end if key still exists (not during eviction)
        if key in self:
            self.move_to_end(key)  # mark as recently used
        return value

    def __setitem__(self, key, value):
        # If key already exists, move it to end before updating
        if key in self:
            self.move_to_end(key)
        # Add the new item
        super().__setitem__(key, value)
        # Evict oldest if over capacity
        if len(self) > self.capacity:
            oldest_key = next(iter(self))  # Get first (oldest) key
            del self[oldest_key]  # Remove it without triggering __getitem__

class OpenAILLM():
    def __init__(self, model_name: str, port: int = 8000):
        self.client = AsyncOpenAI(
            api_key="token-abc123",
            base_url=f"http://localhost:11434/v1/"  # e.g. "http://localhost:11434/v1"
        )
        self.model_name = model_name

    async def fetch_completion(self, chat: List[Dict[str, str]], temperature: float | None = None) -> str:
        # This line will be awaited when the coroutine is executed
        if temperature is None:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=chat,
                timeout=60*60,
                max_tokens=2048,
            )
        else:
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=chat,
                max_tokens=2048,
                temperature=temperature,
                timeout=60*60
            )
        return completion.choices[0].message.content

    async def generate_async(self, chats: List[List[Dict[str, str]]], temperature=None) -> List[str]:
        tasks = [self.fetch_completion(chat, temperature) for chat in chats]
        responses = await asyncio.gather(*tasks)
        return responses

    def generate(self, chats: List[List[Dict[str, str]]]) -> List[str]:
        for _ in range(10):
            try:
                return asyncio.run(self.generate_async(chats))
            except Exception as e:
                print(f"Error during OpenAI API call: {e}. Retrying...")

class OpenAIEmbedding():
    def __init__(self, model_name: str, port: int = 8000):
        self.client = OpenAI(
            api_key="token-abc123",
            base_url=f"http://localhost:11434/v1/"  # e.g. "http://localhost:11434/v1"
        )
        self.model_name = model_name
        self.embeddings_map = {"default": LRUCache(capacity=10000)}

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2-normalize each row in the array."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero
        return vectors / norms

    def embed(self, texts: List[str], column_name: str | None = "default") -> List[float]:
        texts = [str(text) for text in texts]
        if column_name not in self.embeddings_map or len(self.embeddings_map[column_name]) < len(texts):
            self.embeddings_map[column_name] = LRUCache(capacity=max(10000, len(texts) + 100))

        new_texts = [text for text in set(texts) if text not in self.embeddings_map[column_name]]
        print(f"Embedding {len(new_texts)} unique new texts out of {len(texts)} total texts for column '{column_name}'")
        if len(new_texts) > 0:
            resp = self.client.embeddings.create(input=new_texts, model=self.model_name)
            for text, item in zip(new_texts, resp.data):
                emb = np.array(item.embedding, dtype=np.float32)
                self.embeddings_map[column_name][text] = self._normalize(emb.reshape(1, -1))[0].tolist()
        return [self.embeddings_map[column_name][text] for text in texts]
