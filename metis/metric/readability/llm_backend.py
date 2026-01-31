from __future__ import annotations

from email.mime import text
from email.mime import text
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Any

try:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


def _extract_json_object(text: str) -> dict:
    # 1) Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) JSON inside ```json ... ```
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    # 3) First JSON object (non-greedy!)
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, v))


class LLMBackend:
    def score_words(self, words: List[str]) -> Dict[str, Dict[str, float]]:
        raise NotImplementedError

    def score_column(self, column_name: str, sample_values: List[str]) -> float:
        raise NotImplementedError


@dataclass
class HFTransformersBackend(LLMBackend):
    def __init__(self, model_id: str, device: str = "auto", dtype: str = "auto", max_new_tokens: int = 512) -> None:
        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("HFTransformersBackend requires torch and transformers installed.")
        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)

        if device == "cuda":
            self.device = "cuda"
        elif device == "cpu":
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device == "cpu":
            self.model.to("cpu")

    def _chat(self, user_prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": "You output strict JSON only."},
                {"role": "user", "content": user_prompt},
            ]
            enc = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
            )
            input_ids = enc
            pad_id = self.tokenizer.pad_token_id
            attention_mask = (input_ids != pad_id).long() if pad_id is not None else None
        else:
            enc = self.tokenizer(user_prompt, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask")

        if self.device == "cuda":
            input_ids = input_ids.to(self.model.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.model.device)

        gen = self.model.generate(
            input_ids,
    attention_mask=attention_mask,
    max_new_tokens=self.max_new_tokens,
    do_sample=True,
        )
        return self.tokenizer.decode(gen[0], skip_special_tokens=True)


    def score_words(self, words: List[str]) -> Dict[str, Dict[str, float]]:
        prompt = (
            "Return ONLY valid JSON. No extra text. No markdown. No list.\n"
            "Output MUST be a JSON object where each key is exactly one token.\n"
            "Each token maps to an object with numeric fields E,D,A in [0,1].\n"
            "Example:\n"
            "{\n"
            "  \"token1\": {\"E\": 0.9, \"D\": 0.7, \"A\": 0.2},\n"
            "  \"token2\": {\"E\": 0.0, \"D\": 0.0, \"A\": 0.0}\n"
            "}\n\n"
            f"Tokens: {json.dumps(words)}\n"
        )
        text = self._chat(prompt)
        #print("\n[LLM RAW OUTPUT START]\n", text[:800], "\n[LLM RAW OUTPUT END]\n")

        parsed = _extract_json_object(text)

       # HARD FALLBACK: if JSON is missing/invalid, return conservative heuristic scores
        if not isinstance(parsed, dict) or len(parsed) == 0:
            out = {}
            for w in words:
                ww = str(w)
                if ww.isalpha():
                    out[w] = {"E": 0.8, "D": 0.6, "A": 0.2}
                else:
                    out[w] = {"E": 0.0, "D": 0.0, "A": 0.0}

            # Safety: ensure every requested word has an entry
            for w in words:
                out.setdefault(w, {"E": 0.0, "D": 0.0, "A": 0.0})
            return out
    # Normal parsing path
        out = {}
        for w in words:
            v = parsed.get(w)
            if isinstance(v, dict):
                out[w] = {
                    "E": _clamp01(v.get("E", 0.0)),
                    "D": _clamp01(v.get("D", 0.0)),
                    "A": _clamp01(v.get("A", 0.0)),
                }

        for w in words:
            out.setdefault(w, {"E": 0.0, "D": 0.0, "A": 0.0})

        return out

    def score_column(self, column_name: str, sample_values: List[str]) -> float:
        prompt = (
            "Return ONLY valid JSON. No extra text.\n"
            "Output format: {\"score\": <float 0..1>}.\n"
            "Score readability of the column values for general data consumers.\n\n"
            f"Column name: {column_name}\n"
            f"Sample values (unique, representative): {json.dumps(sample_values)}\n"
        )
        text = self._chat(prompt)
        parsed = _extract_json_object(text)
        return _clamp01(parsed.get("score", 0.0))
