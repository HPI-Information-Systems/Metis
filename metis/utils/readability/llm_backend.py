from __future__ import annotations

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
    """
    Extract the first complete JSON object from text.

    Robust against:
      - leading/trailing text
      - multiple JSON objects
      - fenced code blocks
      - braces inside strings
      - incomplete trailing generation (then returns {})
    """
    if not text:
        return {}

    # 1) direct parse
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else {}
    except Exception:
        pass

    # 2) try fenced ```json ... ```
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            v = json.loads(fenced.group(1))
            return v if isinstance(v, dict) else {}
        except Exception:
            pass

    # 3) brace-balancing: find first complete {...}
    start = text.find("{")
    if start < 0:
        return {}

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue

        # not in string:
        if ch == '"':
            in_str = True
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    v = json.loads(candidate)
                    return v if isinstance(v, dict) else {}
                except Exception:
                    # if this candidate isn't parsable, continue searching for another object
                    # (rare, but can happen if model emitted malformed braces earlier)
                    # Try to find a later '{' and restart.
                    next_start = text.find("{", start + 1)
                    if next_start < 0:
                        return {}
                    start = next_start
                    depth = 0
                    in_str = False
                    esc = False

    # no complete object found (likely truncated generation)
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
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
            do_sample=False,
        )

        new_tokens = gen[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


    def score_words(self, words: List[str]) -> Dict[str, Dict[str, float]]:
        def _to_int(x: Any) -> int:
            try:
                return max(0, int(float(x)))
            except Exception:
                return 0

        def _compute_A(d: Dict[str, Any]) -> float:
            syn = _to_int(d.get("synonyms", 0))
            hom = _to_int(d.get("homonyms", 0))
            hyp = _to_int(d.get("hypernyms", 0))
            A = (1.0/3.0) * (1.0/(syn+1.0) + 1.0/(hom+1.0) + 1.0/(hyp+1.0))
            return _clamp01(A)

        def _is_inner_object(d: Any) -> bool:
            # Heuristik: sieht aus wie {"E":..., "D":..., "synonyms":...} statt {"token": {...}}
            if not isinstance(d, dict):
                return False
            keys = set(d.keys())
            allowed = {"E", "D", "synonyms", "homonyms", "hypernyms"}
            return len(keys) > 0 and keys.issubset(allowed)

        prompt = (
            "Return ONLY a JSON object. No extra text. No markdown.\n"
            "Keys must match the provided tokens exactly.\n"
            "Each value must be an object with:\n"
            "  E: float in [0,1]\n"
            "  D: float in [0,1]\n"
            "  synonyms: integer >=0\n"
            "  homonyms: integer >=0\n"
            "  hypernyms: integer >=0\n"
            f"Tokens: {json.dumps(words)}\n"
        )

        # --- FIX 1: Single-token "inner object" wrap ---
        #if _is_inner_object(parsed) and len(words) == 1:
        #    parsed = {words[0]: parsed}

        text = self._chat(prompt)
        parsed = _extract_json_object(text)

        # Retry once or twice if we couldn't parse a complete JSON object
        if not isinstance(parsed, dict) or len(parsed) == 0:
            for _ in range(2):
                retry_prompt = (
                    "Return ONLY a JSON object. No extra text. No markdown.\n"
                    "IMPORTANT: Output MUST be a single JSON object and MUST be complete.\n"
                    "Keys must match the provided tokens exactly.\n"
                    "Each value must be an object with fields: E, D, synonyms, homonyms, hypernyms.\n"
                    f"Tokens: {json.dumps(words)}\n"
                )
                text = self._chat(retry_prompt)
                parsed = _extract_json_object(text)
                if isinstance(parsed, dict) and len(parsed) > 0:
                    break

        # If still no JSON, return conservative zeros instead of crashing
        if not isinstance(parsed, dict) or len(parsed) == 0:
            out = {w: {"E": 0.0, "D": 0.0, "A": 0.0} for w in words}
            return out
        out: Dict[str, Dict[str, float]] = {}

        # Fill whatever is present
        for w in words:
            v = parsed.get(w)
            if isinstance(v, dict):
                E = _clamp01(v.get("E", 0.0))
                D = _clamp01(v.get("D", 0.0))
                A = _compute_A(v)
                out[w] = {"E": E, "D": D, "A": A}

        missing = [w for w in words if w not in out]

        # --- FIX 2: Retry missing tokens (small, targeted) ---
        # (2 attempts are usually enough)
        for _ in range(2):
            if not missing:
                break
            retry_prompt = (
                "Return ONLY a JSON object. No extra text. No markdown.\n"
                "Keys must match the provided tokens exactly.\n"
                "Example format:\n"
                "{\n"
                '  "TOKEN": {"E": 1.0, "D": 0.5, "synonyms": 0, "homonyms": 0, "hypernyms": 0}\n'
                "}\n"
                f"Tokens: {json.dumps(missing)}\n"
            )
            retry_text = self._chat(retry_prompt)
            retry_parsed = _extract_json_object(retry_text)

            # again handle single-token inner object
            if _is_inner_object(retry_parsed) and len(missing) == 1:
                retry_parsed = {missing[0]: retry_parsed}

            if isinstance(retry_parsed, dict):
                for w in list(missing):
                    v = retry_parsed.get(w)
                    if isinstance(v, dict):
                        E = _clamp01(v.get("E", 0.0))
                        D = _clamp01(v.get("D", 0.0))
                        A = _compute_A(v)
                        out[w] = {"E": E, "D": D, "A": A}

            missing = [w for w in words if w not in out]

        # --- FIX 3: Final conservative fill (no positive fallback) ---
        for w in missing:
            out[w] = {"E": 0.0, "D": 0.0, "A": 0.0}

        return out