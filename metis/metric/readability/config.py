from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMTriggerConfig:
    """Controls when to call the LLM fallback."""
    wordnet_unknown_only: bool = True
    also_if_contains_digit_or_symbol: bool = True


@dataclass
class ReadabilityConfig:
    # Core
    sample_size: Optional[int] = None
    random_seed: int = 13
    min_token_length: int = 2
    abbr_csv: Optional[str] = None
    ignore_numeric_columns: bool = True

    #separate schema computation (no mixing)
    compute_schema: bool = True

    # LLM usage, Hugging Face backend (3B default, non-gated)
    use_llm_fallback: bool = True
    hf_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    hf_device: str = "auto"       # "auto", "cpu", "cuda"
    hf_dtype: str = "auto"        # "auto", "float16", "bfloat16", "float32"
    hf_max_new_tokens: int = 512

    # ADDED: LLM mode switch (DQ4AI compatibility vs. optimized fallback)
    # - "strict": LLM scores D/A (and optionally E) for all tokens (DQ4AI-comparable)
    # - "fallback": LLM called only for WordNet-unknown tokens / triggers (faster)
    llm_mode: str = "strict"  # ADDED  ("strict" | "fallback")

    # Shared LLM params
    llm_batch_size: int = 80
    llm_trigger: LLMTriggerConfig = field(default_factory=LLMTriggerConfig)

    # Optional top-down column scoring
    column_level_llm_score: bool = False
    column_level_llm_sample_values: int = 100
    column_level_llm_gamma: float = 0.5

    @staticmethod
    def from_metric_config(metric_config: Optional[str]) -> "ReadabilityConfig":
        cfg = ReadabilityConfig()
        if metric_config is None:
            return cfg

        metric_config = metric_config.strip()
        if metric_config.startswith("{"):
            data = json.loads(metric_config)
        else:
            if not os.path.exists(metric_config):
                raise ValueError(f"metric_config is neither JSON nor an existing path: {metric_config}")
            with open(metric_config, "r", encoding="utf-8") as f:
                data = json.load(f)

        # core
        cfg.sample_size = data.get("sample_size", cfg.sample_size)
        cfg.random_seed = int(data.get("random_seed", cfg.random_seed))
        cfg.min_token_length = int(data.get("min_token_length", cfg.min_token_length))
        cfg.abbr_csv = data.get("abbr_csv", cfg.abbr_csv)
        cfg.ignore_numeric_columns = bool(data.get("ignore_numeric_columns", cfg.ignore_numeric_columns))        
        cfg.compute_schema = bool(data.get("compute_schema", cfg.compute_schema))

        # ADDED: llm_mode parsing + validation
        cfg.llm_mode = str(data.get("llm_mode", cfg.llm_mode)).strip().lower()  # ADDED
        if cfg.llm_mode not in ("strict", "fallback"):  # ADDED
            cfg.llm_mode = "strict"  # ADDED

        # HF LLM
        cfg.use_llm_fallback = bool(data.get("use_llm_fallback", cfg.use_llm_fallback))
        cfg.hf_model_id = str(data.get("hf_model_id", cfg.hf_model_id))
        cfg.hf_device = str(data.get("hf_device", cfg.hf_device))
        cfg.hf_dtype = str(data.get("hf_dtype", cfg.hf_dtype))
        cfg.hf_max_new_tokens = int(data.get("hf_max_new_tokens", cfg.hf_max_new_tokens))

        cfg.llm_batch_size = int(data.get("llm_batch_size", cfg.llm_batch_size))


        trig = data.get("llm_trigger", None)
        if isinstance(trig, dict):
            cfg.llm_trigger = LLMTriggerConfig(
                wordnet_unknown_only=bool(trig.get("wordnet_unknown_only", cfg.llm_trigger.wordnet_unknown_only)),
                also_if_contains_digit_or_symbol=bool(trig.get("also_if_contains_digit_or_symbol", cfg.llm_trigger.also_if_contains_digit_or_symbol)),
            )

        # top-down (content only)
        cfg.column_level_llm_score = bool(data.get("column_level_llm_score", cfg.column_level_llm_score))
        cfg.column_level_llm_sample_values = int(data.get("column_level_llm_sample_values", cfg.column_level_llm_sample_values))
        cfg.column_level_llm_gamma = float(data.get("column_level_llm_gamma", cfg.column_level_llm_gamma))

        # clamp gamma
        cfg.column_level_llm_gamma = max(0.0, min(1.0, cfg.column_level_llm_gamma))
        return cfg
