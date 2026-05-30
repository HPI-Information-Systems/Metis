from dataclasses import dataclass, field

from .readability_base_config import readability_base_config


@dataclass(kw_only=True)
class readability_llm_config(readability_base_config):
    """
    Configuration class for the readability_llm metric.

    This metric measures data readability using a hybrid WordNet-first approach
    with an optional LLM fallback for unknown or ambiguous tokens.

    Inherits from readability_base_config:
    sample_size, random_seed, min_token_length, abbr_csv,
    ignore_numeric_columns, compute_schema.

    :param use_llm_fallback: If True, an LLM is used to score tokens that WordNet
                             cannot resolve.
                             Default: True
    :param hf_model_id: Hugging Face model identifier for the LLM backend.
                        Default: "Qwen/Qwen2.5-3B-Instruct"
    :param hf_device: Device to run the HF model on (e.g. "cpu", "cuda", "auto").
                      Default: "auto"
    :param hf_dtype: Data type for the HF model weights (e.g. "float16", "auto").
                     Default: "auto"
    :param hf_max_new_tokens: Maximum number of new tokens the LLM may generate per call.
                              Default: 512
    :param llm_mode: Scoring mode. "fallback" invokes the LLM only for tokens WordNet
                     cannot score; "strict" always uses the LLM.
                     Default: "fallback"
    :param llm_batch_size: Number of tokens submitted to the LLM in a single batch call.
                           Default: 80
    :param llm_trigger_wordnet_unknown_only: If True, LLM is triggered only for tokens
                                             that WordNet marks as unknown.
                                             Default: True
    :param llm_trigger_also_if_contains_digit_or_symbol: If True, tokens containing
                                                          digits or symbols also trigger
                                                          the LLM regardless of WordNet
                                                          result.
                                                          Default: True
    :param column_level_llm_score: If True, an additional top-down column-level LLM
                                   score is computed and blended with the bottom-up score.
                                   Default: False
    :param column_level_llm_sample_values: Maximum number of unique cell values sampled
                                           per column for the top-down LLM scoring.
                                           Default: 100
    :param column_level_llm_gamma: Blending weight for the bottom-up score when combining
                                   with the top-down score (0.0 to 1.0). A value of 1.0
                                   uses only the bottom-up score; 0.0 uses only the
                                   top-down score.
                                   Default: 0.5
    """

    use_llm_fallback: bool = field(default=True)
    hf_model_id: str = field(default="Qwen/Qwen2.5-3B-Instruct")
    hf_device: str = field(default="auto")
    hf_dtype: str = field(default="auto")
    hf_max_new_tokens: int = field(default=512)

    llm_mode: str = field(default="fallback")
    llm_batch_size: int = field(default=80)
    llm_trigger_wordnet_unknown_only: bool = field(default=True)
    llm_trigger_also_if_contains_digit_or_symbol: bool = field(default=True)

    column_level_llm_score: bool = field(default=False)
    column_level_llm_sample_values: int = field(default=100)
    column_level_llm_gamma: float = field(default=0.5)

    def __post_init__(self):
        super().__post_init__()

        if not isinstance(self.use_llm_fallback, bool):
            raise ValueError(f"use_llm_fallback must be boolean, got {type(self.use_llm_fallback)}")

        if not isinstance(self.hf_model_id, str) or not self.hf_model_id.strip():
            raise ValueError(f"hf_model_id must be a non-empty string, got {self.hf_model_id!r}")

        if not isinstance(self.hf_device, str) or not self.hf_device.strip():
            raise ValueError(f"hf_device must be a non-empty string, got {self.hf_device!r}")

        if not isinstance(self.hf_dtype, str) or not self.hf_dtype.strip():
            raise ValueError(f"hf_dtype must be a non-empty string, got {self.hf_dtype!r}")

        if not isinstance(self.hf_max_new_tokens, int) or self.hf_max_new_tokens < 1:
            raise ValueError(f"hf_max_new_tokens must be a positive integer, got {self.hf_max_new_tokens!r}")

        if self.llm_mode not in ("fallback", "strict"):
            raise ValueError(f"llm_mode must be 'fallback' or 'strict', got {self.llm_mode!r}")

        if not isinstance(self.llm_batch_size, int) or self.llm_batch_size < 1:
            raise ValueError(f"llm_batch_size must be a positive integer, got {self.llm_batch_size!r}")

        if not isinstance(self.llm_trigger_wordnet_unknown_only, bool):
            raise ValueError(f"llm_trigger_wordnet_unknown_only must be boolean, got {type(self.llm_trigger_wordnet_unknown_only)}")

        if not isinstance(self.llm_trigger_also_if_contains_digit_or_symbol, bool):
            raise ValueError(f"llm_trigger_also_if_contains_digit_or_symbol must be boolean, got {type(self.llm_trigger_also_if_contains_digit_or_symbol)}")

        if not isinstance(self.column_level_llm_score, bool):
            raise ValueError(f"column_level_llm_score must be boolean, got {type(self.column_level_llm_score)}")

        if not isinstance(self.column_level_llm_sample_values, int) or self.column_level_llm_sample_values < 1:
            raise ValueError(f"column_level_llm_sample_values must be a positive integer, got {self.column_level_llm_sample_values!r}")

        if not isinstance(self.column_level_llm_gamma, (int, float)):
            raise ValueError(f"column_level_llm_gamma must be numeric, got {type(self.column_level_llm_gamma)}")

        if not 0.0 <= self.column_level_llm_gamma <= 1.0:
            raise ValueError(f"column_level_llm_gamma must be between 0.0 and 1.0, got {self.column_level_llm_gamma}")

        self.column_level_llm_gamma = float(self.column_level_llm_gamma)

    def to_json(self):
        return {
            **self._base_json(),
            "use_llm_fallback": self.use_llm_fallback,
            "hf_model_id": self.hf_model_id,
            "hf_device": self.hf_device,
            "hf_dtype": self.hf_dtype,
            "hf_max_new_tokens": self.hf_max_new_tokens,
            "llm_mode": self.llm_mode,
            "llm_batch_size": self.llm_batch_size,
            "llm_trigger": {
                "wordnet_unknown_only": self.llm_trigger_wordnet_unknown_only,
                "also_if_contains_digit_or_symbol": self.llm_trigger_also_if_contains_digit_or_symbol,
            },
            "column_level_llm_score": self.column_level_llm_score,
            "column_level_llm_sample_values": self.column_level_llm_sample_values,
            "column_level_llm_gamma": self.column_level_llm_gamma,
        }