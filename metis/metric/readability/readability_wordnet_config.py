from dataclasses import dataclass

from .readability_base_config import readability_base_config


@dataclass(kw_only=True)
class readability_wordnet_config(readability_base_config):
    """
    Configuration class for the readability_wordnet metric.

    This metric measures data readability using WordNet only, with no LLM
    or Hugging Face dependencies.

    Inherits all parameters from readability_base_config:
    sample_size, random_seed, min_token_length, abbr_csv,
    ignore_numeric_columns, compute_schema.
    """

    def to_json(self):
        return self._base_json()