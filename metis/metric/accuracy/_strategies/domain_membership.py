import re
from typing import Callable, Dict, Iterable, Optional

import nltk
import pandas as pd

# Strategy signature: (series_of_str, domain | None, **params) -> bool Series
StrategyFn = Callable[..., pd.Series]
DOMAIN_STRATEGIES: Dict[str, StrategyFn] = {}


def register_strategy(name: str, fn: StrategyFn) -> None:
    DOMAIN_STRATEGIES[name] = fn


def _exact_match(
    series: pd.Series,
    domain: Optional[Iterable[str]],
    case_insensitive: bool = False,
) -> pd.Series:
    """Each value is "in domain" iff string-equal to one of the allowed values.

    This is the simplest possible Acc-I-1 check: ``series.isin(domain)`` with
    optional case-insensitivity. For dirty real-world strings consider
    pairing this with a normalizer before calling.
    """
    if domain is None:
        raise ValueError("exact_match strategy requires a domain.")
    if case_insensitive:
        norm_domain = {str(v).lower() for v in domain}
        return series.astype(str).str.lower().isin(norm_domain)
    return series.astype(str).isin(set(map(str, domain)))


_TOKEN_RE = re.compile(r"[A-Za-z]+")
_WORDNET_CACHE: set[str] | None = None


def _wordnet(
    series: pd.Series,
    domain: Optional[Iterable[str]] = None,  # ignored
    case_insensitive: bool = True,
) -> pd.Series:
    global _WORDNET_CACHE
    if _WORDNET_CACHE is None:
        nltk.download("words", quiet=True)
        from nltk.corpus import words as nltk_words
        _WORDNET_CACHE = {w.lower() for w in nltk_words.words()}

    def is_valid(text: str) -> bool:
        tokens = _TOKEN_RE.findall(text.lower() if case_insensitive else text)
        if not tokens:
            return True
        return all(tok in _WORDNET_CACHE for tok in tokens)

    return series.astype(str).map(is_valid)


register_strategy("exact_match", _exact_match)
register_strategy("wordnet", _wordnet)


def available_strategies() -> list[str]:
    return sorted(DOMAIN_STRATEGIES)
