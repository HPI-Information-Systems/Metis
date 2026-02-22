from __future__ import annotations

import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd

from typing import Any
from .llm_backend import LLMBackend


try:
    from nltk.corpus import wordnet as wn  # type: ignore
except Exception:  # pragma: no cover
    wn = None

# Loads a shortcut table from CSV to treat tokens such as id, addr, dept, etc. as “valid/readable” even if WordNet does not recognize them.
def load_abbreviations(abbr_csv: Optional[str]) -> Dict[str, str]:
    if not abbr_csv:
        return {}
    if not os.path.exists(abbr_csv):
        return {}
    try:
        df = pd.read_csv(abbr_csv)
    except Exception:
        return {}

    abbr_col = None
    full_col = None
    for c in df.columns:
        cl = c.lower()
        if cl.startswith("abbr"):
            abbr_col = c
        if cl.startswith("full") or cl.endswith("term"):
            full_col = c

    if abbr_col is None or full_col is None:
        return {}

    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        ab = str(row[abbr_col]).strip().lower()
        full = str(row[full_col]).strip()
        if ab and full:
            out[ab] = full
    return out

# Initialization of WordNet-based scorer: Checks whether NLTK WordNet is available (wn.synsets(“test”)) and Cache for tokens to avoid repeated queries.
class WordNetScorer:
    def __init__(self, abbreviations: Optional[Dict[str, str]] = None) -> None:
        self.abbreviations = abbreviations or {}
        self.cache: Dict[str, Tuple[float, float, float]] = {}
        self.wordnet_available = wn is not None and self._check_wordnet()

    @staticmethod
    def _check_wordnet() -> bool:
        try:
            _ = wn.synsets("test")  # type: ignore[union-attr]
            return True
        except Exception:
            return False

    def score(self, token: str) -> Tuple[float, float, float]:
        # Normalize token
        t = token.strip().lower()
        if not t:
            return (0.0, 0.0, 0.0)
        if t in self.cache:
            return self.cache[t]
        
        # Abbreviations are treated as "existing" (E=1). 
        # D and A must come from LLM in HybridScorer (strict or fallback).
        if t in self.abbreviations:
            res = (1.0, 0.0, 0.0) 
            self.cache[t] = res
            return res

        # WordNet lookup lookup -> only Existence (E) is reliable here
        if self.wordnet_available:
            try:
                synsets = wn.synsets(t)  # type: ignore[union-attr]
            except Exception:
                synsets = []
            E = 1.0 if synsets else 0.0
        else:
            # Keep old "best effort" behavior if WordNet isn't available.
            E = 0.5 if t.isalpha() else 0.0

        # D and A are set to 0 placeholders and can be overwritten by LLM in HybridScorer (strict/fallback).
        D = 0.0  # CHANGED (was length heuristic)
        A = 0.0  # CHANGED (was always 0 anyway, but now explicitly "placeholder")

        res = (float(E), float(D), float(A))
        self.cache[t] = res
        return res

# Adapter that uses only WordNet scorer without LLM fallback, unified interface.
class WordNetOnlyAdapter:
    def __init__(self, wordnet: WordNetScorer) -> None:
        self.wordnet = wordnet

    def score_fast(self, token: str):
        E, D, A = self.wordnet.score(token)
        return (E, D, A, "wordnet")

    def needs_llm(self, token: str) -> bool:
        return False

    def score_llm_batch(self, tokens: List[str]) -> None:
        return

    def score(self, token: str):
        E, D, A = self.wordnet.score(token)
        return (E, D, A, "wordnet")

# Hybrid scorer that first uses WordNet and falls back to LLM based on configuration.
class HybridScorer:
    _digit_or_symbol = re.compile(r"[0-9]|[^a-zA-Z_]")

    def __init__(self, cfg: Any, wordnet: WordNetScorer, backend: Optional[LLMBackend]) -> None:
        self.cfg = cfg
        self.wordnet = wordnet
        self.backend = backend if cfg.use_llm_fallback else None
        self.cache: Dict[str, Tuple[float, float, float, str]] = {}

    # Fast scoring using WordNet with caching.
    def score_fast(self, token: str):
        t = token.strip().lower()
        if not t:
            return (0.0, 0.0, 0.0, "none")
        if t in self.cache:
            return self.cache[t]
        E, D, A = self.wordnet.score(t)
        self.cache[t] = (E, D, A, "wordnet")
        return (E, D, A, "wordnet")

    # Determine if LLM scoring is needed based on token characteristics and configuration. The code implements trigger logic that you parameterize in ReadabilityConfig.llm_trigger.
    def needs_llm(self, token: str) -> bool:
    # If LLM is disabled by config, never use it
        if not self.cfg.use_llm_fallback:
            return False

        t = token.strip().lower()
        if not t:
            return False

        # numeric-only tokens are not LLM-scored
        if t.isdigit():
            return False

        mode = str(getattr(self.cfg, "llm_mode", "strict")).lower()

        # strict: LLM for all tokens (except numeric-only)
        if mode == "strict":
            return True

        # fallback: LLM only if WordNet unknown (plus triggers)
        E, _, _, _ = self.score_fast(t)

        # If WordNet knows the token and config enforces "unknown only": no LLM
        if self.cfg.llm_trigger.wordnet_unknown_only and E > 0.0:
            return False

        # Optional trigger: digit/symbol presence
        if self.cfg.llm_trigger.also_if_contains_digit_or_symbol and self._digit_or_symbol.search(t):
            return True

        # WordNet unknown => needs LLM
        return E == 0.0


    # Evaluates a list of tokens via the LLM in batches and stores results in the cache. Only tokens classified as unknown by WordNet are requested.
    def score_llm_batch(self, tokens: List[str]) -> None:
        if self.backend is None or not tokens:
            return

        to_query: List[str] = []
        for t in tokens:
            tt = t.strip().lower()
            if not tt:
                continue
           
            # skip if already LLM-scored
            cur = self.cache.get(tt, None)  # ADDED
            if cur is not None and cur[3] == "llm":  # ADDED
                continue  # ADDED

            if str(self.cfg.llm_mode).lower() == "strict":  # ADDED
                # strict mode: always query (D and A must come from LLM)
                to_query.append(tt)  # ADDED
            else:
                # fallback mode: only query if WordNet says unknown (E==0) or trigger conditions apply
                Ewn, _, _, _ = self.score_fast(tt)
                if Ewn == 0.0:
                    to_query.append(tt)

        if not to_query:
            return

        # Batch processing
        bs = max(10, int(self.cfg.llm_batch_size))
        for i in range(0, len(to_query), bs):
            batch = to_query[i:i+bs]
            scored = self.backend.score_words(batch)
            
            # safety if backend returns unexpected type
            if not isinstance(scored, dict):  # CHANGED
                continue  # CHANGED

            for w, s in scored.items():
                # store full E/D/A from LLM
                self.cache[w] = (
                    float(s.get("E", 0.0)),
                    float(s.get("D", 0.0)),
                    float(s.get("A", 0.0)),
                    "llm",
                )

    # Main scoring function that first checks the cache, then uses WordNet, and falls back to LLM if needed.
    # NOTE: LLM scoring is performed via score_llm_batch() externally (pre-scan) in readability.py
    def score(self, token: str):
        t = token.strip().lower()
        if not t:
            return (0.0, 0.0, 0.0, "none")
        if t in self.cache:
            return self.cache[t]
        return self.score_fast(t)

# Calculates the readability of a column name (at the schema level) based on its tokens and case sensitivity.
def schema_label_score(tokens, s_case, scorer):
    """
    Ehrlinger et al. 2019 (Eq. 5): Red(s) = avg_i ( #fcrit_i / #crit )

    #crit = 4 Kriterien:
      1) Word existence (WordNet)
      2) case consistency
      3) synonyms (fulfilled = no synonyms)
      4) hypernyms (fulfilled = no hypernyms)

    scorer: WordNetOnlyAdapter oder HybridScorer (beide unterstützen score_fast()).
    """
    if not tokens:
        return 0.0

    # (2) Case consistency is label-global
    case_ok = 1 if float(s_case) >= 1.0 else 0

    # Abbreviation-Shortcut (falls verfügbar)
    abbr = None
    if hasattr(scorer, "wordnet") and hasattr(scorer.wordnet, "abbreviations"):
        abbr = scorer.wordnet.abbreviations or {}

    per_token_scores: List[float] = []
    for t in tokens:
        tt = str(t).strip().lower()
        if not tt:
            continue

        # (1) existence über scorer (WordNet / Hybrid)
        if hasattr(scorer, "score_fast"):
            E, _, _, _ = scorer.score_fast(tt)
        else:
            E, _, _, _ = scorer.score(tt)

        exists_ok = 1 if float(E) > 0.0 else 0

        # (3)/(4) synonyms/hypernyms
        # criterion fulfilled if WordNet provides synsets / hypernyms
        
        if abbr is not None and tt in abbr:
            # Abbreviations are treated as fully fulfilled
            syn_ok = 1
            hyp_ok = 1
        else:
            syn_ok = 0
            hyp_ok = 0

            if wn is not None and exists_ok == 1:
                try:
                    synsets = wn.synsets(tt)
                except Exception:
                    synsets = []

                # Synonyms criterion fulfilled if at least one synset exists
                syn_ok = 1 if len(synsets) > 0 else 0

                # Hypernyms criterion fulfilled if at least one hypernym relation exists
                has_hypernym = False
                for ss in synsets:
                    if ss.hypernyms():
                        has_hypernym = True
                        break

                hyp_ok = 1 if has_hypernym else 0

        fcrit = exists_ok + case_ok + syn_ok + hyp_ok
        per_token_scores.append(fcrit / 4.0)

    return float(sum(per_token_scores) / len(per_token_scores)) if per_token_scores else 0.0

# Calculates readability per cell (content level) from tokens. In addition, counters are kept for analysis/annotations.
def content_cell_score(tokens, scorer, unknown_counter=None, difficult_counter=None, llm_candidate_counter=None) -> float:
    if not tokens:
        return 0.0
    word_scores = []
    for t in tokens:
        E, D, A, _ = scorer.score(t)

        # Keep the gate: if existence is 0, score is 0
        if E == 0.0:
            word_scores.append(0.0)
            if unknown_counter is not None:
                unknown_counter[t] += 1
            # Count only those unknowns that would actually trigger LLM
            if llm_candidate_counter is not None and hasattr(scorer, "needs_llm") and scorer.needs_llm(t):
                llm_candidate_counter[t] += 1

        else:
            # Content formula already correct: unweighted mean over criteria
            # score_word = (E + D + A) / 3
            s = (E + D + A) / 3.0
            word_scores.append(s)
            if difficult_counter is not None and D < 0.6:
                difficult_counter[t] += 1
   
    return float(sum(word_scores) / len(word_scores)) if word_scores else 0.0
