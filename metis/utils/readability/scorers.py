"""
Scoring helpers for the readability metrics.

This module contains the shared scoring logic used by the readability metrics, including abbreviation loading, WordNet-based checks, and optional hybrid/LLM-assisted scoring for schema labels and textual cell content.
"""

from __future__ import annotations

import os
import re
import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

from .llm_backend import LLMBackend

try:
    from nltk.corpus import wordnet as wn
except Exception:  # pragma: no cover
    wn = None


# ----------------------------
# Utilities
# ----------------------------

def load_abbreviations(abbr_csv: Optional[str]) -> Dict[str, str]:
    """Load abbreviation table to treat tokens like id, addr, dept as valid/readable."""
    if not abbr_csv or not os.path.exists(abbr_csv):
        return {}
    try:
        df = pd.read_csv(abbr_csv)
    except Exception:
        return {}

    abbr_col = None
    full_col = None
    for c in df.columns:
        cl = str(c).lower()
        if cl.startswith("abbr"):
            abbr_col = c
        if cl.startswith("full") or cl.endswith("term"):
            full_col = c

    if abbr_col is None or full_col is None:
        return {}

    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        ab = str(row.get(abbr_col, "")).strip().lower()
        full = str(row.get(full_col, "")).strip()
        if ab and full:
            out[ab] = full
    return out


def _is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)


# ----------------------------
# WordNet helper (schema + cognates proxy)
# ----------------------------

class WordNetHelper:
    def __init__(self, abbreviations: Optional[Dict[str, str]] = None) -> None:
        self.abbreviations = abbreviations or {}
        self.wordnet_available = wn is not None and self._check_wordnet()
        self._synsets_cache: Dict[str, List[Any]] = {}
        self._lemmas_cache: Dict[str, Set[str]] = {}
        self._hypernyms_cache: Dict[str, Set[str]] = {}

    @staticmethod
    def _check_wordnet() -> bool:
        try:
            _ = wn.synsets("test")  # type: ignore[attr-defined]
            return True
        except Exception:
            return False

    def synsets(self, t: str) -> List[Any]:
        t = str(t).strip().lower()
        if not t:
            return []
        if t in self._synsets_cache:
            return self._synsets_cache[t]
        if not self.wordnet_available:
            self._synsets_cache[t] = []
            return []
        try:
            ss = wn.synsets(t)  # type: ignore[attr-defined]
        except Exception:
            ss = []
        self._synsets_cache[t] = ss
        return ss

    def existence(self, t: str) -> float:
        tt = str(t).strip().lower()
        if not tt:
            return 0.0
        if tt in self.abbreviations:
            return 1.0
        return 1.0 if len(self.synsets(tt)) > 0 else 0.0

    def synonym_lemmas(self, t: str) -> Set[str]:
        """Proxy set of synonyms via lemma names across synsets."""
        tt = str(t).strip().lower()
        if not tt:
            return set()
        if tt in self._lemmas_cache:
            return self._lemmas_cache[tt]
        if tt in self.abbreviations:
            self._lemmas_cache[tt] = set()
            return set()

        lemmas: Set[str] = set()
        for ss in self.synsets(tt):
            try:
                names = ss.lemma_names()
            except Exception:
                names = []
            for l in names:
                lemmas.add(str(l).lower().replace("_", " "))
        lemmas.discard(tt)
        lemmas.discard(tt.replace("_", " "))
        self._lemmas_cache[tt] = lemmas
        return lemmas

    def hypernym_lemmas(self, t: str) -> Set[str]:
        """Proxy set of hypernyms via lemma names of direct hypernym synsets."""
        tt = str(t).strip().lower()
        if not tt:
            return set()
        if tt in self._hypernyms_cache:
            return self._hypernyms_cache[tt]
        if tt in self.abbreviations:
            self._hypernyms_cache[tt] = set()
            return set()

        hypers: Set[str] = set()
        for ss in self.synsets(tt):
            try:
                hs = ss.hypernyms()
            except Exception:
                hs = []
            for h in hs:
                try:
                    for l in h.lemma_names():
                        hypers.add(str(l).lower().replace("_", " "))
                except Exception:
                    continue
        hypers.discard(tt)
        hypers.discard(tt.replace("_", " "))
        self._hypernyms_cache[tt] = hypers
        return hypers

    def homonyms_count_proxy(self, t: str) -> int:
        """
        Proxy for 'homonyms' via polysemy: number of synsets - 1.
        """
        tt = str(t).strip().lower()
        if not tt or tt in self.abbreviations:
            return 0
        return max(0, len(self.synsets(tt)) - 1)

    def cognates_counts_wordnet(self, t: str) -> Tuple[int, int, int]:
        """
        Counts needed for DQ4AI Eq.(3) inner term:
          synonyms, homonyms, hypernyms
        """
        tt = str(t).strip().lower()
        if not tt or tt in self.abbreviations:
            return (0, 0, 0)
        syn = len(self.synonym_lemmas(tt))
        hom = self.homonyms_count_proxy(tt)
        hyp = len(self.hypernym_lemmas(tt))
        return (syn, hom, hyp)

    @staticmethod
    def cognates_score_from_counts(syn: int, hom: int, hyp: int) -> float:
        # Inner term of DQ4AI Eq.(3), mapped to [0,1]
        return (1.0 / 3.0) * (1.0 / (syn + 1.0) + 1.0 / (hom + 1.0) + 1.0 / (hyp + 1.0))


# ----------------------------
# Word-level scorer used by METIS (E, D, A)
# ----------------------------

class WordNetScorer:
    """
    WordNet-based (fast) scorer:
      E = existence (0/1)
      D = NaN (not available from WordNet in our implementation)
      A = cognates-score proxy (DQ4AI Eq.(3) inner term) from WordNet counts
    """
    def __init__(self, abbreviations: Optional[Dict[str, str]] = None) -> None:
        self.helper = WordNetHelper(abbreviations=abbreviations)

    def score(self, token: str) -> Tuple[float, float, float]:
        t = str(token).strip().lower()
        if not t:
            return (0.0, float("nan"), 0.0)

        E = float(self.helper.existence(t))
        if E == 0.0:
            return (0.0, float("nan"), 0.0)

        D = float("nan")
        syn, hom, hyp = self.helper.cognates_counts_wordnet(t)
        A = float(self.helper.cognates_score_from_counts(syn, hom, hyp))
        return (E, D, A)


class WordNetOnlyAdapter:
    """Adapter to unify interface: returns (E, D, A, source)."""
    def __init__(self, wordnet: WordNetScorer) -> None:
        self.wordnet = wordnet

    def score_fast(self, token: str) -> Tuple[float, float, float, str]:
        E, D, A = self.wordnet.score(token)
        return (E, D, A, "wordnet")

    def needs_llm(self, token: str) -> bool:
        return False

    def score_llm_batch(self, tokens: List[str]) -> None:
        return

    def score(self, token: str) -> Tuple[float, float, float, str]:
        E, D, A = self.wordnet.score(token)
        return (E, D, A, "wordnet")


# ----------------------------
# Ehrlinger 2019 schema readability (Eq.5, reproduced-ish)
# ----------------------------

def _case_ok(schema_case_score: float) -> int:
    return 1 if float(schema_case_score) >= 1.0 else 0


def schema_readability_ehrlinger_2019(
    schema_tokens: List[str],
    schema_case_score: float,
    wnh: WordNetHelper,
    schema_vocab: Optional[Iterable[str]] = None,
) -> float:
    """
    Ehrlinger 2019 Eq.(5) approximation:
      Red(s) = (1/|w|) * sum_i (#fcrit_i / 4)
    Criteria:
      1) existence
      2) case consistency (label-global)
      3) no-synonym-relation-with-schema
      4) no-hypernym-relation-with-schema

    NOTE: For strict reproduction you should pass schema_vocab (all schema tokens).
    If schema_vocab is None, we fall back to label-local vocab (still meaningful, but not strict-reproduced).
    """
    tokens = [str(t).strip().lower() for t in (schema_tokens or []) if str(t).strip()]
    if not tokens:
        return 0.0

    vocab_src = schema_vocab if schema_vocab is not None else tokens
    vocab = set(str(t).strip().lower() for t in vocab_src if str(t).strip())
    vocab.discard("")

    c_ok = _case_ok(schema_case_score)

    exists: Dict[str, int] = {t: (1 if wnh.existence(t) > 0.0 else 0) for t in vocab}
    syn_lemmas: Dict[str, Set[str]] = {}
    hyp_lemmas: Dict[str, Set[str]] = {}

    for t in vocab:
        if exists[t] == 1:
            syn_lemmas[t] = wnh.synonym_lemmas(t)
            hyp_lemmas[t] = wnh.hypernym_lemmas(t)
        else:
            syn_lemmas[t] = set()
            hyp_lemmas[t] = set()

    per_token_scores: List[float] = []
    for t in tokens:
        ex = exists.get(t, 0)

        # synonyms fulfilled iff no schema token is in synonym lemma set (or vice versa)
        syn_ok = 1
        hyp_ok = 1

        if ex != 1:
            syn_ok = 0
            hyp_ok = 0
        else:
            for u in vocab:
                if u == t:
                    continue
                if u in syn_lemmas[t] or t in syn_lemmas[u]:
                    syn_ok = 0
                    break
            for u in vocab:
                if u == t:
                    continue
                if u in hyp_lemmas[t] or t in hyp_lemmas[u]:
                    hyp_ok = 0
                    break

        fcrit = ex + c_ok + syn_ok + hyp_ok
        per_token_scores.append(fcrit / 4.0)

    return float(sum(per_token_scores) / len(per_token_scores)) if per_token_scores else 0.0


def schema_label_score(tokens, s_case, scorer, schema_vocab=None) -> float:
    """
    Backward-compatible wrapper (METIS expects this signature).
    If schema_vocab is None, we DO NOT crash; we fall back to label-local vocab.
    """
    wnh = None
    if hasattr(scorer, "wordnet") and hasattr(scorer.wordnet, "helper"):
        wnh = scorer.wordnet.helper
    elif hasattr(scorer, "helper"):
        wnh = scorer.helper

    if wnh is None or not isinstance(wnh, WordNetHelper):
        wnh = WordNetHelper()

    return schema_readability_ehrlinger_2019(
        schema_tokens=list(tokens or []),
        schema_case_score=float(s_case),
        wnh=wnh,
        schema_vocab=schema_vocab,  # may be None => fallback behavior
    )


# ----------------------------
# DQ4AI 2025 content readability (Eq.2/3/4, reproduced)
# ----------------------------

def content_cell_score(tokens, scorer, unknown_counter=None, difficult_counter=None, llm_candidate_counter=None) -> float:
    """
    DQ4AI 2025 reproduced aggregation:
      Read(z,k) = (1/|Wz|) * sum_{w in Wz} S(w,k)
      Read(z)   = (1/|K_eff|) * sum_{k in available criteria} Read(z,k)
    Dependency: if E(w)=0 then D(w) and A(w) contribute 0.
    """
    toks = [str(t).strip().lower() for t in (tokens or []) if str(t).strip()]
    if not toks:
        return 0.0

    n = float(len(toks))
    sum_E = 0.0
    sum_D = 0.0
    sum_A = 0.0

    has_D = False  # track if difficulty is available for this cell

    for t in toks:
        E, D, A, src = scorer.score(t)

        E = float(E)
        sum_E += E

        if E == 0.0:
            if unknown_counter is not None:
                unknown_counter[t] += 1
            # candidate counter: only if scorer would use LLM for this token
            if llm_candidate_counter is not None and hasattr(scorer, "needs_llm") and scorer.needs_llm(t):
                llm_candidate_counter[t] += 1
            continue

        # Difficulty only if present (LLM)
        if not _is_nan(D):
            has_D = True
            d = float(D)
            sum_D += d
            if difficult_counter is not None and src == "llm" and d < 0.6:
                difficult_counter[t] += 1

        # Cognates (A) should exist for WordNet + LLM; if NaN -> treat as 0
        if not _is_nan(A):
            sum_A += float(A)

    read_E = sum_E / n
    read_A = sum_A / n

    if has_D:
        read_D = sum_D / n
        return float((read_E + read_D + read_A) / 3.0)
    else:
        # WordNet-only behavior: only E and A are available
        return float((read_E + read_A) / 2.0)


# ----------------------------
# Hybrid scorer (WordNet first, LLM fallback)
# ----------------------------

class HybridScorer:
    _digit_or_symbol = re.compile(r"[0-9]|[^a-zA-Z_]")

    def __init__(self, cfg: Any, wordnet: WordNetScorer, backend: Optional[LLMBackend]) -> None:
        self.cfg = cfg
        self.wordnet = wordnet
        # backend is allowed to be None (lazy init in readability_llm)
        self.backend = backend if getattr(cfg, "use_llm_fallback", False) else None
        self.cache: Dict[str, Tuple[float, float, float, str]] = {}

    def _trig_bool(self, key: str, default: bool) -> bool:
        """Read trigger bool from cfg.llm_trigger (supports dataclass/object OR dict OR missing)."""
        trig = getattr(self.cfg, "llm_trigger", None)
        if trig is None:
            return default
        if isinstance(trig, dict):
            return bool(trig.get(key, default))
        return bool(getattr(trig, key, default))

    def score_fast(self, token: str) -> Tuple[float, float, float, str]:
        t = str(token).strip().lower()
        if not t:
            return (0.0, float("nan"), 0.0, "none")
        if t in self.cache:
            return self.cache[t]

        E, D, A = self.wordnet.score(t)
        res = (float(E), float(D), float(A), "wordnet")
        self.cache[t] = res
        return res

    def needs_llm(self, token: str) -> bool:
        # LLM global disabled?
        if not getattr(self.cfg, "use_llm_fallback", False):
            return False

        t = str(token).strip().lower()
        if not t:
            return False

        # numeric-only tokens: never LLM
        if t.isdigit():
            return False

        mode = str(getattr(self.cfg, "llm_mode", "fallback")).lower()

        # strict: always query (except numeric-only)
        if mode == "strict":
            return True

        # fallback:
        E, _, _, _ = self.score_fast(t)

        # unknown-only gate: if WordNet knows it -> no LLM
        if self._trig_bool("wordnet_unknown_only", True) and float(E) > 0.0:
            return False

        # optional trigger: digit/symbol inside token
        if self._trig_bool("also_if_contains_digit_or_symbol", False):
            if self._digit_or_symbol.search(t):
                return True

        # default fallback: WordNet unknown
        return float(E) == 0.0

    def score_llm_batch(self, tokens: List[str]) -> None:
        if self.backend is None or not tokens:
            return

        mode = str(getattr(self.cfg, "llm_mode", "fallback")).lower()
        to_query: List[str] = []

        for t in tokens:
            tt = str(t).strip().lower()
            if not tt:
                continue
            cur = self.cache.get(tt)
            if cur is not None and cur[3] == "llm":
                continue

            if mode == "strict" or self.needs_llm(tt):
                to_query.append(tt)

        if not to_query:
            return

        bs = max(1, int(getattr(self.cfg, "llm_batch_size", 10)))
        for i in range(0, len(to_query), bs):
            batch = to_query[i:i + bs]
            scored = self.backend.score_words(batch)
            if not isinstance(scored, dict):
                continue

            for w, s in scored.items():
                ww = str(w).strip().lower()
                if not ww:
                    continue
                E = float(s.get("E", 0.0))
                D = float(s.get("D", float("nan")))
                A = float(s.get("A", 0.0))
                self.cache[ww] = (E, D, A, "llm")

    def score(self, token: str) -> Tuple[float, float, float, str]:
        t = str(token).strip().lower()
        if not t:
            return (0.0, float("nan"), 0.0, "none")
        if t in self.cache:
            return self.cache[t]
        return self.score_fast(t)