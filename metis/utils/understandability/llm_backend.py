from __future__ import annotations

"""Contextual Hugging Face backend for the understandability metric.

Tokenization is completed once by the metric before pipeline selection. Every
LLM request therefore contains the same exact word occurrences used by the
resource-based and hybrid pipelines. The model returns one contextual score per
supplied word occurrence and one separate holistic score for the complete cell.
The cell score is never calculated from the word scores.
"""

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


def _debug_enabled() -> bool:
    return str(os.environ.get("METIS_LLM_DEBUG", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _debug_dir() -> Path:
    configured = os.environ.get("METIS_LLM_DEBUG_DIR")
    if configured:
        return Path(configured)
    return Path.cwd() / ".cache" / "metis" / "understandability" / "llm_debug"


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        return repr(value)


def _debug_event(stage: str, payload: Dict[str, Any]) -> None:
    if not _debug_enabled():
        return

    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        **{key: _json_safe(value) for key, value in payload.items()},
    }
    directory = _debug_dir()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "llm_backend_debug.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, ensure_ascii=False) + "\n")

    requested = event.get("requested_cell_ids") or []
    missing = event.get("missing_cell_ids") or []
    raw_preview = str(event.get("raw_response", ""))[:500].replace("\n", "\\n")
    print(f"[LLM DEBUG] {stage}: requested={requested} missing={missing} raw='{raw_preview}'")


def _parse_json(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_balanced_json(text: str, opening: str, closing: str) -> Optional[Any]:
    start = text.find(opening)
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    parsed = _parse_json(text[start : index + 1])
                    if parsed is not None:
                        return parsed
                    break
        start = text.find(opening, start + 1)
    return None


def _extract_json_payload(text: str) -> Any:
    """Extract one JSON object while tolerating accidental code fences."""

    if not text:
        return {}
    parsed = _parse_json(text.strip())
    if parsed is not None:
        return parsed

    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    for block in fenced:
        parsed = _parse_json(block.strip())
        if parsed is not None:
            return parsed
        parsed = _extract_balanced_json(block, "{", "}")
        if parsed is not None:
            return parsed

    parsed = _extract_balanced_json(text, "{", "}")
    return parsed if parsed is not None else {}


def _finite_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except Exception:
        return None
    if score != score or score in (float("inf"), float("-inf")):
        return None
    if not 0.0 <= score <= 1.0:
        return None
    return score


def _target_words_from_request(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    target_words: List[Dict[str, Any]] = []
    for position, raw in enumerate(request.get("target_words", []) or []):
        if not isinstance(raw, dict):
            continue
        word_id = str(raw.get("word_id", raw.get("id", "")))
        surface = str(raw.get("surface", raw.get("token", "")))
        if not word_id or not surface:
            continue
        target_words.append(
            {
                "word_id": word_id,
                "surface": surface,
                "token_position": int(raw.get("token_position", position)),
            }
        )
    return target_words


def _find_cell_word_payload(parsed: Any, request_id: str) -> Optional[Dict[str, Any]]:
    """Find the requested cell object.

    The active schema is a flat object with ``cell_id``. Legacy nested objects
    are still accepted by the parser so old debug outputs remain inspectable.
    """

    if isinstance(parsed, dict):
        if str(parsed.get("cell_id", "")) == request_id and isinstance(
            parsed.get("word_scores"), list
        ):
            return parsed
        direct = parsed.get(request_id)
        if isinstance(direct, dict):
            return direct
        for wrapper in ("cells", "results", "items", "data", "output"):
            found = _find_cell_word_payload(parsed.get(wrapper), request_id)
            if found is not None:
                return found
    elif isinstance(parsed, list):
        for item in parsed:
            found = _find_cell_word_payload(item, request_id)
            if found is not None:
                return found
    return None


def _validate_cell_word_response(
    parsed: Any,
    request: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Validate one complete word- and cell-level LLM response.

    The LLM only needs to return IDs and scores. Surface forms and token
    positions are reconstructed from the authoritative pre-tokenized request.
    Partial responses are rejected.
    """

    request_id = str(request.get("id", ""))
    expected = _target_words_from_request(request)
    payload = _find_cell_word_payload(parsed, request_id)
    if payload is None:
        return None, [f"missing response object for cell ID {request_id!r}"]

    returned = payload.get("word_scores")
    if not isinstance(returned, list):
        return None, ["word_scores must be a JSON array"]

    errors: List[str] = []
    if len(returned) != len(expected):
        errors.append(f"expected {len(expected)} word entries but received {len(returned)}")

    expected_ids = [item["word_id"] for item in expected]
    returned_ids = [
        str(item.get("word_id", item.get("id", "")))
        for item in returned
        if isinstance(item, dict)
    ]
    if len(returned_ids) != len(set(returned_ids)):
        errors.append("one or more word IDs occur more than once")

    unexpected_ids = [word_id for word_id in returned_ids if word_id not in set(expected_ids)]
    missing_ids = [word_id for word_id in expected_ids if word_id not in set(returned_ids)]
    if unexpected_ids:
        errors.append(f"unexpected word IDs: {unexpected_ids}")
    if missing_ids:
        errors.append(f"missing word IDs: {missing_ids}")

    validated: List[Dict[str, Any]] = []
    for index, expected_word in enumerate(expected):
        if index >= len(returned):
            break
        raw = returned[index]
        if not isinstance(raw, dict):
            errors.append(f"entry {index} is not an object")
            continue

        word_id = str(raw.get("word_id", raw.get("id", "")))
        if word_id != expected_word["word_id"]:
            errors.append(
                f"entry {index} must use word ID {expected_word['word_id']!r}, not {word_id!r}"
            )

        score = _finite_score(raw.get("score", raw.get("understandability_score")))
        if score is None:
            errors.append(
                f"score for {expected_word['word_id']!r} must be numeric and between 0 and 1"
            )
            continue

        validated.append(
            {
                "word_id": expected_word["word_id"],
                "surface": expected_word["surface"],
                "token_position": expected_word["token_position"],
                "understandability_score": score,
            }
        )

    cell_score = _finite_score(
        payload.get(
            "cell_score",
            payload.get("cell_understandability_score", payload.get("understandability_score")),
        )
    )
    if cell_score is None:
        errors.append("cell_score must be numeric and between 0 and 1")

    errors = list(dict.fromkeys(errors))
    if errors or len(validated) != len(expected) or cell_score is None:
        return None, errors or ["incomplete word-and-cell response"]

    return (
        {
            "cell_id": request_id,
            "understandability_score": float(cell_score),
            "cell_understandability_score": float(cell_score),
            "cell_score": float(cell_score),
            "word_scores": validated,
            "expected_word_count": len(expected),
            "returned_word_count": len(validated),
            "assessment_coverage": 1.0 if expected else 0.0,
            "confidence": None,
            "reason": None,
            "aggregation": "independent_holistic_cell_assessment",
            "cell_score_source": "llm_holistic_cell_assessment",
        },
        [],
    )


class LLMBackend:
    def score_understandability_cells(
        self, requests: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError


@dataclass
class HFTransformersBackend(LLMBackend):
    model_id: str
    model_revision: Optional[str] = None
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 768
    context_batch_size: int = 1

    def __post_init__(self) -> None:
        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("HFTransformersBackend requires torch and transformers.")

        if self.device == "cuda":
            resolved_device = "cuda"
        elif self.device == "cpu":
            resolved_device = "cpu"
        else:
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resolved_device = resolved_device

        requested_dtype = str(self.dtype).strip().lower()
        if requested_dtype == "float16":
            torch_dtype = torch.float16
        elif requested_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif requested_dtype == "float32":
            torch_dtype = torch.float32
        elif resolved_device == "cuda" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif resolved_device == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
        self.resolved_torch_dtype = torch_dtype

        load_kwargs: Dict[str, Any] = {}
        if self.model_revision:
            load_kwargs["revision"] = self.model_revision

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
            **load_kwargs,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if resolved_device == "cuda" else None,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )
        if resolved_device == "cpu":
            self.model.to("cpu")
        self.model.eval()

    @staticmethod
    def _system_prompt() -> str:
        return (
            "ROLE\n"
            "You are an evaluator of content-level understandability in tabular data.\n\n"
            "TASK\n"
            "For exactly one non-empty textual table cell, produce two distinct assessments:\n"
            "1. one understandability score for every supplied target word occurrence, and\n"
            "2. one independent understandability score for the complete cell value.\n\n"
            "DEFINITION\n"
            "Understandability is the degree to which a member of the specified target user "
            "group can recognize the supplied expression and infer its intended meaning from "
            "the unchanged cell value and the supplied table context. Evaluate understandability, "
            "not factual correctness, popularity, usefulness, completeness, or data accuracy.\n\n"
            "INPUT DATA\n"
            "The user message contains exactly one JSON object with:\n"
            "- cell_id: the unique cell identifier,\n"
            "- target_user_group: the intended users,\n"
            "- data_type: the type of value being assessed,\n"
            "- column_name: interpretive context only, not an object to assess,\n"
            "- cell_value: the complete unchanged value,\n"
            "- target_words: authoritative word occurrences created before all assessment pipelines,\n"
            "- row_context: other assessable values from the same record, and\n"
            "- column_context: representative other values from the same column.\n\n"
            "TOKENIZATION CONSTRAINTS\n"
            "The supplied target_words are final. Do not tokenize again. Do not split, merge, "
            "add, remove, rename, normalize, translate, repair, or correct a target word. Return "
            "exactly one score for every supplied word_id in the same order.\n\n"
            "EVALUATION CRITERIA\n"
            "Consider the following evidence jointly without returning separate feature scores:\n"
            "- recognizability as a word, name, abbreviation, code, identifier, technical term, "
            "or other conventional unit,\n"
            "- familiarity for the specified target user group,\n"
            "- unresolved semantic ambiguity after using the complete cell and context,\n"
            "- clarity of the visible notation and whether it is structured, damaged, irregular, "
            "or difficult to parse,\n"
            "- processing effort required by the visible word form,\n"
            "- support from the other values in the same row, and\n"
            "- support from representative values in the same column.\n\n"
            "ASSESSMENT RULES\n"
            "- Use the column name only to interpret the value. Do not evaluate the column name.\n"
            "- Use general linguistic and domain knowledge appropriate for the target user group, "
            "but do not rely on specific external facts about named entities, products, movies, "
            "books, people, companies, or datasets.\n"
            "- A proper name can be understandable as a name even when its real-world referent is unknown.\n"
            "- Absence from a general dictionary does not automatically imply low understandability.\n"
            "- Technical terms and abbreviations may be understandable in the supplied context.\n"
            "- Digits, hyphens, punctuation, capitalization, and special characters are not negative by themselves.\n"
            "- Lower the score when notation is corrupted, inconsistent, damaged, or difficult to parse.\n"
            "- A long word is not automatically difficult, and a short word is not automatically clear.\n"
            "- Context can resolve ambiguity. Contextual similarity alone is not proof of understandability.\n"
            "- Do not use a fixed default score. Use the continuous range proportionally.\n\n"
            "CELL-LEVEL ASSESSMENT\n"
            "Assess the understandability of the complete unchanged cell value, not merely "
            "whether its general topic, category, or semantic role can be inferred.\n"
            "A cell score of 1.00 is appropriate only when all meaning-bearing components "
            "of the complete cell are clear and interpretable and no relevant uncertainty, "
            "malformed expression, damaged word form, or unresolved component remains.\n"
            "Do not ignore an unclear, corrupted, malformed, or difficult target word merely "
            "because the surrounding words reveal the general category of the cell.\n"
            "If one component is unclear but the remaining cell still permits an approximate "
            "interpretation, reduce the cell score according to the importance of that "
            "component for understanding the complete cell.\n"
            "Distinguish between understanding the general category of the cell and "
            "understanding the complete unchanged cell value. A cell is not fully "
            "understandable merely because its general topic can be inferred.\n"
            "The cell score remains an independent holistic assessment. Do not calculate it "
            "as the mean, minimum, maximum, or another mathematical aggregation of the word "
            "scores. Treat the word-level observations as evidence for the holistic cell "
            "assessment, not as an aggregation formula.\n\n"
            "INTERNAL PROCEDURE\n"
            "Evaluate each supplied word in its exact occurrence within the complete cell. "
            "Use the complete cell value and the supplied row and column context to determine "
            "whether the intended meaning of that word occurrence is clear.\n"
            "Then evaluate the complete unchanged cell separately as a whole, including its "
            "composition, ordering, multi-word meaning, list structure, contextual fit, and "
            "all unclear or malformed components.\n"
            "Before assigning 1.00, verify that every meaning-bearing component of the cell "
            "is clear and that no relevant uncertainty remains.\n"
            "Perform this reasoning internally and output only the required JSON.\n\n"
            "SCORING\n"
            "Use the complete continuous range from 0.00 to 1.00. Do not restrict scores "
            "to a small set of preferred values.\n"
            "1.00 means that the complete unchanged expression is immediately and fully "
            "understandable. Every meaning-bearing component is clear, conventionally "
            "expressed, and interpretable, and no relevant uncertainty, malformed form, "
            "damaged word, or unresolved component remains.\n"
            "Scores around 0.85 indicate that the complete cell is clearly understandable "
            "overall but contains a minor irregularity, limited ambiguity, or a component "
            "that requires some interpretive effort.\n"
            "Scores around 0.70 indicate that the general meaning can be recovered, but at "
            "least one relevant component is unclear, malformed, ambiguous, unfamiliar, or "
            "requires substantial contextual inference.\n"
            "0.50 means that only part of the intended meaning can be understood reliably "
            "and important uncertainty remains.\n"
            "Scores below 0.50 indicate that major parts of the expression remain unclear "
            "or cannot be interpreted reliably.\n"
            "0.00 means that the expression cannot be interpreted meaningfully.\n"
            "Use intermediate values proportionally whenever the assessment lies between "
            "these reference points.\n"
            "Do not assign 1.00 merely because the cell category, semantic role, or general "
            "topic is recognizable. Reserve 1.00 for completely clear unchanged cell values.\n"
            "Before returning the cell score, compare it qualitatively with the word-level "
            "assessment. A low score for a meaning-bearing word is evidence that the complete "
            "cell is not fully understandable. Use this evidence in the holistic assessment, "
            "but do not calculate the cell score as a mean, minimum, maximum, weighted sum, "
            "or any other mathematical aggregation of the word scores.\n\n"
            "OUTPUT\n"
            "Return exactly one valid JSON object. Return no Markdown, code fence, explanation, "
            "confidence, reason, or additional key. Use this exact schema:\n"
            "{\n"
            "  \"cell_id\": \"<unchanged input cell_id>\",\n"
            "  \"cell_score\": <number from 0.00 to 1.00>,\n"
            "  \"word_scores\": [\n"
            "    {\"word_id\": \"<unchanged input word_id>\", \"score\": <number from 0.00 to 1.00>}\n"
            "  ]\n"
            "}\n"
            "The word_scores array must contain exactly one entry for every supplied target word, "
            "in the same order as the input."
        )

    @staticmethod
    def _input_payload(request: Dict[str, Any]) -> Dict[str, Any]:
        target_words = [
            {
                "word_id": word["word_id"],
                "surface": word["surface"],
            }
            for word in _target_words_from_request(request)
        ]
        row_context = [
            {"column": str(column), "value": str(value)}
            for column, value in (request.get("row_context", {}) or {}).items()
        ]
        return {
            "cell_id": str(request.get("id", "")),
            "target_user_group": str(
                request.get("target_user_group", "general adult data users")
            ),
            "data_type": "textual table-cell value",
            "column_name": str(request.get("column", "")),
            "cell_value": str(request.get("cell_value", "")),
            "target_words": target_words,
            "row_context": row_context,
            "column_context": [
                str(value)
                for value in (request.get("column_context_examples", []) or [])
            ],
        }

    def _cell_prompt(
        self,
        request: Dict[str, Any],
        validation_feedback: Optional[List[str]] = None,
    ) -> str:
        sections: List[str] = []
        if validation_feedback:
            sections.extend(
                [
                    "FORMAL CORRECTION REQUIRED",
                    "The previous answer violated the output contract:",
                    json.dumps(validation_feedback, ensure_ascii=False),
                    "Return a complete new assessment for the same input. Do not copy or repair a partial answer.",
                    "",
                ]
            )
        sections.extend(
            [
                "INPUT DATA",
                json.dumps(self._input_payload(request), ensure_ascii=False, indent=2),
                "",
                "Return the required JSON object only.",
            ]
        )
        return "\n".join(sections)

    def _chat(self, user_prompt: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            encoded: Any = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True,
                return_dict=True,
            )
        else:  # pragma: no cover - compatibility path for non-chat tokenizers
            plain = self._system_prompt() + "\n\n" + user_prompt
            encoded = self.tokenizer(plain, return_tensors="pt")

        target_device = getattr(self.model, "device", None)
        if hasattr(encoded, "to") and target_device is not None:
            encoded = encoded.to(target_device)
        elif isinstance(encoded, dict) and target_device is not None:
            encoded = {key: value.to(target_device) for key, value in encoded.items()}

        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": False,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        attention_mask = (
            encoded.get("attention_mask") if isinstance(encoded, dict) else encoded.attention_mask
        )
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask

        with torch.inference_mode():
            generated = self.model.generate(input_ids=input_ids, **generation_kwargs)

        new_tokens = generated[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _score_single_cell(
        self,
        request: Dict[str, Any],
        attempt: str,
        validation_feedback: Optional[List[str]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        prompt = self._cell_prompt(request, validation_feedback=validation_feedback)
        raw_response = self._chat(prompt)
        parsed = _extract_json_payload(raw_response)
        result, errors = _validate_cell_word_response(parsed, request)

        request_id = str(request.get("id", ""))
        _debug_event(
            "score_cell_word",
            {
                "attempt": attempt,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "requested_cell_ids": [request_id],
                "requests": [request],
                "prompt": prompt,
                "raw_response": raw_response,
                "parsed_payload": parsed,
                "mapped_scores": {} if result is None else {request_id: result},
                "validation_errors": {} if result is not None else {request_id: errors},
                "missing_cell_ids": [] if result is not None else [request_id],
            },
        )
        return result, errors

    def score_understandability_cells(
        self, requests: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Assess each cell in one deterministic model call plus at most one retry."""

        clean_requests: List[Dict[str, Any]] = []
        seen_ids = set()
        for request in requests:
            request_id = str(request.get("id", ""))
            target_words = _target_words_from_request(request)
            if not request_id or request_id in seen_ids or not target_words:
                continue
            clean_requests.append(
                {
                    **request,
                    "id": request_id,
                    "cell_value": str(request.get("cell_value", "")),
                    "target_words": target_words,
                }
            )
            seen_ids.add(request_id)

        out: Dict[str, Dict[str, Any]] = {}
        for request in clean_requests:
            request_id = request["id"]
            result, errors = self._score_single_cell(
                request,
                attempt="initial_cell_assessment",
            )
            if result is None:
                result, _retry_errors = self._score_single_cell(
                    request,
                    attempt="retry_after_formal_validation_error",
                    validation_feedback=errors or ["invalid response"],
                )
            if result is not None:
                out[request_id] = result

        _debug_event(
            "score_cell_words_final",
            {
                "requested_cell_ids": [request["id"] for request in clean_requests],
                "returned_scores": out,
                "missing_cell_ids": [
                    request["id"] for request in clean_requests if request["id"] not in out
                ],
            },
        )
        return out
