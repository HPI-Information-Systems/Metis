from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from metis.metric.accuracy._strategies.domain_membership import available_strategies
from metis.metric.config import MetricConfig
from metis.utils.reference_loader import ReferenceSource, describe_reference_source


@dataclass
class accuracy_syntacticDomain_config(MetricConfig):
    """Configuration for ``accuracy_syntacticDomain`` (ISO/IEC 25024 Acc-I-1).

    :param reference: An optional domain table or value set. Looked up after
        ``domains``. Accepts a DataFrame, a CSV path, a DataConfig mapping,
        or an inline list/set of values.
    :param method: Strategy key — ``"exact_match"`` (default) or
        ``"wordnet"``. Custom strategies can be registered via
        :func:`metis.metric.accuracy._strategies.domain_membership.register_strategy`.
    :param domains: Per-column allowed values. Looked up before the
        ``reference`` DataFrame. Omit a column to fall through to
        ``reference``; omit both to skip the column with a warning.
    :param method_params: Extra kwargs forwarded to the strategy
        (e.g. ``{"case_insensitive": True}``).
    """

    reference: ReferenceSource = None
    method: Literal["exact_match", "wordnet"] = "exact_match"
    domains: Dict[str, List[str]] | None = None
    method_params: Dict[str, Any] | None = None

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "reference": describe_reference_source(self.reference),
            "method": self.method,
            "domains": self.domains or {},
            "method_params": self.method_params or {},
        }

    def validate(self):
        methods = available_strategies()
        if self.method not in methods:
            raise ValueError(
                f"method must be one of {methods} but was '{self.method}'."
            )
        if self.domains is not None:
            for col, values in self.domains.items():
                if not isinstance(values, (list, tuple, set)):
                    raise ValueError(
                        f"domains['{col}'] must be a list/tuple/set of values."
                    )
        if self.method_params is not None and not isinstance(self.method_params, dict):
            raise ValueError("method_params must be a dict or None.")
