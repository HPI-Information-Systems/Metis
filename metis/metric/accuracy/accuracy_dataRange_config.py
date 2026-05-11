from dataclasses import dataclass
from typing import Dict, Literal, Tuple

from metis.metric.config import MetricConfig

VALID_FALLBACKS = ["profiling", "skip"]


@dataclass
class accuracy_dataRange_config(MetricConfig):
    """
    Configuration for ``accuracy_dataRange`` (ISO/IEC 25024 Acc-I-7).

    :param intervals: Per-column inclusive ``[min, max]`` interval supplied by
        the user. Columns absent from this dict fall through to ``fallback``.
    :param fallback: What to do for columns without a user-supplied interval:
        ``"profiling"`` uses the observed min/max from the cached
        ``value_range`` profile; ``"skip"`` leaves the column unassessed.
    """

    intervals: Dict[str, Tuple[float, float]] | None = None
    fallback: Literal["profiling", "skip"] = "skip"

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "intervals": (
                {k: list(v) for k, v in self.intervals.items()}
                if self.intervals
                else {}
            ),
            "fallback": self.fallback,
        }

    def validate(self):
        if self.fallback not in VALID_FALLBACKS:
            raise ValueError(
                f"fallback must be one of {VALID_FALLBACKS} but was {self.fallback}"
            )
        if self.intervals is None:
            return
        for col, interval in self.intervals.items():
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError(
                    f"interval for column '{col}' must be a 2-element sequence (min, max)."
                )
            lo, hi = interval
            if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
                raise ValueError(
                    f"interval bounds for '{col}' must be numeric (got {type(lo)}, {type(hi)})."
                )
            if lo > hi:
                raise ValueError(
                    f"interval for column '{col}' has min={lo} > max={hi}."
                )
