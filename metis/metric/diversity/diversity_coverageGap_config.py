"""Configuration for the MUP-based coverage-gap metric."""

from __future__ import annotations

from dataclasses import dataclass

from metis.metric.config import MetricConfig


@dataclass
class diversity_coverageGap_config(MetricConfig):
    """Describe the MUP source and its positional dataset-column mapping.

    ``mups_path`` is intended for Python/API use. The Streamlit GUI supplies
    ``mups_content`` directly after upload. Exactly one source must be set.
    MUP fields are matched positionally to ``attributes``. The final field of
    every non-empty row is the MUP's actual dataset coverage. It is validated
    against ``mincov`` when the threshold is available, but it does not affect
    the geometric DNF union count.
    """

    mups_path: str | None = None
    mups_content: str | None = None
    mups_filename: str | None = None
    attributes: list[str] | None = None
    mincov: int | None = None
    wildcard: str = "x"
    delimiter: str = ","

    def validate(self) -> None:
        sources = int(self.mups_path is not None) + int(self.mups_content is not None)
        if sources != 1:
            raise ValueError("Provide exactly one of mups_path or mups_content.")
        if self.mups_path is not None and not self.mups_path.strip():
            raise ValueError("mups_path must not be empty.")
        if not self.attributes:
            raise ValueError(
                "Select the dataset attributes that correspond positionally to the MUP fields."
            )
        if len(set(self.attributes)) != len(self.attributes):
            raise ValueError("attributes must not contain duplicates.")
        if self.mincov is not None and self.mincov <= 0:
            raise ValueError("mincov must be positive when provided.")
        if not self.wildcard:
            raise ValueError("wildcard must not be empty.")
        if len(self.delimiter) != 1:
            raise ValueError("delimiter must be exactly one character.")

    def to_json(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "mups_file": self.mups_filename or self.mups_path,
            "attributes": list(self.attributes or []),
            "mincov": self.mincov,
            "wildcard": self.wildcard,
            "delimiter": self.delimiter,
        }
