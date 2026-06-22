from dataclasses import dataclass, field

from metis.metric.config import MetricConfig


@dataclass(kw_only=True)
class minimality_clustering_config(MetricConfig):
    """
    Configuration class for the minimality_clustering metric.

    This metric measures data minimality by clustering similar rows and
    calculating the ratio of unique clusters to total rows.

    :param use_semhash: If True, uses SemHash for semantic deduplication.
                        If False, uses custom type-aware similarity with hierarchical clustering.
                        Default: False
    :param similarity_threshold: Similarity threshold for clustering (0.0 to 1.0).
                                 Higher values require more similarity to group rows together.
                                 Default: 0.85
    """

    use_semhash: bool = False
    similarity_threshold: float = field(default=0.85)

    def __post_init__(self):
        if not isinstance(self.use_semhash, bool):
            raise ValueError(f"use_semhash must be boolean, got {type(self.use_semhash)}")

        if not isinstance(self.similarity_threshold, (int, float)):
            raise ValueError(f"similarity_threshold must be numeric, got {type(self.similarity_threshold)}")

        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}")

    def to_json(self):
        return {
            "name": self.__class__.__name__,
            "use_semhash": self.use_semhash,
            "similarity_threshold": self.similarity_threshold
        }
