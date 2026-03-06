"""Data profile importers registry."""

from typing import Dict

from .base import BaseImporter
from .fd_importer import FDImporter
from .histogram_importer import HistogramImporter
from .ind_importer import INDImporter
from .jaccard_importer import JaccardImporter
from .patterns_importer import PatternsImporter
from .quartiles_importer import QuartilesImporter
from .scalar_importer import ScalarImporter, create_scalar_importers
from .ucc_importer import UCCImporter


def _build_registry() -> Dict[str, BaseImporter]:
    """Build the complete importer registry."""
    registry: Dict[str, BaseImporter] = {}

    # Scalar tasks (column, value)
    registry.update(create_scalar_importers())

    # Histogram tasks
    registry["equi_width_histogram"] = HistogramImporter("equi_width_histogram")
    registry["equi_depth_histogram"] = HistogramImporter("equi_depth_histogram")

    # Other complex tasks
    registry["patterns"] = PatternsImporter()
    registry["quartiles"] = QuartilesImporter()

    # 2-column tasks
    registry["jaccard_similarity"] = JaccardImporter("jaccard_similarity")
    registry["jaccard_similarity_ngrams"] = JaccardImporter("jaccard_similarity_ngrams")

    # Dependencies
    registry["fd"] = FDImporter()
    registry["ucc"] = UCCImporter()
    registry["ind"] = INDImporter()

    return registry


IMPORTER_REGISTRY: Dict[str, BaseImporter] = _build_registry()


def get_importer(task_name: str) -> BaseImporter:
    """Get the importer for a given task name.

    Raises:
        KeyError: If no importer exists for the task
    """
    if task_name not in IMPORTER_REGISTRY:
        raise KeyError(
            f"No importer registered for task '{task_name}'. "
            f"Available tasks: {list(IMPORTER_REGISTRY.keys())}"
        )
    return IMPORTER_REGISTRY[task_name]


__all__ = [
    "BaseImporter",
    "IMPORTER_REGISTRY",
    "get_importer",
    "FDImporter",
    "HistogramImporter",
    "INDImporter",
    "JaccardImporter",
    "PatternsImporter",
    "QuartilesImporter",
    "ScalarImporter",
    "UCCImporter",
]
