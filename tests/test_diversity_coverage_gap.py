from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from metis.metric.diversity.coverage_space_counter import CoverageSpaceCounter
from metis.metric.diversity.diversity_coverageGap import diversity_coverageGap
from metis.utils.dq_dimension import DQDimension
from metis.utils.dq_granularity import DQGranularity


class CoverageSpaceCounterTest(unittest.TestCase):
    def test_single_mup_specialization_volume(self) -> None:
        score = CoverageSpaceCounter([2, 2]).calculate([[(0, 0)]])
        self.assertEqual(score.uncovered_patterns, 3)
        self.assertEqual(score.total_patterns, 9)
        self.assertAlmostEqual(score.coverage_gap, 1 / 3)

    def test_overlapping_mups_are_counted_once(self) -> None:
        score = CoverageSpaceCounter([2, 2]).calculate([
            [(0, 0)],
            [(1, 0)],
        ])
        self.assertEqual(score.uncovered_patterns, 5)
        self.assertEqual(score.total_patterns, 9)

    def test_redundant_specialized_term_is_removed(self) -> None:
        score = CoverageSpaceCounter([2, 2]).calculate([
            [(0, 0)],
            [(0, 0), (1, 0)],
            [(0, 0)],
        ])
        self.assertEqual(score.uncovered_patterns, 3)
        self.assertEqual(score.effective_term_count, 1)


class DiversityCoverageGapMetricTest(unittest.TestCase):
    def test_metric_reports_gap_and_higher_is_better_score(self) -> None:
        data = pd.DataFrame({
            "a": ["red", "blue"],
            "b": ["circle", "square"],
            "id": [1, 2],
        })
        config = json.dumps({
            "mups_content": "red,x,1\nx,circle,1\n",
            "mups_filename": "mups_example_mincov_2.txt",
            "attributes": ["a", "b"],
            "mincov": 2,
        })

        result = diversity_coverageGap().assess(data, metric_config=config)[0]

        self.assertEqual(result.DQdimension, DQDimension.DIVERSITY)
        self.assertEqual(result.DQgranularity, DQGranularity.TABLE)
        self.assertAlmostEqual(result.DQexplanation["coverage_gap"], 5 / 9)
        self.assertAlmostEqual(result.DQvalue, 4 / 9)
        self.assertEqual(result.DQexplanation["uncovered_patterns"], "5")
        self.assertEqual(result.DQexplanation["mup_coverage_min"], 1)
        self.assertEqual(result.DQexplanation["mup_coverage_max"], 1)
        self.assertEqual(result.DQexplanation["ignored_intermediate_fields_per_mup"], 0)

    def test_file_config_and_numeric_normalization(self) -> None:
        data = pd.DataFrame({"a": [1.0, 2.0], "b": [0, 1]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mups.txt"
            path.write_text("1,x,0\n", encoding="utf-8")
            config = json.dumps({
                "mups_path": str(path),
                "attributes": ["a", "b"],
            })
            result = diversity_coverageGap().assess(data, metric_config=config)[0]

        self.assertAlmostEqual(result.DQexplanation["coverage_gap"], 1 / 3)

    def test_unknown_mup_value_is_rejected(self) -> None:
        data = pd.DataFrame({"a": ["red"], "b": ["circle"]})
        config = json.dumps({
            "mups_content": "green,x,0\n",
            "attributes": ["a", "b"],
        })
        with self.assertRaisesRegex(ValueError, "Unknown value 'green'"):
            diversity_coverageGap().assess(data, metric_config=config)

    def test_empty_mup_frontier_means_no_coverage_gap(self) -> None:
        data = pd.DataFrame({"a": ["red", "blue"], "b": ["circle", "square"]})
        config = json.dumps({
            "mups_content": "",
            "attributes": ["a", "b"],
            "mincov": 1,
        })
        result = diversity_coverageGap().assess(data, metric_config=config)[0]

        self.assertEqual(result.DQexplanation["uncovered_patterns"], "0")
        self.assertEqual(result.DQexplanation["coverage_gap"], 0.0)
        self.assertEqual(result.DQvalue, 1.0)

    def test_mup_coverage_must_be_below_mincov(self) -> None:
        data = pd.DataFrame({"a": ["red"], "b": ["circle"]})
        config = json.dumps({
            "mups_content": "red,x,2\n",
            "attributes": ["a", "b"],
            "mincov": 2,
        })
        with self.assertRaisesRegex(ValueError, "is not below mincov 2"):
            diversity_coverageGap().assess(data, metric_config=config)


if __name__ == "__main__":
    unittest.main()
