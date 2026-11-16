import json
import pathlib

import pandas as pd
import pytest

from benchmarks import datagen, fixtures
from metis.metric.metric import Metric


def test_frame_has_the_requested_shape():
    df = datagen.make_frame(rows=100, cols=8)
    assert df.shape == (100, 8)


def test_generation_is_deterministic_for_a_seed():
    a = datagen.make_frame(rows=50, cols=6, seed=7)
    b = datagen.make_frame(rows=50, cols=6, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_produce_different_data():
    a = datagen.make_frame(rows=50, cols=6, seed=1)
    b = datagen.make_frame(rows=50, cols=6, seed=2)
    assert not a.equals(b)


def test_frame_mixes_column_kinds():
    # Assert the property, not a dtype spelling: pandas 2 stores text columns
    # as "object" while pandas 3 stores them as "str".
    df = datagen.make_frame(rows=40, cols=8)
    dtypes = {str(d) for d in df.dtypes}
    assert any(d.startswith("int") or d.startswith("float") for d in dtypes)
    assert any(d.startswith(("object", "str")) for d in dtypes)


def test_frame_contains_nulls_and_duplicates_so_metrics_have_signal():
    df = datagen.make_frame(rows=200, cols=8)
    assert df.isna().any().any()
    assert df.duplicated().any()


def test_a_tiny_column_count_still_works():
    assert datagen.make_frame(rows=10, cols=1).shape == (10, 1)


# Metric.__init_subclass__ registers EVERY subclass globally and forever, so
# test doubles defined in tests/test_metric_base.py (_Described, _Undescribed)
# are in the registry too once that module has been collected. This benchmark
# is about the project's real metrics, so private names are excluded (see
# tests/test_metric_meta_conformance.py for the same fix and rationale).
@pytest.mark.parametrize("name", sorted(n for n in Metric.registry if not n.startswith("_")))
def test_every_metric_has_a_fixture_or_a_skip_reason(name, tmp_path):
    if name in fixtures.SKIPPED:
        assert isinstance(fixtures.SKIPPED[name], str)
        return
    frame = datagen.make_frame(rows=30, cols=6)
    config = fixtures.config_for(name, frame, tmp_path)
    assert config is not None or not Metric.registry[name].describe().config_required


# See the comment on test_every_metric_has_a_fixture_or_a_skip_reason above:
# private names are excluded to keep out permanently-registered test doubles.
@pytest.mark.parametrize("name", sorted(n for n in Metric.registry if not n.startswith("_")))
def test_every_non_skipped_metric_produces_nonempty_results(name, tmp_path):
    """
    A metric that raises no exception but returns zero results is not
    actually benchmarked: it measures the cost of skipping every column
    rather than the cost of doing the metric's work. This caught
    accuracy_dataRange and accuracy_syntacticDomain silently no-op-ing on
    their default configs.
    """
    if name in fixtures.SKIPPED:
        pytest.skip(fixtures.SKIPPED[name])

    frame = datagen.make_frame(rows=30, cols=6)
    config = fixtures.config_for(name, frame, tmp_path)
    results = Metric.registry[name]().assess(frame, metric_config=config)
    assert len(results) > 0, f"{name} produced no results with its benchmark fixture"


def test_correctness_reference_is_not_identical_to_the_data(tmp_path):
    """
    ``correctness_heinrich.measure_correctness`` starts with
    ``if value == reference_value: return 1``, short-circuiting before both
    its numeric-distance branch and its Levenshtein-distance branch. A
    reference built as ``frame.copy()`` would make every cell take that
    shortcut, so the sweep would time an equality scan instead of the
    metric's real comparison work. Guard against that regression directly.
    """
    frame = datagen.make_frame(rows=100, cols=6)
    config = fixtures.config_for("correctness_heinrich", frame, tmp_path)
    reference = config.reference

    assert reference.shape == frame.shape

    identical = frame.astype(str).to_numpy() == reference.astype(str).to_numpy()
    assert identical.any(), "reference should still match the data on some cells"
    assert not identical.all(), "correctness reference must not be identical to the data"
    assert identical.mean() < 0.9, (
        "correctness reference is nearly identical to the data; the metric's "
        "numeric/string comparison branches would barely run"
    )


def test_fd_fixture_determinant_has_high_cardinality(tmp_path):
    """
    A low-cardinality determinant (e.g. a 5-valued categorical column)
    paired with an independently-generated dependent guarantees that nearly
    every group violates the functional dependency: ``DQvalue`` collapses
    to ~0 and the metric serialises almost the whole table into
    ``DQexplanation`` (measured at 151KB for a 500-row frame). Guard against
    that regression by requiring the configured determinant to be close to
    a key.
    """
    frame = datagen.make_frame(rows=500, cols=6)
    path = fixtures.config_for("consistency_countFDViolations", frame, tmp_path)
    spec = json.loads(pathlib.Path(path).read_text())
    (determinant, _dependents), = spec.items()

    # The generator's DUPLICATE_FRACTION (10%) caps any column's cardinality
    # ratio at ~0.9, since duplicate rows are exact copies. 0.8 leaves
    # headroom below that ceiling while still failing hard for a
    # low-cardinality determinant like the 5-valued categorical column
    # (ratio ~0.01).
    cardinality_ratio = frame[determinant].nunique() / len(frame)
    assert cardinality_ratio > 0.8, (
        f"FD determinant '{determinant}' has low cardinality "
        f"({cardinality_ratio:.2f} of rows); most groups would violate the "
        "dependency by construction rather than because the metric found "
        "something"
    )


def test_fd_fixture_determinant_stays_near_unique_at_sweep_scale(tmp_path):
    """
    The integer column's value range used to be fixed at 1,000,000
    regardless of row count. As row count approaches and passes that floor,
    birthday-paradox collisions erode the near-key property the FD fixture
    relies on: measured at a 0.823 cardinality ratio at 200,000 rows with a
    fixed range (and 79MB of DQexplanation / DQvalue=0.407 at a million
    rows). ``integer_upper_bound`` scales the range with row count so the
    ratio stays high at every sweep size, not just small ones -- this test
    checks at 200,000 rows specifically, near the top of the sweep's range,
    because that is exactly where the first version of this fix (a fixed
    high-cardinality column, chosen without checking at scale) silently
    broke again.
    """
    frame = datagen.make_frame(rows=200_000, cols=6)
    path = fixtures.config_for("consistency_countFDViolations", frame, tmp_path)
    spec = json.loads(pathlib.Path(path).read_text())
    (determinant, _dependents), = spec.items()

    # 0.85 sits strictly between the fixed-range failure (0.823, measured)
    # and the scaled-range result (~0.878, measured), so this fails today
    # against the fixed-range generator and passes against the scaled one.
    cardinality_ratio = frame[determinant].nunique() / len(frame)
    assert cardinality_ratio > 0.85, (
        f"FD determinant '{determinant}' cardinality ratio "
        f"({cardinality_ratio:.4f}) collapsed at sweep scale; the integer "
        "column's value range must grow with row count"
    )


def test_text_vocabulary_grows_with_row_count():
    """
    The two-word text columns are built from a word pool. A fixed 16-word
    pool caps distinct cell values at 256, which saturates any per-token
    cache (e.g. readability_wordnet's WordNet lookups) within a few thousand
    rows and makes the metric look far more scalable than it is on real
    text. Vocabulary must keep growing with row count instead of plateauing.
    """
    small = datagen.make_frame(rows=500, cols=6)["text_3"].nunique()
    large = datagen.make_frame(rows=5000, cols=6)["text_3"].nunique()

    assert small > 256, "text vocabulary is stuck at the old 16-word cap"
    assert large > small, "text vocabulary must keep growing with row count"
