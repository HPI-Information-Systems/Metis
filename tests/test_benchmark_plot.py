import pytest

from benchmarks import measure, plot, sweep


def _m(metric, axis, rows, cols, status, seconds, peak_mb, n_results, error):
    """Build a Measurement, filling the fields the brief's shorthand omits."""
    return measure.Measurement(
        metric=metric, axis=axis, rows=rows, cols=cols,
        repeat=0, instrumented=False, status=status,
        seconds=seconds, generation_seconds=0.05 if seconds is not None else None,
        result_seconds=None, compute_seconds=None,
        baseline_mb=peak_mb * 0.5 if peak_mb is not None else None,
        peak_mb=peak_mb, n_results=n_results, error=error,
    )


@pytest.fixture
def csv_file(tmp_path):
    rows = [
        _m("m_fast", "rows", 1_000, 12, "ok", 0.1, 50.0, 12, None),
        _m("m_fast", "rows", 10_000, 12, "ok", 0.8, 90.0, 12, None),
        _m("m_slow", "rows", 1_000, 12, "ok", 5.0, 120.0, 12, None),
        _m("m_slow", "rows", 10_000, 12, "timeout", None, None, None, "exceeded 120s"),
    ]
    path = tmp_path / "sweep.csv"
    sweep.write_csv(rows, path)
    return path


def test_load_parses_numeric_columns(csv_file):
    df = plot.load(csv_file)
    assert df["rows"].dtype.kind in "iu"
    assert df["seconds"].dtype.kind == "f"


def test_limits_table_reports_the_largest_surviving_size(csv_file):
    limits = plot.limits_table(plot.load(csv_file)).set_index("metric")
    assert limits.loc["m_fast", "max_rows_ok"] == 10_000
    assert limits.loc["m_slow", "max_rows_ok"] == 1_000
    assert limits.loc["m_slow", "stopped_by"] == "timeout"


def test_render_all_writes_png_files(csv_file, tmp_path):
    written = plot.render_all(plot.load(csv_file), tmp_path / "figs")
    assert written
    assert all(p.suffix == ".png" and p.exists() for p in written)


def test_limits_table_breaks_ties_alphabetically(tmp_path):
    # Three metrics tied on max_rows_ok, listed out of alphabetical order,
    # so a non-stable sort has room to shuffle them. groupby(sort=True)
    # already hands limits_table alphabetically ordered input, so this only
    # catches a regression if the sort_values call stops being stable.
    rows = [
        _m("m_charlie", "rows", 10_000, 12, "ok", 1.0, 50.0, 12, None),
        _m("m_alpha", "rows", 10_000, 12, "ok", 1.0, 50.0, 12, None),
        _m("m_bravo", "rows", 10_000, 12, "ok", 1.0, 50.0, 12, None),
    ]
    path = tmp_path / "sweep.csv"
    sweep.write_csv(rows, path)

    limits = plot.limits_table(plot.load(path))
    assert list(limits["metric"]) == ["m_alpha", "m_bravo", "m_charlie"]


def test_limits_table_reports_zero_for_a_metric_that_never_ran(tmp_path):
    # A metric whose every measurement fails on the smallest rung of the
    # ladder never hit a scaling limit, it never ran at all. That has to be
    # distinguishable from a metric that scaled and then broke, which is
    # what max_rows_ok == 0 (rather than some genuine small size) signals.
    rows = [
        _m("m_broken", "rows", 1_000, 12, "error", None, None, None, "ImportError: no fahes"),
    ]
    path = tmp_path / "sweep.csv"
    sweep.write_csv(rows, path)

    limits = plot.limits_table(plot.load(path)).set_index("metric")
    assert limits.loc["m_broken", "max_rows_ok"] == 0
    assert limits.loc["m_broken", "stopped_by"] == "error"
