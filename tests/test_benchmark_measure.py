import signal
import subprocess

import pytest

from benchmarks import measure, sweep


def test_a_cheap_metric_measures_ok():
    m = measure.run_one(
        metric="completeness_nullRatio", rows=500, cols=6,
        axis="rows", timeout_s=120, seed=13,
    )
    assert m.status == "ok"
    assert m.seconds > 0
    assert m.generation_seconds > 0
    assert m.baseline_mb > 0
    assert m.peak_mb > 0
    assert m.n_results > 0


def test_an_unknown_metric_is_reported_as_an_error_not_a_crash():
    m = measure.run_one(
        metric="does_not_exist", rows=100, cols=4,
        axis="rows", timeout_s=30, seed=13,
    )
    assert m.status == "error"
    assert m.error


def test_a_tight_timeout_is_reported_as_a_timeout(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="fake", timeout=kwargs.get("timeout"))

    monkeypatch.setattr(measure.subprocess, "run", fake_run)
    m = measure.run_one(
        metric="minimality_clustering", rows=4000, cols=6,
        axis="rows", timeout_s=0.5, seed=13,
    )
    assert m.status == "timeout"
    assert m.error


def test_a_sigkill_is_reported_as_memory(monkeypatch):
    class FakeCompletedProcess:
        returncode = -signal.SIGKILL
        stdout = ""
        stderr = ""

    monkeypatch.setattr(measure.subprocess, "run", lambda *a, **k: FakeCompletedProcess())
    m = measure.run_one(
        metric="completeness_nullRatio", rows=1, cols=1,
        axis="rows", timeout_s=5, seed=13,
    )
    assert m.status == "memory"


def test_a_sigsegv_is_reported_as_error_naming_the_signal(monkeypatch):
    class FakeCompletedProcess:
        returncode = -signal.SIGSEGV
        stdout = ""
        stderr = ""

    monkeypatch.setattr(measure.subprocess, "run", lambda *a, **k: FakeCompletedProcess())
    m = measure.run_one(
        metric="completeness_nullRatio", rows=1, cols=1,
        axis="rows", timeout_s=5, seed=13,
    )
    assert m.status == "error"
    assert "SIGSEGV" in m.error


def test_ladders_are_increasing():
    assert sweep.ROW_LADDER == sorted(sweep.ROW_LADDER)
    assert sweep.COL_LADDER == sorted(sweep.COL_LADDER)


def test_an_unrecognised_axis_raises_rather_than_silently_running_cols():
    with pytest.raises(ValueError):
        sweep.run_sweep(
            metrics=["completeness_nullRatio"], axes=("row",),
            timeout_s=10, seed=13,
        )


def test_sweep_stops_a_metric_after_a_breach(monkeypatch):
    calls = []

    def fake_run_one(metric, rows, cols, axis, timeout_s, seed, repeat=0, instrumented=False):
        calls.append(rows)
        status = "ok" if rows <= sweep.ROW_LADDER[0] else "timeout"
        return measure.Measurement(
            metric=metric, axis=axis, rows=rows, cols=cols,
            repeat=repeat, instrumented=instrumented,
            status=status, seconds=1.0, generation_seconds=0.1,
            result_seconds=None, compute_seconds=None,
            baseline_mb=1.0, peak_mb=1.0, n_results=1, error=None,
        )

    monkeypatch.setattr(measure, "run_one", fake_run_one)
    results = sweep.run_sweep(
        metrics=["completeness_nullRatio"], axes=("rows",),
        timeout_s=10, seed=13, repeats=1,
    )
    # The first rung is measured once plus one instrumented run, the second
    # rung breaches on its first measurement and ends the ladder.
    assert calls == [sweep.ROW_LADDER[0], sweep.ROW_LADDER[0], sweep.ROW_LADDER[1]]
    assert [r.status for r in results] == ["ok", "ok", "timeout"]


def test_write_csv_round_trips(tmp_path):
    import csv

    rows = [
        measure.Measurement(
            metric="m", axis="rows", rows=10, cols=4,
            repeat=0, instrumented=False,
            status="ok", seconds=0.5, generation_seconds=0.1,
            result_seconds=None, compute_seconds=None,
            baseline_mb=8.0, peak_mb=12.0, n_results=4, error=None,
        )
    ]
    out = tmp_path / "results.csv"
    sweep.write_csv(rows, out)
    parsed = list(csv.DictReader(out.open(encoding="utf-8")))
    assert parsed[0]["metric"] == "m"
    assert parsed[0]["status"] == "ok"
    assert parsed[0]["generation_seconds"] == "0.1"
    assert parsed[0]["baseline_mb"] == "8.0"
