"""Run one metric at one input size in an isolated subprocess.

Isolation gives a clean peak-RSS figure per measurement, and means an
out-of-memory kill or an unhandled crash is recorded as a result rather than
taking down the whole sweep. On a test whose purpose is finding limits,
failures are the data.
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
from dataclasses import asdict, dataclass, fields

_CHILD = r"""
import json, pathlib, resource, sys, tempfile, time

import pandas as pd

import metis.metric  # noqa: F401
from metis.metric.metric import Metric
from metis.utils.result import DQResult
from benchmarks import datagen, fixtures

metric, rows, cols, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
instrumented = sys.argv[5] == "1"


def _mb(raw):
    # Linux reports kilobytes, macOS reports bytes.
    return raw / 1024 if sys.platform.startswith("linux") else raw / (1024 * 1024)


# Time spent building result objects, accumulated across every DQResult the
# metric creates. Metrics build one result per cell, row or column, so at
# cell granularity this can rival or dwarf the actual computation. Every
# metric passes timestamp=pd.Timestamp.now() when it builds a result, and
# that call runs before __init__, so it is timed here as well. The one
# other caller, timeliness_heinrich's default assessment date, runs once
# per column and is negligible.
#
# The wrappers cost up to ~20% of runtime on metrics that build a million
# results, so they are installed only for the dedicated instrumented run.
# The plain runs stay unwrapped and provide the headline timings.
_result_seconds = 0.0
_orig_init = DQResult.__init__
_orig_now = pd.Timestamp.now


def _timed_init(self, *args, **kwargs):
    global _result_seconds
    t = time.perf_counter()
    _orig_init(self, *args, **kwargs)
    _result_seconds += time.perf_counter() - t


def _timed_now(cls, tz=None):
    global _result_seconds
    t = time.perf_counter()
    ts = _orig_now(tz)
    _result_seconds += time.perf_counter() - t
    return ts


if instrumented:
    DQResult.__init__ = _timed_init
    pd.Timestamp.now = classmethod(_timed_now)


try:
    cls = Metric.registry.get(metric)
    if cls is None:
        raise KeyError(f"{metric} is not a registered metric")

    gen_start = time.perf_counter()
    frame = datagen.make_frame(rows=rows, cols=cols, seed=seed)
    work = pathlib.Path(tempfile.mkdtemp())
    config = fixtures.config_for(metric, frame, work)
    generation_seconds = time.perf_counter() - gen_start

    # Captured right before assess() runs, so it reads as interpreter
    # startup plus generation, not as metric cost. peak_mb below is a
    # high-water mark over the whole process, so it is reported alongside
    # rather than replaced by a peak-minus-baseline delta: ru_maxrss never
    # falls, so a naive subtraction reads zero whenever generation peaked
    # higher than the metric itself.
    baseline_mb = _mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    start = time.perf_counter()
    results = cls().assess(frame, metric_config=config)
    seconds = time.perf_counter() - start

    peak_mb = _mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    print("METIS_BENCH " + json.dumps({
        "status": "ok", "seconds": seconds, "generation_seconds": generation_seconds,
        "result_seconds": _result_seconds if instrumented else None,
        "compute_seconds": seconds - _result_seconds if instrumented else None,
        "baseline_mb": baseline_mb, "peak_mb": peak_mb,
        "n_results": len(results), "error": None,
    }))
except MemoryError as e:
    print("METIS_BENCH " + json.dumps({
        "status": "memory", "seconds": None, "generation_seconds": None,
        "result_seconds": None, "compute_seconds": None,
        "baseline_mb": None, "peak_mb": None, "n_results": None, "error": str(e),
    }))
except BaseException as e:
    print("METIS_BENCH " + json.dumps({
        "status": "error", "seconds": None, "generation_seconds": None,
        "result_seconds": None, "compute_seconds": None,
        "baseline_mb": None, "peak_mb": None, "n_results": None,
        "error": f"{type(e).__name__}: {e}",
    }))
"""


@dataclass
class Measurement:
    """One metric measured at one input size."""

    metric: str
    axis: str
    rows: int
    cols: int
    repeat: int
    instrumented: bool
    status: str
    seconds: float | None
    generation_seconds: float | None
    result_seconds: float | None
    compute_seconds: float | None
    baseline_mb: float | None
    peak_mb: float | None
    n_results: int | None
    error: str | None

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def as_dict(self) -> dict:
        return asdict(self)


def run_one(
    metric: str,
    rows: int,
    cols: int,
    axis: str,
    timeout_s: float,
    seed: int,
    repeat: int = 0,
    instrumented: bool = False,
) -> Measurement:
    """
    Measure one metric at one input size in a fresh subprocess.

    :param metric: Registry name of the metric.
    :param rows: Row count for the generated frame.
    :param cols: Column count for the generated frame.
    :param axis: Which ladder this measurement belongs to, ``rows`` or ``cols``.
    :param timeout_s: Wall-clock budget. Exceeding it yields status ``timeout``.
    :param seed: Seed for data generation.
    :param repeat: Index of this run among repeated runs of the same point.
    :param instrumented: Time result-object construction separately. This
        adds overhead, so an instrumented run's ``seconds`` is not comparable
        with a plain run's and is kept out of the headline figures.
    :return: The :class:`Measurement`.
    """
    base = dict(
        metric=metric, axis=axis, rows=rows, cols=cols,
        repeat=repeat, instrumented=instrumented,
    )

    try:
        proc = subprocess.run(
            [
                sys.executable, "-c", _CHILD, metric, str(rows), str(cols),
                str(seed), "1" if instrumented else "0",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return Measurement(
            **base, status="timeout", seconds=None, generation_seconds=None,
            result_seconds=None, compute_seconds=None,
            baseline_mb=None, peak_mb=None, n_results=None,
            error=f"exceeded {timeout_s}s",
        )

    payload = _parse(proc.stdout)
    if payload is None:
        status, detail = _classify_signal_death(proc.returncode, proc.stderr)
        return Measurement(
            **base, status=status, seconds=None, generation_seconds=None,
            result_seconds=None, compute_seconds=None,
            baseline_mb=None, peak_mb=None, n_results=None, error=detail,
        )

    return Measurement(**base, **payload)


def _classify_signal_death(returncode: int | None, stderr: str | None) -> tuple[str, str]:
    """
    Classify a child that exited without printing a payload.

    Only SIGKILL (return code -9) is the signature of an OOM kill, so only
    that signal is reported as ``memory``. Any other signal (SIGSEGV,
    SIGABRT, ...) is a crash, not evidence of a memory limit, so it is
    reported as ``error`` with the signal named -- misfiling e.g. a native
    library's SIGSEGV as "memory" would corrupt the sweep's headline finding.

    :param returncode: The child's exit code, negative when killed by a signal.
    :param stderr: The child's captured stderr, used for extra detail.
    :return: A ``(status, error_detail)`` pair.
    """
    tail = (stderr or "").strip()[-500:]

    if returncode is not None and returncode < 0:
        sig = -returncode
        if sig == signal.SIGKILL:
            return "memory", tail or "killed by SIGKILL (likely an OOM kill)"
        try:
            sig_name = signal.Signals(sig).name
        except ValueError:
            sig_name = f"signal {sig}"
        detail = f"killed by {sig_name}"
        if tail:
            detail = f"{detail}: {tail}"
        return "error", detail

    return "error", tail or f"exit code {returncode}"


def _parse(stdout: str) -> dict | None:
    """Extract the child's JSON payload from its stdout."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("METIS_BENCH "):
            return json.loads(line[len("METIS_BENCH "):])
    return None
