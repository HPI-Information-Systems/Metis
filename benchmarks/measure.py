"""Run one metric at one input size in an isolated subprocess.

Isolation gives a clean peak-RSS figure per measurement, and means an
out-of-memory kill or an unhandled crash is recorded as a result rather than
taking down the whole sweep. On a test whose purpose is finding limits,
failures are the data.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, fields

_CHILD = r"""
import json, pathlib, resource, sys, tempfile, time

import metis.metric  # noqa: F401
from metis.metric.metric import Metric
from benchmarks import datagen, fixtures

metric, rows, cols, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

try:
    cls = Metric.registry.get(metric)
    if cls is None:
        raise KeyError(f"{metric} is not a registered metric")
    frame = datagen.make_frame(rows=rows, cols=cols, seed=seed)
    work = pathlib.Path(tempfile.mkdtemp())
    config = fixtures.config_for(metric, frame, work)

    start = time.perf_counter()
    results = cls().assess(frame, config)
    seconds = time.perf_counter() - start

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes, macOS reports bytes.
    peak_mb = peak / 1024 if sys.platform.startswith("linux") else peak / (1024 * 1024)

    print("METIS_BENCH " + json.dumps({
        "status": "ok", "seconds": seconds,
        "peak_mb": peak_mb, "n_results": len(results), "error": None,
    }))
except MemoryError as e:
    print("METIS_BENCH " + json.dumps({
        "status": "memory", "seconds": None, "peak_mb": None,
        "n_results": None, "error": str(e),
    }))
except BaseException as e:
    print("METIS_BENCH " + json.dumps({
        "status": "error", "seconds": None, "peak_mb": None,
        "n_results": None, "error": f"{type(e).__name__}: {e}",
    }))
"""


@dataclass
class Measurement:
    """One metric measured at one input size."""

    metric: str
    axis: str
    rows: int
    cols: int
    status: str
    seconds: float | None
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
) -> Measurement:
    """
    Measure one metric at one input size in a fresh subprocess.

    :param metric: Registry name of the metric.
    :param rows: Row count for the generated frame.
    :param cols: Column count for the generated frame.
    :param axis: Which ladder this measurement belongs to, ``rows`` or ``cols``.
    :param timeout_s: Wall-clock budget. Exceeding it yields status ``timeout``.
    :param seed: Seed for data generation.
    :return: The :class:`Measurement`.
    """
    base = dict(metric=metric, axis=axis, rows=rows, cols=cols)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, metric, str(rows), str(cols), str(seed)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return Measurement(
            **base, status="timeout", seconds=None, peak_mb=None,
            n_results=None, error=f"exceeded {timeout_s}s",
        )

    payload = _parse(proc.stdout)
    if payload is None:
        # Killed by a signal (an OOM kill is the common case) or crashed before
        # it could report.
        status = "memory" if proc.returncode and proc.returncode < 0 else "error"
        detail = (proc.stderr or "").strip()[-500:] or f"exit code {proc.returncode}"
        return Measurement(
            **base, status=status, seconds=None, peak_mb=None,
            n_results=None, error=detail,
        )

    return Measurement(**base, **payload)


def _parse(stdout: str) -> dict | None:
    """Extract the child's JSON payload from its stdout."""
    for line in reversed(stdout.splitlines()):
        if line.startswith("METIS_BENCH "):
            return json.loads(line[len("METIS_BENCH "):])
    return None
