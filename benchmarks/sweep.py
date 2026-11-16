"""Climb an input-size ladder per metric until it breaches its budget."""
from __future__ import annotations

import argparse
import csv
import pathlib

import metis.metric  # noqa: F401 - populates Metric.registry
from metis.metric.metric import Metric

from benchmarks import fixtures, measure

# 1-2-5 steps per decade so each curve has enough points to show its shape
# and to fit a growth exponent. It starts at 100 rather than 1k so that
# metrics with a low ceiling (minimality_clustering) still get several
# points before they breach.
ROW_LADDER: list[int] = [
    100, 200, 500,
    1_000, 2_000, 5_000,
    10_000, 20_000, 50_000,
    100_000, 200_000, 500_000,
    1_000_000,
]

# Multiples of 6, because datagen cycles through 6 column kinds. A column
# count that is not a multiple of 6 gives an uneven kind mix (5 columns has
# no sparse column at all, so no nulls anywhere), and the mix would then
# shift non-monotonically along the ladder, which reads as a complexity
# effect in the plots when it is really an artifact of the column count.
COL_LADDER: list[int] = [6, 12, 18, 24, 36, 48, 60, 78]

FIXED_COLS: int = 12
FIXED_ROWS: int = 10_000

DEFAULT_TIMEOUT_S: float = 120.0
DEFAULT_SEED: int = 13
DEFAULT_REPEATS: int = 3

_BREACH = ("timeout", "memory", "error")

VALID_AXES: tuple[str, ...] = ("rows", "cols")


def run_sweep(
    metrics: list[str],
    axes: tuple[str, ...] = ("rows", "cols"),
    timeout_s: float = DEFAULT_TIMEOUT_S,
    seed: int = DEFAULT_SEED,
    repeats: int = DEFAULT_REPEATS,
    on_result=None,
) -> list[measure.Measurement]:
    """
    Sweep each metric up its ladders, stopping that metric on a breach.

    A metric that times out at 100k rows is not retried at a million. The
    headline result is therefore the largest input each metric survived.

    Each point is measured ``repeats`` times, each in a fresh subprocess and
    with the same seed, so the spread across repeats is timing noise rather
    than a difference in data. A breach in any repeat counts as a breach of
    the point: the remaining repeats are skipped, so a timeout is waited
    out once rather than ``repeats`` times.

    A point that passes all its repeats gets one more, instrumented run that
    splits the runtime into computing and building result objects. That run
    is slowed by its own timers, so it is kept apart from the plain repeats
    and does not count towards a breach.

    :param metrics: Registry names to sweep.
    :param axes: Which ladders to climb, ``rows`` and/or ``cols``.
    :param timeout_s: Per-measurement wall-clock budget in seconds.
    :param seed: Seed for data generation.
    :param repeats: How many times to measure each point. Must be at least 1.
    :param on_result: Optional callback invoked with each Measurement.
    :return: Every measurement taken, in order.
    :raises ValueError: If ``repeats`` is below 1, or if any entry in
        ``axes`` is not ``rows`` or ``cols``.
        An unrecognised axis would silently fall through to the column
        ladder while the result is still labelled with the given axis,
        which produces mislabelled data with no visible error.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")
    bad = [a for a in axes if a not in VALID_AXES]
    if bad:
        raise ValueError(f"axes must be one of {VALID_AXES}, got {bad!r}")

    out: list[measure.Measurement] = []

    for metric in metrics:
        for axis in axes:
            ladder = ROW_LADDER if axis == "rows" else COL_LADDER
            for step in ladder:
                rows = step if axis == "rows" else FIXED_ROWS
                cols = FIXED_COLS if axis == "rows" else step

                breached = False
                for repeat in range(repeats):
                    result = measure.run_one(
                        metric=metric, rows=rows, cols=cols,
                        axis=axis, timeout_s=timeout_s, seed=seed,
                        repeat=repeat,
                    )
                    out.append(result)
                    if on_result:
                        on_result(result)
                    if result.status in _BREACH:
                        breached = True
                        break
                if breached:
                    break

                result = measure.run_one(
                    metric=metric, rows=rows, cols=cols,
                    axis=axis, timeout_s=timeout_s, seed=seed,
                    repeat=repeats, instrumented=True,
                )
                out.append(result)
                if on_result:
                    on_result(result)

    return out


def write_csv(measurements: list[measure.Measurement], path: pathlib.Path) -> None:
    """
    Write measurements to CSV so runs stay comparable over time.

    :param measurements: The measurements to write.
    :param path: Destination file. Parent directories are created.
    :return: None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=measure.Measurement.field_names())
        writer.writeheader()
        for m in measurements:
            writer.writerow(m.as_dict())


def main() -> None:
    """Command-line entry point for the sweep."""
    parser = argparse.ArgumentParser(description="Metis scalability sweep.")
    parser.add_argument("--out", default="benchmarks/results/sweep.csv")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help="Measure each point this many times (default 3).",
    )
    parser.add_argument("--axes", default="rows,cols")
    parser.add_argument(
        "--metric", action="append", default=None,
        help="Sweep only this metric. Repeatable. Defaults to all non-skipped.",
    )
    parser.add_argument(
        "--include-skipped", action="store_true",
        help="Also sweep metrics excluded by default (LLM and native-library metrics).",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error(f"--repeats must be at least 1, got {args.repeats}")

    axes = tuple(a.strip() for a in args.axes.split(",") if a.strip())
    bad_axes = [a for a in axes if a not in VALID_AXES]
    if bad_axes:
        parser.error(f"--axes accepts only {VALID_AXES}, got {bad_axes!r}")

    if args.metric:
        metrics = args.metric
    else:
        metrics = sorted(Metric.registry)
        if not args.include_skipped:
            for name, reason in sorted(fixtures.SKIPPED.items()):
                print(f"skipping {name}: {reason}")
            metrics = [m for m in metrics if m not in fixtures.SKIPPED]

    def report(m: measure.Measurement) -> None:
        if m.status == "ok":
            detail = (
                f"gen={m.generation_seconds:.2f}s assess={m.seconds:.2f}s "
                + (f"(results={m.result_seconds:.2f}s) " if m.instrumented else "")
                + f"baseline={m.baseline_mb:.0f}MB peak={m.peak_mb:.0f}MB"
            )
        else:
            detail = m.error
        size = m.rows if m.axis == "rows" else m.cols
        run = "timed" if m.instrumented else f"#{m.repeat}"
        print(f"{m.status:<8} {m.metric:<32} {m.axis}={size:<9} {run:<5} {detail}")

    results = run_sweep(
        metrics=metrics,
        axes=axes,
        timeout_s=args.timeout,
        seed=args.seed,
        repeats=args.repeats,
        on_result=report,
    )

    out = pathlib.Path(args.out)
    write_csv(results, out)
    print(f"\nwrote {len(results)} measurements to {out}")


if __name__ == "__main__":
    main()
