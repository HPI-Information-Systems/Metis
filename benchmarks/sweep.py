"""Climb an input-size ladder per metric until it breaches its budget."""
from __future__ import annotations

import argparse
import csv
import pathlib

import metis.metric  # noqa: F401 - populates Metric.registry
from metis.metric.metric import Metric

from benchmarks import fixtures, measure

ROW_LADDER: list[int] = [1_000, 10_000, 100_000, 1_000_000]

# Multiples of 6, because datagen cycles through 6 column kinds. A column
# count that is not a multiple of 6 gives an uneven kind mix (5 columns has
# no sparse column at all, so no nulls anywhere), and the mix would then
# shift non-monotonically along the ladder, which reads as a complexity
# effect in the plots when it is really an artifact of the column count.
COL_LADDER: list[int] = [6, 12, 24, 48, 78]

FIXED_COLS: int = 12
FIXED_ROWS: int = 10_000

DEFAULT_TIMEOUT_S: float = 120.0
DEFAULT_SEED: int = 13

_BREACH = ("timeout", "memory", "error")

VALID_AXES: tuple[str, ...] = ("rows", "cols")


def run_sweep(
    metrics: list[str],
    axes: tuple[str, ...] = ("rows", "cols"),
    timeout_s: float = DEFAULT_TIMEOUT_S,
    seed: int = DEFAULT_SEED,
    on_result=None,
) -> list[measure.Measurement]:
    """
    Sweep each metric up its ladders, stopping that metric on a breach.

    A metric that times out at 100k rows is not retried at a million. The
    headline result is therefore the largest input each metric survived.

    :param metrics: Registry names to sweep.
    :param axes: Which ladders to climb, ``rows`` and/or ``cols``.
    :param timeout_s: Per-measurement wall-clock budget in seconds.
    :param seed: Seed for data generation.
    :param on_result: Optional callback invoked with each Measurement.
    :return: Every measurement taken, in order.
    :raises ValueError: If any entry in ``axes`` is not ``rows`` or ``cols``.
        An unrecognised axis would silently fall through to the column
        ladder while the result is still labelled with the given axis,
        which produces mislabelled data with no visible error.
    """
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

                result = measure.run_one(
                    metric=metric, rows=rows, cols=cols,
                    axis=axis, timeout_s=timeout_s, seed=seed,
                )
                out.append(result)
                if on_result:
                    on_result(result)
                if result.status in _BREACH:
                    break

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
                f"baseline={m.baseline_mb:.0f}MB peak={m.peak_mb:.0f}MB"
            )
        else:
            detail = m.error
        size = m.rows if m.axis == "rows" else m.cols
        print(f"{m.status:<8} {m.metric:<32} {m.axis}={size:<9} {detail}")

    results = run_sweep(
        metrics=metrics,
        axes=axes,
        timeout_s=args.timeout,
        seed=args.seed,
        on_result=report,
    )

    out = pathlib.Path(args.out)
    write_csv(results, out)
    print(f"\nwrote {len(results)} measurements to {out}")


if __name__ == "__main__":
    main()
