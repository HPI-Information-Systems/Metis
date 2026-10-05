"""Turn a sweep CSV into limit tables and runtime/memory plots."""
from __future__ import annotations

import math
import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import EngFormatter  # noqa: E402
from scipy import stats  # noqa: E402

_NUMERIC = (
    "rows", "cols", "repeat", "seconds", "generation_seconds",
    "result_seconds", "compute_seconds", "baseline_mb", "peak_mb", "n_results",
)

_HUES: tuple[str, ...] = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7",
)
_STYLES: tuple[tuple[str, str], ...] = (("-", "o"), ("--", "s"))

SCALES: tuple[str, ...] = ("log", "linear")

FIT_DECADES: float = 1.0
MIN_FIT_POINTS: int = 3


def load(path: pathlib.Path) -> pd.DataFrame:
    """
    Read a sweep CSV with correct dtypes.

    CSVs written before repeats and instrumented runs existed have none of
    their columns. They are read as a single plain repeat with no timing
    breakdown, so older sweeps still plot.

    :param path: Path to the CSV written by ``sweep.write_csv``.
    :return: The measurements as a DataFrame.
    """
    df = pd.read_csv(path)
    if "repeat" not in df.columns:
        df["repeat"] = 0
    if "instrumented" not in df.columns:
        df["instrumented"] = False
    df["instrumented"] = df["instrumented"].astype(str).str.lower() == "true"
    for col in _NUMERIC:
        if col not in df.columns:
            df[col] = float("nan")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("rows", "cols"):
        df[col] = df[col].astype("Int64")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the plain repeats of each point into median, min and max.

    A point counts only if every plain repeat at it finished ``ok``. The
    sweep stops a point on its first breach, so a point with one ``ok``
    repeat followed by a timeout is a breached point, not a slow one.
    Instrumented runs are left out because their own timers slow them down.

    :param df: Loaded measurements.
    :return: One row per surviving point, with ``<value>_median``,
        ``<value>_min`` and ``<value>_max`` for seconds and peak_mb.
    """
    plain = df[~df["instrumented"]]
    keys = ["metric", "axis", "rows", "cols"]
    all_ok = plain.groupby(keys)["status"].transform(lambda s: (s == "ok").all())
    ok = plain[all_ok]
    agg = ok.groupby(keys).agg(
        seconds_median=("seconds", "median"),
        seconds_min=("seconds", "min"),
        seconds_max=("seconds", "max"),
        peak_mb_median=("peak_mb", "median"),
        peak_mb_min=("peak_mb", "min"),
        peak_mb_max=("peak_mb", "max"),
        repeats=("seconds", "size"),
    )
    return agg.reset_index()


def breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Report how much of each instrumented run went into building results.

    :param df: Loaded measurements.
    :return: One row per instrumented ``ok`` run with ``result_share``, the
        fraction of ``assess()`` time spent constructing ``DQResult`` objects.
    """
    timed = df[df["instrumented"] & (df["status"] == "ok")].copy()
    timed["result_share"] = timed["result_seconds"] / timed["seconds"]
    return timed[[
        "metric", "axis", "rows", "cols", "n_results",
        "seconds", "result_seconds", "compute_seconds", "result_share",
    ]]


def growth_exponent(rows_axis: pd.DataFrame) -> tuple[float, float]:
    """
    Fit ``seconds ~ rows ** k`` over the top decade of surviving sizes.

    Every plain repeat enters the fit, so the standard error reflects the
    spread between repeats as well as the curvature of the points.

    :param rows_axis: Plain ``ok`` measurements of one metric on the row axis.
    :return: ``(k, standard_error)``, or ``(nan, nan)`` when the top decade
        holds fewer than ``MIN_FIT_POINTS`` distinct sizes.
    """
    if rows_axis.empty:
        return float("nan"), float("nan")
    top = float(rows_axis["rows"].max())
    window = rows_axis[rows_axis["rows"] >= top / 10 ** FIT_DECADES]
    window = window[window["seconds"] > 0]
    if window["rows"].nunique() < MIN_FIT_POINTS:
        return float("nan"), float("nan")
    fit = stats.linregress(
        np.log10(window["rows"].astype(float)),
        np.log10(window["seconds"].astype(float)),
    )
    return float(fit.slope), float(fit.stderr)


def limits_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the largest input size each metric survived.

    :param df: Loaded measurements.
    :return: One row per metric with its ceiling, what stopped it, the
        median runtime and peak memory at the ceiling, its fitted growth
        exponent, and the share of runtime spent building result objects
        at the ceiling.
    """
    points = aggregate(df)
    shares = breakdown(df)
    plain = df[~df["instrumented"]]
    records = []
    for metric, group in plain.groupby("metric", sort=True):
        rows_axis = group[group["axis"] == "rows"]
        ok_points = points[(points["metric"] == metric) & (points["axis"] == "rows")]
        breached = rows_axis[rows_axis["status"] != "ok"]

        if ok_points.empty:
            max_rows, seconds, peak = 0, float("nan"), float("nan")
        else:
            top = ok_points.loc[ok_points["rows"].idxmax()]
            max_rows = int(top["rows"])
            seconds = float(top["seconds_median"])
            peak = float(top["peak_mb_median"])

        survived = rows_axis[
            (rows_axis["status"] == "ok") & (rows_axis["rows"] <= max_rows)
        ]
        k, k_err = growth_exponent(survived)

        share = shares[
            (shares["metric"] == metric) & (shares["axis"] == "rows")
            & (shares["rows"] == max_rows)
        ]["result_share"]

        records.append({
            "metric": metric,
            "max_rows_ok": max_rows,
            "stopped_by": breached["status"].iloc[0] if not breached.empty else "not reached",
            "median_seconds": seconds,
            "peak_mb": peak,
            "growth_k": k,
            "growth_k_stderr": k_err,
            "result_share": float(share.iloc[0]) if not share.empty else float("nan"),
        })
    # kind="stable" so metrics tied on max_rows_ok keep their alphabetical
    # groupby order rather than whatever quicksort happens to leave them in.
    return pd.DataFrame(records).sort_values(
        "max_rows_ok", ascending=False, kind="stable"
    )


def render_all(df: pd.DataFrame, outdir: pathlib.Path) -> list[pathlib.Path]:
    """
    Render every plot for a sweep.

    Each axis and value gets a shared chart with every metric on one set of
    axes, for comparing metrics against each other, and a panel grid with
    one small chart per metric on its own scales, so every curve's shape is
    visible however fast or slow the metric is and however early it stopped. Both come in log-log and
    in linear scale.

    :param df: Loaded measurements.
    :param outdir: Directory to write PNGs into. Created if missing.
    :return: Paths of the files written.
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    points = aggregate(df)
    looks = _looks(sorted(df["metric"].unique()))
    written: list[pathlib.Path] = []

    for axis, x_col, x_label in (
        ("rows", "rows", "Rows"),
        ("cols", "cols", "Columns"),
    ):
        subset = points[points["axis"] == axis]
        if subset.empty:
            continue
        for value, y_label, stem in (
            ("seconds", "Runtime (s)", "runtime"),
            ("peak_mb", "Peak memory (MB)", "memory"),
        ):
            for scale in SCALES:
                path = outdir / f"{stem}_by_{axis}_{scale}.png"
                _shared_plot(subset, x_col, value, x_label, y_label, scale, looks, path)
                written.append(path)

                path = outdir / f"{stem}_by_{axis}_{scale}_panels.png"
                _panel_plot(subset, x_col, value, x_label, y_label, scale, looks, path)
                written.append(path)

    return written


def _looks(metrics: list[str]) -> dict[str, dict]:
    """Assign each metric a fixed color, line style and marker."""
    return {
        metric: {
            "color": _HUES[i % len(_HUES)],
            "linestyle": _STYLES[(i // len(_HUES)) % len(_STYLES)][0],
            "marker": _STYLES[(i // len(_HUES)) % len(_STYLES)][1],
        }
        for i, metric in enumerate(metrics)
    }


def _draw(ax, group: pd.DataFrame, x_col: str, value: str, look: dict, label: str | None) -> None:
    """Draw one metric's median line with its min-max band."""
    ordered = group.sort_values(x_col)
    x = ordered[x_col].astype(float)
    ax.fill_between(
        x, ordered[f"{value}_min"], ordered[f"{value}_max"],
        color=look["color"], alpha=0.2, linewidth=0,
    )
    ax.plot(
        x, ordered[f"{value}_median"], label=label,
        color=look["color"], linestyle=look["linestyle"],
        marker=look["marker"], markersize=4, linewidth=2,
    )


def _style_axes(ax, scale: str) -> None:
    """Apply the scale and the recessive grid shared by every chart."""
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    if scale == "linear":
        ax.xaxis.set_major_formatter(EngFormatter(sep=""))
        ax.set_ylim(bottom=0)
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)


def _shared_plot(
    df: pd.DataFrame,
    x_col: str,
    value: str,
    x_label: str,
    y_label: str,
    scale: str,
    looks: dict[str, dict],
    path: pathlib.Path,
) -> None:
    """Draw every metric on one set of axes and save it."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for metric, group in df.groupby("metric", sort=True):
        _draw(ax, group, x_col, value, looks[metric], label=metric)

    _style_axes(ax, scale)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} by {x_label.lower()} ({scale}, median and min-max of repeats)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _panel_plot(
    df: pd.DataFrame,
    x_col: str,
    value: str,
    x_label: str,
    y_label: str,
    scale: str,
    looks: dict[str, dict],
    path: pathlib.Path,
) -> None:
    """Draw one small chart per metric, each on its own scales, and save it."""
    metrics = sorted(df["metric"].unique())
    ncols = 4
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False,
    )
    for ax, metric in zip(axes.flat, metrics):
        _draw(ax, df[df["metric"] == metric], x_col, value, looks[metric], label=None)
        _style_axes(ax, scale)
        ax.set_title(metric, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)

    fig.supxlabel(x_label)
    fig.supylabel(y_label)
    fig.suptitle(
        f"{y_label} by {x_label.lower()} per metric "
        f"({scale}, own scales per panel, median and min-max of repeats)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    """Command-line entry point: read a sweep CSV, write plots and a summary."""
    import argparse

    parser = argparse.ArgumentParser(description="Plot a Metis sweep.")
    parser.add_argument("--csv", default="benchmarks/results/sweep.csv")
    parser.add_argument("--outdir", default="benchmarks/results")
    args = parser.parse_args()

    df = load(pathlib.Path(args.csv))
    written = render_all(df, pathlib.Path(args.outdir))
    print(limits_table(df).to_string(index=False))
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
