"""Turn a sweep CSV into limit tables and log-log plots."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_NUMERIC = ("rows", "cols", "seconds", "peak_mb", "n_results")


def load(path: pathlib.Path) -> pd.DataFrame:
    """
    Read a sweep CSV with correct dtypes.

    :param path: Path to the CSV written by ``sweep.write_csv``.
    :return: The measurements as a DataFrame.
    """
    df = pd.read_csv(path)
    for col in _NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("rows", "cols"):
        df[col] = df[col].astype("Int64")
    return df


def limits_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise the largest input size each metric survived.

    :param df: Loaded measurements.
    :return: One row per metric with its ceiling and what stopped it.
    """
    records = []
    for metric, group in df.groupby("metric", sort=True):
        rows_axis = group[group["axis"] == "rows"]
        ok = rows_axis[rows_axis["status"] == "ok"]
        breached = rows_axis[rows_axis["status"] != "ok"]
        records.append({
            "metric": metric,
            "max_rows_ok": int(ok["rows"].max()) if not ok.empty else 0,
            "stopped_by": breached["status"].iloc[0] if not breached.empty else "not reached",
            "slowest_ok_seconds": float(ok["seconds"].max()) if not ok.empty else float("nan"),
            "peak_mb": float(ok["peak_mb"].max()) if not ok.empty else float("nan"),
        })
    # kind="stable" so metrics tied on max_rows_ok keep their alphabetical
    # groupby order rather than whatever quicksort happens to leave them in.
    return pd.DataFrame(records).sort_values(
        "max_rows_ok", ascending=False, kind="stable"
    )


def render_all(df: pd.DataFrame, outdir: pathlib.Path) -> list[pathlib.Path]:
    """
    Render every plot for a sweep.

    :param df: Loaded measurements.
    :param outdir: Directory to write PNGs into. Created if missing.
    :return: Paths of the files written.
    """
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []

    for axis, x_col, x_label in (
        ("rows", "rows", "Rows"),
        ("cols", "cols", "Columns"),
    ):
        subset = df[(df["axis"] == axis) & (df["status"] == "ok")]
        if subset.empty:
            continue
        for value_col, y_label, stem in (
            ("seconds", "Runtime (s)", "runtime"),
            ("peak_mb", "Peak memory (MB)", "memory"),
        ):
            path = outdir / f"{stem}_by_{axis}.png"
            _line_plot(subset, x_col, value_col, x_label, y_label, path)
            written.append(path)

    return written


def _line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    path: pathlib.Path,
) -> None:
    """Draw one log-log line per metric and save it."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for metric, group in df.groupby("metric", sort=True):
        ordered = group.sort_values(x_col)
        ax.plot(ordered[x_col], ordered[y_col], marker="o", label=metric)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} by {x_label.lower()}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))
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
