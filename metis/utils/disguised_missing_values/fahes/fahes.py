import subprocess
import tempfile
from pathlib import Path
from statistics import mean

import pandas as pd

"""FAHES paper: https://raulcastrofernandez.com/papers/kdd18-fahes.pdf, Code: https://github.com/qcri/FAHES_Code.git, and Demo: https://github.com/qcri/Fahes_Demo.git"""

FAHES_PRECISION = mean([0.384, 0.484, 0.385, 0.371, 0.522])
FAHES_RECALL = mean([0.952, 0.978, 0.87, 0.929, 0.725])
FAHES_F1 = 2 * FAHES_PRECISION * FAHES_RECALL / (FAHES_PRECISION + FAHES_RECALL)


def call_fahes(tab_full_name, output_dir):
    executable = (
        Path(__file__).parent.resolve() / "lib" / "FAHES_Code" / "src" / "FAHES"
    )
    if not executable.exists():
        raise FileNotFoundError(
            f"FAHES executable not found at: {executable}. Please clone https://github.com/qcri/FAHES_Code.git into {executable.parent} and compile it using the makefile at {executable.parent / 'makefile'}."
        )

    cmd = [str(executable), tab_full_name, output_dir, "4"]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FAHES failed: {e.stderr.decode()}")


# Based on https://github.com/qcri/Fahes_Demo.git
def run_fahes(
    data: Path | str | pd.DataFrame, results_path: Path | None = None
) -> pd.DataFrame | None:
    """
    Run FAHES on the given data file and return the resulting DataFrame. The resulting DataFrame contains the disguised missing values identified by FAHES.
    Example resulting DataFrame structure:

    | Table Name | Attribute Name |        DMV        | Frequency | Detecting Tool |
    |------------|----------------|-------------------|-----------|----------------|
    | adult.csv  | workclass      | ?                 |    183    |  Rand          |

    :param data: Path to the input CSV data file or DataFrame containing the data. Warning: if a DataFrame is provided, it will be saved to a temporary CSV file before processing.
    :return: DataFrame with disguised missing values identified by FAHES.
    """
    tmp_file = None

    try:
        if isinstance(data, pd.DataFrame):
            tmp_file = tempfile.NamedTemporaryFile(suffix=".csv")
            data.to_csv(tmp_file.name, index=False)
            data_file_path = Path(tmp_file.name)
        else:
            data_file_path = Path(data)

        if not data_file_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_file_path}")

        with tempfile.TemporaryDirectory() as results_dir:
            call_fahes(str(data_file_path.absolute()), results_dir)
            result_file = Path(results_dir) / ("DMV_" + data_file_path.name)
            if result_file.stat().st_size > 0:
                detected_dmvs = pd.read_csv(result_file, on_bad_lines="warn")
                if results_path is not None:
                    results_path.mkdir(parents=True, exist_ok=True)
                    detected_dmvs.to_csv(
                        results_path / "fahes_detected_dmvs.csv", index=False
                    )
                return detected_dmvs
    finally:
        if tmp_file is not None:
            tmp_file.close()
