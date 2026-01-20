import ctypes
import os
import tempfile
from pathlib import Path
from statistics import mean

import pandas as pd

FAHES_PRECISION = mean([0.384, 0.484, 0.385, 0.371, 0.522])
FAHES_RECALL = mean([0.952, 0.978, 0.87, 0.929, 0.725])
FAHES_F1 = 2 * FAHES_PRECISION * FAHES_RECALL / (FAHES_PRECISION + FAHES_RECALL)


def call_fahes(tab_full_name, output_dir):
    path = Path(__file__).parent.resolve() / "lib" / "FAHES_Code" / "libFahes.so"
    if not path.exists():
        raise FileNotFoundError(
            f"Fahes shared library not found at: {path}. Please clone https://github.com/qcri/FAHES_Code.git into {path.parent} and compile it using the provided makefile at {path.parent.parent / 'makefile'}."
        )

    LP_c_char = ctypes.POINTER(ctypes.c_char)
    LP_LP_c_char = ctypes.POINTER(LP_c_char)
    try:
        Fahes = ctypes.CDLL(str(path), use_errno=True)
    except OSError as e:
        raise ImportError(f"Failed to load Fahes shared library: {path}") from e

    try:
        Fahes.main.argtypes = (ctypes.c_int, LP_LP_c_char)
    except AttributeError as e:
        raise AttributeError(
            "Fahes library missing 'main' or has unexpected signature"
        ) from e

    ctypes.set_errno(0)
    args = [str(path), tab_full_name, output_dir, "4"]
    argc = len(args)
    argv = (LP_c_char * (argc + 1))()
    for i, arg in enumerate(args):
        enc_arg = arg.encode("utf-8")
        argv[i] = ctypes.create_string_buffer(enc_arg)

    rc = Fahes.main(argc, argv)
    if rc != 0:
        err = ctypes.get_errno()
        err_msg = os.strerror(err) if err else "Unknown C error"
        raise RuntimeError(f"Fahes.main failed (rc={rc}, errno={err}: {err_msg})")


# Based on https://github.com/qcri/Fahes_Demo.git
def run_fahes(data: Path | str | pd.DataFrame) -> pd.DataFrame:
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

            return pd.read_csv(result_file)
    finally:
        if tmp_file is not None:
            tmp_file.close()
