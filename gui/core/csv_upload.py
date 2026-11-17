"""Read an uploaded CSV file, declining files that cannot be assessed."""
from __future__ import annotations

import csv
import io
import re
import warnings
from dataclasses import dataclass, field

import pandas as pd

from metis.utils.dataset_checks import DatasetLoadError, EmptyDatasetError

CSV_FORMAT_EXPECTATION: str = (
    "**Expected format**\n"
    "- Values are separated by commas (`,`). Semicolons, tabs and pipes are "
    "detected automatically.\n"
    "- The first row holds the column names.\n"
    "- A value that contains the separator, a double quote or a line break is "
    'enclosed in double quotes (`"`). A double quote inside such a value is '
    'written twice (`""`).\n'
    "- Decimal numbers use a point (`3.14`).\n"
    "- The file is UTF-8 encoded."
)

_SKIPPED_LINE_RE: re.Pattern[str] = re.compile(r"Skipping line (\d+)")
# Control bytes text files lack (tab, line breaks, form feed, DOS EOF excepted).
_BINARY_BYTE_RE: re.Pattern[bytes] = re.compile(rb"[\x00-\x08\x0e-\x19\x1b-\x1f]")
DELIMITER_NAMES: dict[str, str] = {
    ",": "commas", ";": "semicolons", "\t": "tabs", "|": "pipes",
}
_DEFAULT_DELIMITER: str = ","
_SNIFF_BYTES: int = 64 * 1024
_MAX_SKIPPED_LINES_LISTED: int = 5


@dataclass
class CsvUpload:
    """A successfully read upload.

    :param frame: The parsed data.
    :param warnings: Things the user should know about how the file was read,
        such as skipped rows. Empty when the file was read as is.
    :param delimiter: The delimiter the file was read with.
    """

    frame: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    delimiter: str = _DEFAULT_DELIMITER


def detect_delimiter(raw: bytes) -> str:
    """
    Guess the delimiter of CSV content from its first lines.

    :param raw: The content of the file.
    :return: One of the delimiters in ``DELIMITER_NAMES``. A comma when the
        content does not point to any of them, as in a single-column file.
    """
    sample = raw[:_SNIFF_BYTES].decode("utf-8", errors="replace")
    if len(raw) > _SNIFF_BYTES:
        sample = sample.rsplit("\n", 1)[0]
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(DELIMITER_NAMES)).delimiter
    except csv.Error:
        return _DEFAULT_DELIMITER


def read_csv_upload(
    raw: bytes, file_name: str, delimiter: str | None = None
) -> CsvUpload:
    """
    Parse the bytes of an uploaded CSV file.

    The file is read as UTF-8, with a Latin-1 fallback. Rows with too many
    fields are skipped and reported in the returned warnings.

    :param raw: The content of the uploaded file.
    :param file_name: The name of the uploaded file, used in messages.
    :param delimiter: The delimiter to read the file with. ``None`` detects
        it from the content.
    :raises EmptyDatasetError: If the file has no content, no columns or no
        data rows.
    :raises DatasetLoadError: If the file is not text or not a valid CSV file.
    :return: The parsed data together with any warnings.
    """
    if not raw.strip():
        raise EmptyDatasetError(f"**{file_name}** is empty. There is nothing to assess.")

    notes: list[str] = []
    if _BINARY_BYTE_RE.search(raw):
        raise DatasetLoadError(
            f"**{file_name}** is not a text file, so it cannot be read as CSV. "
            "Export the data as a CSV file and upload that."
        )

    delimiter = delimiter or detect_delimiter(raw)
    try:
        frame, skipped = _parse(raw, "utf-8", delimiter)
    except UnicodeDecodeError:
        frame, skipped = _parse_or_decline(raw, "latin-1", delimiter, file_name)
        notes.append(
            "The file is not UTF-8 encoded and was read as Latin-1. Check "
            "that special characters look right in the preview."
        )
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise _declined(e, file_name) from e

    if len(frame.columns) == 0:
        raise EmptyDatasetError(f"**{file_name}** has no columns. There is nothing to assess.")

    if len(frame) == 0:
        raise EmptyDatasetError(
            f"**{file_name}** has column names but no data rows. "
            "There is nothing to assess."
        )

    if skipped:
        notes.append(_skipped_note(skipped))

    return CsvUpload(frame=frame, warnings=notes, delimiter=delimiter)


def _parse(
    raw: bytes, encoding: str, delimiter: str
) -> tuple[pd.DataFrame, list[int]]:
    """
    Parse CSV bytes and collect the line numbers pandas had to skip.

    :param raw: The content of the file.
    :param encoding: The encoding to decode the content with.
    :param delimiter: The delimiter that separates the values.
    :return: The parsed frame and the 1-based numbers of the skipped lines.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", pd.errors.ParserWarning)
        frame = pd.read_csv(
            io.BytesIO(raw), sep=delimiter, encoding=encoding,
            on_bad_lines="warn", low_memory=False,
        )
    skipped = [
        int(number)
        for warning in caught
        if issubclass(warning.category, pd.errors.ParserWarning)
        for number in _SKIPPED_LINE_RE.findall(str(warning.message))
    ]
    return frame, skipped


def _parse_or_decline(
    raw: bytes, encoding: str, delimiter: str, file_name: str
) -> tuple[pd.DataFrame, list[int]]:
    """
    Parse CSV bytes, turning a parse failure into a decline message.

    :param raw: The content of the file.
    :param encoding: The encoding to decode the content with.
    :param delimiter: The delimiter that separates the values.
    :param file_name: The name of the uploaded file, used in messages.
    :raises DatasetError: If the content cannot be parsed.
    :return: The parsed frame and the 1-based numbers of the skipped lines.
    """
    try:
        return _parse(raw, encoding, delimiter)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise _declined(e, file_name) from e


def _declined(error: Exception, file_name: str) -> Exception:
    """
    Translate a pandas parse error into the message shown to the user.

    :param error: The error pandas raised.
    :param file_name: The name of the uploaded file.
    :return: The error to raise instead.
    """
    if isinstance(error, pd.errors.EmptyDataError):
        return EmptyDatasetError(
            f"**{file_name}** contains no data. There is nothing to assess."
        )
    return DatasetLoadError(
        f"**{file_name}** is not a valid CSV file and could not be read "
        f"({error}). Check that quoted values are closed and that the file "
        "matches the expected format."
    )


def _skipped_note(skipped: list[int]) -> str:
    """
    Describe the rows that were left out of the dataset.

    :param skipped: The 1-based line numbers of the skipped rows.
    :return: A warning naming how many rows were skipped and where.
    """
    listed = ", ".join(str(n) for n in skipped[:_MAX_SKIPPED_LINES_LISTED])
    if len(skipped) > _MAX_SKIPPED_LINES_LISTED:
        listed += ", ..."
    return (
        f"{len(skipped):,} row(s) have more fields than the header and were "
        f"skipped (line {listed}). They are not part of the assessment. This "
        "usually means a value contains the separator and is not enclosed "
        "in double quotes."
    )
