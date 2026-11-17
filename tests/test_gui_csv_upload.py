"""The GUI reads a valid CSV upload and declines everything else with a reason."""
import pathlib
import sys

import pytest

_GUI = pathlib.Path(__file__).parent.parent / "gui"
if str(_GUI) not in sys.path:
    sys.path.insert(0, str(_GUI))

from core.csv_upload import CSV_FORMAT_EXPECTATION, read_csv_upload  # noqa: E402
from metis.utils.dataset_checks import DatasetLoadError, EmptyDatasetError  # noqa: E402


def test_a_valid_file_is_read_without_warnings():
    upload = read_csv_upload(b"name,age\nAda,36\nGrace,45\n", "people.csv")
    assert upload.frame.shape == (2, 2)
    assert list(upload.frame.columns) == ["name", "age"]
    assert upload.warnings == []


def test_quoted_values_may_contain_commas_and_doubled_quotes():
    upload = read_csv_upload(b'a,b\n"x, y","he said ""hi"""\n', "quoted.csv")
    assert upload.frame.iloc[0].tolist() == ["x, y", 'he said "hi"']


@pytest.mark.parametrize("raw", [b"", b"\n\n", b"   \n"], ids=["empty", "blank_lines", "spaces"])
def test_a_file_without_content_is_declined(raw):
    with pytest.raises(EmptyDatasetError, match="is empty"):
        read_csv_upload(raw, "nothing.csv")


def test_a_file_with_only_a_header_is_declined():
    with pytest.raises(EmptyDatasetError, match="column names but no data rows"):
        read_csv_upload(b"name,age\n", "header.csv")


def test_a_binary_file_is_declined():
    with pytest.raises(DatasetLoadError, match="is not a text file"):
        read_csv_upload(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00", "image.csv")


def test_an_unclosed_quote_is_declined():
    with pytest.raises(DatasetLoadError, match="is not a valid CSV file"):
        read_csv_upload(b'a,b\n"1,2\n3,4\n', "broken.csv")


@pytest.mark.parametrize(
    "delimiter", [",", ";", "\t", "|"], ids=["comma", "semicolon", "tab", "pipe"]
)
def test_the_delimiter_is_detected(delimiter):
    raw = delimiter.join(["name", "age"]) + "\n" + delimiter.join(["Ada", "36"]) + "\n"
    upload = read_csv_upload(raw.encode(), "any.csv")
    assert list(upload.frame.columns) == ["name", "age"]
    assert upload.frame.iloc[0].tolist() == ["Ada", 36]
    assert upload.delimiter == delimiter


def test_commas_inside_quoted_values_do_not_hide_a_semicolon_delimiter():
    upload = read_csv_upload(b'name;note\nAda;"x, y, z"\nGrace;"p, q"\n', "notes.csv")
    assert upload.delimiter == ";"
    assert upload.frame["note"].tolist() == ["x, y, z", "p, q"]


def test_a_given_delimiter_overrides_the_detected_one():
    raw = b"name;age\nAda;36\n"
    assert read_csv_upload(raw, "semi.csv").frame.shape == (1, 2)

    forced = read_csv_upload(raw, "semi.csv", delimiter=",")
    assert forced.delimiter == ","
    assert list(forced.frame.columns) == ["name;age"]


def test_a_single_column_file_is_accepted():
    upload = read_csv_upload(b"name\nAda\nGrace\n", "one.csv")
    assert upload.frame.shape == (2, 1)
    assert upload.delimiter == ","


def test_rows_with_too_many_fields_are_skipped_and_reported():
    upload = read_csv_upload(b"a,b\n1,2\n3,4,5\n6,7\n8,9,10,11\n", "ragged.csv")
    assert upload.frame["a"].tolist() == [1, 6]
    assert len(upload.warnings) == 1
    assert "2 row(s)" in upload.warnings[0]
    assert "line 3, 5" in upload.warnings[0]


def test_a_latin1_file_is_read_with_a_warning():
    upload = read_csv_upload("name,city\nJosé,Köln\n".encode("latin-1"), "latin.csv")
    assert upload.frame.iloc[0].tolist() == ["José", "Köln"]
    assert "Latin-1" in upload.warnings[0]


def test_a_utf8_byte_order_mark_does_not_end_up_in_the_column_names():
    upload = read_csv_upload("name,age\nAda,36\n".encode("utf-8-sig"), "bom.csv")
    assert list(upload.frame.columns) == ["name", "age"]
    assert upload.warnings == []


def test_a_utf16_file_is_declined():
    with pytest.raises(DatasetLoadError, match="is not a text file"):
        read_csv_upload("name,age\nAda,36\n".encode("utf-16"), "utf16.csv")


def test_decline_messages_name_the_file():
    with pytest.raises(EmptyDatasetError, match=r"\*\*my_data\.csv\*\*"):
        read_csv_upload(b"", "my_data.csv")


def test_the_stated_expectation_covers_separator_quoting_and_encoding():
    for expected in ("commas", "first row", "double quotes", "UTF-8"):
        assert expected in CSV_FORMAT_EXPECTATION
