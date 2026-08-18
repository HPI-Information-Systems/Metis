"""Upload and configure a FLAPS MUP file for coverage-gap assessment."""

from __future__ import annotations

import csv
import io
import re

import streamlit as st

_MINCOV_RE = re.compile(r"mincov[_=-]?(\d+)", re.IGNORECASE)
_IDENTIFIER_NAMES = {"id", "row_id", "rowid", "index"}


def render(config_class, key_prefix: str, df_columns: list[str]):
    """Render the MUP upload and positional attribute mapping controls."""
    st.caption(
        "Upload the MUP output for this dataset, then select the dataset "
        "attributes used during MUP discovery. Attributes are interpreted in "
        "dataset-column order. The final field of each row is read as that "
        "MUP's actual coverage."
    )

    delimiter = st.text_input(
        "MUP delimiter",
        value=",",
        max_chars=1,
        key=f"{key_prefix}__delimiter",
    )
    wildcard = st.text_input(
        "Wildcard token",
        value="x",
        key=f"{key_prefix}__wildcard",
        help="Token used by FLAPS for an unspecified attribute.",
    )
    uploaded = st.file_uploader(
        "MUP file",
        type=["txt", "csv"],
        key=f"{key_prefix}__mups_upload",
        help="FLAPS rows contain the positional pattern followed by its actual coverage.",
    )

    if uploaded is None:
        st.caption("Upload a MUP file to complete this metric configuration.")
        return None

    raw = uploaded.getvalue()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")

    if len(delimiter) != 1:
        st.error("The MUP delimiter must be exactly one character.")
        return None

    first_fields = _first_mup_fields(content, delimiter)
    file_id = f"{uploaded.name}::{uploaded.size}"
    file_state_key = f"{key_prefix}__mups_file_id"
    attributes_key = f"{key_prefix}__attributes"
    mincov_key = f"{key_prefix}__mincov"
    if st.session_state.get(file_state_key) != file_id:
        st.session_state[file_state_key] = file_id
        st.session_state[attributes_key] = _default_attributes(
            df_columns, len(first_fields)
        )
        inferred_mincov = _infer_mincov(uploaded.name)
        st.session_state[mincov_key] = (
            str(inferred_mincov) if inferred_mincov is not None else ""
        )

    selected = st.multiselect(
        "Diversity attributes",
        options=df_columns,
        key=attributes_key,
        help=(
            "Select exactly the attributes used to discover these MUPs. Their "
            "dataset order must match the positional fields in the MUP file."
        ),
    )
    attributes = [column for column in df_columns if column in set(selected)]

    mincov_raw = st.text_input(
        "Minimum coverage threshold (mincov, optional)",
        key=mincov_key,
        placeholder="e.g. 19000",
        help=(
            "Checks that every MUP's final coverage value is below the "
            "threshold. The supplied MUP frontier determines the DNF count."
        ),
    )

    if not attributes:
        st.caption("Select at least one diversity attribute.")
        return None
    if first_fields and len(first_fields) < len(attributes) + 1:
        st.error(
            f"The first MUP row has {len(first_fields)} fields, but "
            f"{len(attributes)} pattern fields plus one final coverage field "
            "are required."
        )
        return None

    intermediate = len(first_fields) - len(attributes) - 1 if first_fields else 0
    mapping = ", ".join(
        f"{index + 1}: `{attribute}`" for index, attribute in enumerate(attributes)
    )
    st.caption(f"Positional mapping — {mapping}")
    if not first_fields:
        st.warning(
            "The uploaded file contains no MUP rows. If this is intentional, "
            "the coverage gap is 0 and the coverage-space score is 1."
        )
    else:
        st.caption(
            "The final field of every MUP row is parsed as its actual coverage; "
            "it validates the frontier but is not a DNF dimension."
        )
    if intermediate > 0:
        st.caption(
            f"The {intermediate} field{'s' if intermediate != 1 else ''} between "
            "the pattern and final coverage will be ignored as metadata."
        )

    mincov = None
    if mincov_raw.strip():
        try:
            mincov = int(mincov_raw)
        except ValueError:
            st.error("mincov must be a positive integer.")
            return None

    try:
        config = config_class(
            mups_content=content,
            mups_filename=uploaded.name,
            attributes=attributes,
            mincov=mincov,
            wildcard=wildcard,
            delimiter=delimiter,
        )
        config.validate()
        return config
    except (TypeError, ValueError) as exc:
        st.error(f"Config error: {exc}")
        return None


def _first_mup_fields(content: str, delimiter: str) -> list[str]:
    for fields in csv.reader(io.StringIO(content), delimiter=delimiter):
        if not fields or all(not field.strip() for field in fields):
            continue
        if fields[0].lstrip().startswith("#"):
            continue
        return fields
    return []


def _default_attributes(df_columns: list[str], mup_field_count: int) -> list[str]:
    non_identifiers = [
        column for column in df_columns if column.lower() not in _IDENTIFIER_NAMES
    ]
    # FLAPS output commonly appends one coverage value after the pattern.
    likely_pattern_width = max(1, mup_field_count - 1)
    if len(non_identifiers) == likely_pattern_width:
        return non_identifiers
    if len(df_columns) == likely_pattern_width:
        return list(df_columns)
    return non_identifiers or list(df_columns)


def _infer_mincov(filename: str) -> int | None:
    match = _MINCOV_RE.search(filename)
    return int(match.group(1)) if match else None
