from __future__ import annotations

DEFAULT_DIMENSION_ICON: str = ":material/analytics:"

DIMENSION_ICONS: dict[str, str] = {
    "Completeness": ":material/water_drop:",
    "Consistency":  ":material/link:",
    "Correctness":  ":material/check_circle:",
    "Minimality":   ":material/compress:",
    "Timeliness":   ":material/schedule:",
    "Validity":     ":material/fact_check:",
}


def icon_for(dimension: str) -> str:
    """
    Return the material icon string for a DQ dimension.

    :param dimension: Dimension name (e.g. "Completeness").
    :return: Streamlit material icon spec, or a generic analytics icon for unknown dimensions.
    """
    return DIMENSION_ICONS.get(dimension, DEFAULT_DIMENSION_ICON)
