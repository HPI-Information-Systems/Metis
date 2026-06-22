"""Shared Altair color scale for DQ scores (0 = red, 0.5 = yellow, 1 = green)."""
from __future__ import annotations

import altair as alt

DQ_COLOR_SCALE: alt.Scale = alt.Scale(scheme="redyellowgreen", domain=[0, 1])


def dq_color(field: str = "DQvalue:Q", title: str = "DQ Score") -> alt.Color:
    """
    Build an Altair ``Color`` channel that uses the shared DQ score scale.

    :param field: Altair field expression for the encoded value.
    :param title: Axis/legend title.
    :return: An ``alt.Color`` channel.
    """
    return alt.Color(field, scale=DQ_COLOR_SCALE, title=title)
