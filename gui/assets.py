"""Static branding assets for the Metis Streamlit GUI.

The logos live at ``images/logo_metis.png`` and ``images/logo_hpi.png`` and
are read from disk once at import. They are exposed as base64 data URIs
because the header and footer render them inside ``st.markdown`` blocks, and
a browser cannot fetch a local filesystem path from an ``<img src>`` tag.
Building the URIs in memory keeps the images real files that can be replaced
without touching any code.

If a file is missing the GUI still runs, it just renders without that logo.
"""
from __future__ import annotations

import base64
import io
import os

from PIL import Image

IMAGES_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "images",
)

LOGO_PATH: str = os.path.join(IMAGES_DIR, "logo_metis.png")
HPI_LOGO_PATH: str = os.path.join(IMAGES_DIR, "logo_hpi.png")

LOGO_WIDTH_PX: int = 320
"""Width the logos are downscaled to. The sources are far larger than the
header or a browser tab needs, and the encoded images are inlined into every
page."""


def _load_data_uri(path: str) -> str:
    """
    Read an image, downscale it, and encode it as a PNG data URI.

    :param path: Path to the image file.
    :return: A ``data:image/png;base64,...`` string, or an empty string when
        the file cannot be read.
    """
    try:
        image = Image.open(path).convert("RGBA")
    except (OSError, ValueError):
        return ""

    height = round(image.height * LOGO_WIDTH_PX / image.width)
    image = image.resize((LOGO_WIDTH_PX, height), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


LOGO_DATA_URI: str = _load_data_uri(LOGO_PATH)
HPI_LOGO_DATA_URI: str = _load_data_uri(HPI_LOGO_PATH)


def _img_html(data_uri: str, alt: str, height: int | str) -> str:
    """
    Return an ``<img>`` tag for an inlined logo.

    :param data_uri: The logo's data URI.
    :param alt: Alternative text.
    :param height: Rendered height, in pixels when an int, otherwise any CSS
        length such as ``"1em"``.
    :return: An HTML string safe to pass to ``st.markdown``, or an empty
        string when the logo could not be loaded.
    """
    if not data_uri:
        return ""
    css_height = f"{height}px" if isinstance(height, int) else height
    return (
        f'<img src="{data_uri}" alt="{alt}" '
        f'style="height:{css_height};vertical-align:middle;" />'
    )


def logo_html(height: int | str = 44) -> str:
    """
    Return an ``<img>`` tag for the Metis logo at a given height.

    :param height: Rendered height, in pixels when an int, otherwise any CSS
        length.
    :return: An HTML string, or an empty string when the logo is missing.
    """
    return _img_html(LOGO_DATA_URI, "Metis", height)


def hpi_logo_html(height: int | str = "1em") -> str:
    """
    Return an ``<img>`` tag for the HPI logo at a given height.

    :param height: Rendered height, in pixels when an int, otherwise any CSS
        length.
    :return: An HTML string, or an empty string when the logo is missing.
    """
    return _img_html(HPI_LOGO_DATA_URI, "HPI", height)
