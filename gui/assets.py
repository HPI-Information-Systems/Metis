"""Static branding assets for the Metis Streamlit GUI.

The logo lives at ``images/logo.png`` and is read from disk once at import.
It is exposed as a base64 data URI because the header renders the logo
inside an ``st.markdown`` block, and a browser cannot fetch a local
filesystem path from an ``<img src>`` tag. Building the URI in memory keeps
the image a real file that can be replaced without touching any code.

If the file is missing the GUI still runs, it just renders without a logo.
"""
from __future__ import annotations

import base64
import io
import os

from PIL import Image

LOGO_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "images",
    "logo.png",
)

LOGO_WIDTH_PX: int = 320
"""Width the logo is downscaled to. The source is far larger than the header
or a browser tab needs, and the encoded image is inlined into every page."""


def _load_logo_data_uri() -> str:
    """
    Read the logo, downscale it, and encode it as a PNG data URI.

    :return: A ``data:image/png;base64,...`` string, or an empty string when
        the file cannot be read.
    """
    try:
        image = Image.open(LOGO_PATH).convert("RGBA")
    except (OSError, ValueError):
        return ""

    height = round(image.height * LOGO_WIDTH_PX / image.width)
    image = image.resize((LOGO_WIDTH_PX, height), Image.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


LOGO_DATA_URI: str = _load_logo_data_uri()


def logo_html(height_px: int = 44) -> str:
    """
    Return an ``<img>`` tag for the Metis logo at a given height.

    :param height_px: Rendered height in pixels.
    :return: An HTML string safe to pass to ``st.markdown``, or an empty
        string when the logo could not be loaded.
    """
    if not LOGO_DATA_URI:
        return ""
    return (
        f'<img src="{LOGO_DATA_URI}" alt="Metis" '
        f'style="height:{height_px}px;vertical-align:middle;" />'
    )
