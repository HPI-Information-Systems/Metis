import base64
import sys
import pathlib

_GUI = pathlib.Path(__file__).parent.parent / "gui"
if str(_GUI) not in sys.path:
    sys.path.insert(0, str(_GUI))

import assets  # noqa: E402


def test_logo_is_a_png_data_uri():
    assert assets.LOGO_DATA_URI.startswith("data:image/png;base64,")


def test_logo_payload_decodes_to_a_png():
    payload = assets.LOGO_DATA_URI.split(",", 1)[1]
    raw = base64.b64decode(payload)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_logo_stays_small_enough_to_inline():
    assert len(assets.LOGO_DATA_URI) < 200_000


def test_logo_html_embeds_the_uri_and_height():
    html = assets.logo_html(50)
    assert assets.LOGO_DATA_URI in html
    assert "height:50px" in html
