"""Persistent project footer shown on every page."""
from __future__ import annotations

import streamlit as st

import assets
from theme import HPI_RED

PROJECT_URL: str = "https://www.metisdq.org"
REPO_URL: str = "https://github.com/HPI-Information-Systems/Metis"
INSTITUTE: str = "HPI Information Systems"
IMPRINT_URL: str = "https://hpi.de/impressum/"

_FOOTER_CSS: str = """
<style>
.metis-footer {
    margin-top: 3rem;
    padding-top: 0.75rem;
    border-top: 1px solid rgba(128, 128, 128, 0.25);
    font-size: 0.82rem;
    opacity: 0.75;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem 1.25rem;
}
.metis-footer > span {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}
.metis-footer a { color: %(accent)s; text-decoration: none; }
.metis-footer a:hover { text-decoration: underline; }
</style>
""" % {"accent": HPI_RED}


def render() -> None:
    """
    Render the project footer with links to the website, the repository
    and the HPI imprint.

    :return: None.
    """
    st.markdown(_FOOTER_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metis-footer">
            <span>{assets.logo_html("1.2em")}<span><strong>Metis</strong> Data Quality Framework</span></span>
            <span>{assets.hpi_logo_html("1.2em")}<span>{INSTITUTE}</span></span>
            <a href="{PROJECT_URL}" target="_blank" rel="noopener">metisdq.org</a>
            <a href="{REPO_URL}" target="_blank" rel="noopener">GitHub</a>
            <a href="{IMPRINT_URL}" target="_blank" rel="noopener">Imprint</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
