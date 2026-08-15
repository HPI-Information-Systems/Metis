"""Persistent project footer shown on every page."""
from __future__ import annotations

import streamlit as st

from theme import HPI_RED

PROJECT_URL: str = "https://www.metisdq.org"
REPO_URL: str = "https://github.com/HPI-Information-Systems/Metis"
INSTITUTE: str = "HPI Information Systems"

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
.metis-footer a { color: %(accent)s; text-decoration: none; }
.metis-footer a:hover { text-decoration: underline; }
</style>
""" % {"accent": HPI_RED}


def render() -> None:
    """
    Render the project footer with links to the website and the repository.

    :return: None.
    """
    st.markdown(_FOOTER_CSS, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metis-footer">
            <span><strong>Metis</strong> Data Quality Framework</span>
            <span>{INSTITUTE}</span>
            <a href="{PROJECT_URL}" target="_blank" rel="noopener">metisdq.org</a>
            <a href="{REPO_URL}" target="_blank" rel="noopener">GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
