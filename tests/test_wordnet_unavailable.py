"""The readability metrics refuse to run without WordNet instead of scoring 0."""
import pandas as pd
import pytest

from metis.metric.readability.readability_wordnet import readability_wordnet
from metis.metric.readability.readability_wordnet_config import (
    readability_wordnet_config,
)
from metis.utils.readability import scorers
from metis.utils.readability.scorers import WordNetUnavailableError


class _MissingCorpus:
    """Stands in for NLTK's lazy WordNet loader when the corpus is absent."""

    def synsets(self, _token):
        raise LookupError("Resource wordnet not found.")


@pytest.fixture
def data():
    return pd.DataFrame({"city_name": ["Berlin", "Paris"]})


def test_a_missing_corpus_raises_with_the_install_command(monkeypatch, data):
    monkeypatch.setattr(scorers, "wn", _MissingCorpus())
    with pytest.raises(WordNetUnavailableError, match="python -m nltk.downloader wordnet"):
        readability_wordnet().assess(data, metric_config=readability_wordnet_config())


def test_a_missing_nltk_package_raises_with_the_install_command(monkeypatch, data):
    monkeypatch.setattr(scorers, "wn", None)
    with pytest.raises(WordNetUnavailableError, match="pip install -r requirements.txt"):
        readability_wordnet().assess(data, metric_config=readability_wordnet_config())


def test_with_wordnet_installed_the_metric_runs(data):
    pytest.importorskip("nltk")
    try:
        scorers.require_wordnet()
    except WordNetUnavailableError:
        pytest.skip("WordNet corpus is not installed")
    assert readability_wordnet().assess(data, metric_config=readability_wordnet_config())
