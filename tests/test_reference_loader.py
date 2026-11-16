import pandas as pd
import pytest

from metis.utils.reference_loader import (
    default_reference,
    describe_reference_source,
    has_reference,
    load_reference_frame,
    load_reference_vocabulary,
)


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "clean.csv"
    p.write_text("city;country\nBerlin;DE\nParis;FR\n", encoding="utf-8")
    return str(p)


def test_none_resolves_to_none():
    assert load_reference_frame(None) is None
    assert load_reference_vocabulary(None) is None


def test_dataframe_passes_through_unchanged():
    df = pd.DataFrame({"a": [1, 2]})
    assert load_reference_frame(df) is df


def test_path_string_loads_with_default_options(tmp_path):
    p = tmp_path / "plain.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    out = load_reference_frame(str(p))
    assert list(out.columns) == ["a", "b"]
    assert len(out) == 1


def test_mapping_honours_the_delimiter(csv_path):
    out = load_reference_frame({"file_name": csv_path, "delimiter": ";"})
    assert list(out.columns) == ["city", "country"]
    assert len(out) == 2


def test_mapping_without_file_name_is_rejected():
    with pytest.raises(ValueError, match="file_name"):
        load_reference_frame({"delimiter": ";"})


def test_list_becomes_a_single_column_frame():
    out = load_reference_frame(["apple", "banana"])
    assert list(out.columns) == ["value"]
    assert out["value"].tolist() == ["apple", "banana"]


def test_unsupported_type_is_rejected():
    with pytest.raises(TypeError, match="reference"):
        load_reference_frame(42)


def test_vocabulary_from_a_set_is_normalised():
    assert load_reference_vocabulary({" Apple ", "BANANA"}) == {"apple", "banana"}


def test_vocabulary_from_a_single_column_frame():
    df = pd.DataFrame({"word": ["Apple", "banana", None]})
    assert load_reference_vocabulary(df) == {"apple", "banana"}


def test_vocabulary_drops_missing_values_from_inline_lists():
    # A config loaded from JSON can legitimately contain null.
    # An inline list with None must drop it, matching DataFrame behavior.
    list_vocab = load_reference_vocabulary(["apple", None, "Banana"])
    df_vocab = load_reference_vocabulary(pd.DataFrame({"w": ["apple", None, "Banana"]}))
    assert list_vocab == df_vocab == {"apple", "banana"}


def test_vocabulary_rejects_a_multi_column_frame():
    df = pd.DataFrame({"a": ["x"], "b": ["y"]})
    with pytest.raises(ValueError, match="single column"):
        load_reference_vocabulary(df)


def test_describe_echoes_serialisable_sources(csv_path):
    assert describe_reference_source(None) is None
    assert describe_reference_source(csv_path) == csv_path
    assert describe_reference_source(["a", "b"]) == ["a", "b"]
    assert sorted(describe_reference_source({"a", "b"})) == ["a", "b"]


def test_describe_summarises_a_live_dataframe():
    out = describe_reference_source(pd.DataFrame({"a": [1, 2]}))
    assert out == {"dataframe": {"rows": 2, "columns": ["a"]}}


def test_none_resolves_to_the_default_reference_while_one_is_active():
    default = pd.DataFrame({"a": [1, 2]})
    with default_reference(default):
        assert load_reference_frame(None) is default
    assert load_reference_frame(None) is None


def test_an_explicit_source_wins_over_the_default_reference():
    own = pd.DataFrame({"a": [1]})
    with default_reference(pd.DataFrame({"a": [9, 9]})):
        assert load_reference_frame(own) is own


def test_a_default_reference_is_never_used_as_a_vocabulary():
    with default_reference(pd.DataFrame({"a": ["x"], "b": ["y"]})):
        assert load_reference_vocabulary(None) is None


def test_has_reference_accepts_none_only_under_a_default():
    assert has_reference(None) is False
    assert has_reference("clean.csv") is True
    with default_reference(pd.DataFrame({"a": [1]})):
        assert has_reference(None) is True


def test_describe_records_the_default_reference_that_was_used():
    with default_reference(pd.DataFrame({"a": [1, 2]})):
        assert describe_reference_source(None) == {
            "dataset_default": {"dataframe": {"rows": 2, "columns": ["a"]}}
        }
