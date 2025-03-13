import pytest

from diffusion_lam.utils.attrdict import AttrDict


@pytest.fixture
def sample_dict():
    return AttrDict({"a": 1, "b": 2, "c": {"d": 3, "e": {"f": 4}}})


def test_access_with_dot(sample_dict):
    assert sample_dict.a == 1
    assert sample_dict.b == 2
    assert sample_dict.c.d == 3
    assert sample_dict.c.e.f == 4


def test_access_with_index(sample_dict):
    assert sample_dict["a"] == 1
    assert sample_dict["b"] == 2
    assert sample_dict["c"]["d"] == 3
    assert sample_dict["c"]["e"]["f"] == 4


def test_mixed_access(sample_dict):
    assert sample_dict.c["d"] == 3
    assert sample_dict["c"].e.f == 4


def test_set_value_with_dot(sample_dict):
    sample_dict.a = 1
    sample_dict.c.d = 3
    assert sample_dict.a == 1
    assert sample_dict.c.d == 3
    assert sample_dict["a"] == 1
    assert sample_dict["c"]["d"] == 3


def test_set_value_with_index(sample_dict):
    sample_dict["b"] = 2
    sample_dict["c"]["e"]["f"] = 4
    assert sample_dict.b == 2
    assert sample_dict.c.e.f == 4
    assert sample_dict["b"] == 2
    assert sample_dict["c"]["e"]["f"] == 4


def test_delete_attribute(sample_dict):
    del sample_dict.a
    del sample_dict.c.d
    assert "a" not in sample_dict
    assert "d" not in sample_dict["c"]


def test_delete_attribute_with_index(sample_dict):
    del sample_dict["b"]
    del sample_dict["c"]["e"]
    assert "b" not in sample_dict
    assert "e" not in sample_dict["c"]


def test_missing_key(sample_dict):
    with pytest.raises(AttributeError):
        sample_dict.x
    with pytest.raises(KeyError):
        sample_dict["x"]
