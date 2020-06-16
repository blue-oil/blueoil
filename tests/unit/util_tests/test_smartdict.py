import pytest

from blueoil.utils.smartdict import SmartDict


def test_init():
    d = SmartDict(a=10, b=20, c=30)
    assert isinstance(d, SmartDict)
    assert d.a == 10
    assert d.b == 20
    assert d.c == 30
    assert d["a"] == 10
    assert d["b"] == 20
    assert d["c"] == 30


def test_init_with_dict():
    d = SmartDict({"a": 10, "b": 20, "c": 30})
    assert isinstance(d, SmartDict)
    assert d.a == 10
    assert d.b == 20
    assert d.c == 30
    assert d["a"] == 10
    assert d["b"] == 20
    assert d["c"] == 30


def test_setitem():
    d = SmartDict()
    d["a"] = [
        100,
        {"a": 10, "b": 20, "c": 30},
    ]
    assert d.a[0] == 100
    assert isinstance(d.a[1], SmartDict)
    assert d.a[1].a == 10
    assert d.a[1].b == 20
    assert d.a[1].c == 30

    d["b"] = {"a": 10, "b": 20, "c": 30}
    assert isinstance(d.b, SmartDict)
    assert d.b.a == 10
    assert d.b.b == 20
    assert d.b.c == 30


def test_getattr():
    d = SmartDict(a=10, b=20, c=30)
    assert d.a == 10
    assert d.b == 20
    assert d.c == 30
    with pytest.raises(AttributeError):
        d.d


def test_setattr():
    d = SmartDict()
    d.a = 10
    d.b = 20
    d.c = 30
    assert d["a"] == 10
    assert d["b"] == 20
    assert d["c"] == 30
