import pytest
from Position import Position


def test_position_constructor():
    Position(0, 0)


def test_position_constructor_exceptions():
    with pytest.raises(ValueError):
        Position(-1, -1)


def test_position_is_on_same_row():
    a = Position(0, 0)
    assert a.is_on_same_row(Position(0, 1))
    assert not a.is_on_same_row(Position(1, 0))