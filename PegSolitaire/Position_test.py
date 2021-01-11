import pytest
from Position import Position


def test_position_constructor():
    Position((0, 0))


def test_position_constructor_exceptions():
    with pytest.raises(ValueError):
        Position((-1, -1))


def test_position_is_on_same_row():
    a = Position((0, 0))
    assert a.is_on_same_row(Position((0, 1)))
    assert not a.is_on_same_row(Position((1, 0)))


def test_position_is_on_same_column():
    a = Position((0, 0))
    assert a.is_on_same_column(Position((1, 0)))
    assert not a.is_on_same_column(Position((0, 1)))


def test_position_is_on_same_diagonal():
    a = Position((1, 1))
    assert a.is_on_same_diagonal(Position((2, 0)))
    assert a.is_on_same_diagonal(Position((0, 2)))
    assert not a.is_on_same_diagonal(Position((1, 2)))


def test_position_straight_distance():
    a = Position((1, 1))
    assert a.straight_distance(Position((2, 0))) == 1
    assert a.straight_distance(Position((1, 0))) == 1
    assert a.straight_distance(Position((1, 5))) == 4
    assert a.straight_distance(Position((5, 1))) == 4


def test_position_straight_distance_exceptions():
    with pytest.raises(ValueError):
        Position((1, 1)).straight_distance(Position((0, 0)))


def test_position_get_middle_position():
    a = Position((0, 0))
    assert a.get_middle_position(Position((2, 0))) == Position((1, 0))

    b = Position((2, 0))
    assert b.get_middle_position(Position((0, 2))) == Position((1, 1))


def test_position_get_middle_position_exceptions():
    with pytest.raises(ValueError):
        Position((0, 0)).get_middle_position(Position((1, 0)))

    with pytest.raises(ValueError):
        Position((0, 0)).get_middle_position(Position((1, 1)))

    with pytest.raises(ValueError):
        Position((0, 0)).get_middle_position(Position((2, 1)))