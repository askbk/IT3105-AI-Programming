import pytest
from Board import Board


def test_board_constructor():
    Board()
    Board(hole_count=2, size=4, shape="triangle")
    Board(hole_count=3, size=5, shape="diamond")


def test_board_constructor_exceptions():
    with pytest.raises(ValueError):
        Board(hole_count=0)

    sizes = range(0, 4)
    for size in sizes:
        with pytest.raises(ValueError):
            Board(size=size)

    shapes = ["", "square"]
    for shape in shapes:
        with pytest.raises(ValueError):
            Board(shape=shape)