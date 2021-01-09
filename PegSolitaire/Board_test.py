import pytest
from Board import Board


def test_board_constructor_exceptions():
    with pytest.raises(ValueError):
        Board(hole_count=0)

    sizes = range(0, 4)
    for size in sizes:
        with pytest.raises(ValueError):
            Board(size=size)