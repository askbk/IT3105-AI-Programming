import pytest
from Board import Board
from Position import Position


def test_board_constructor():
    Board()
    Board(
        size=5,
        shape="triangle",
        hole_positions=[Position((1, 2)), Position((1, 3))],
    )


def test_board_constructor_exceptions():
    with pytest.raises(ValueError):
        Board(hole_positions=[])

    sizes = range(0, 4)
    for size in sizes:
        with pytest.raises(ValueError):
            Board(size=size)

    shapes = ["", "square"]
    for shape in shapes:
        with pytest.raises(ValueError):
            Board(shape=shape)

    with pytest.raises(ValueError):
        Board(shape="triangle", size=5, hole_positions=[Position((4, 2))])

    with pytest.raises(ValueError):
        Board(
            shape="triangle",
            size=5,
            hole_positions=[Position((1, 1)), Position((1, 1))],
        )


def test_get_possible_moves():
    assert sorted(
        Board(
            size=4, shape="diamond", hole_positions=[Position((1, 1))]
        ).get_possible_moves()
    ) == sorted(
        [(Position((1, 3)), Position((1, 1))), (Position((3, 1)), Position((1, 1)))]
    )

    assert sorted(
        Board(
            size=4,
            shape="triangle",
            hole_positions=[Position((1, 2)), Position((2, 1))],
        ).get_possible_moves()
    ) == sorted(
        [(Position((0, 1)), Position((2, 1))), (Position((1, 0)), Position((1, 2)))]
    )


def test_make_move():
    assert sorted(
        Board(size=4, shape="diamond", hole_positions=[Position((1, 1))])
        .make_move((Position((1, 3)), Position((1, 1))))
        .get_possible_moves()
    ) == sorted(
        [
            (Position((1, 0)), Position((1, 2))),
            (Position((3, 0)), Position((1, 2))),
            (Position((3, 1)), Position((1, 3))),
            (Position((3, 2)), Position((1, 2))),
            (Position((3, 3)), Position((1, 3))),
        ]
    )