import pytest
from PegSolitaire.Game import Board, Position


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

    sizes = range(0, 3)
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
        Board(size=5, hole_positions=[Position((6, 0))])

    with pytest.raises(ValueError):
        Board(size=5, hole_positions=[Position((0, 6))])

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


def test_make_move_exceptions():
    with pytest.raises(ValueError):
        Board().make_move((Position((0, 0)), Position((1, 1))))

    with pytest.raises(ValueError):
        Board().make_move((Position((0, 0)), Position((0, -1))))

    with pytest.raises(ValueError):
        Board(size=5).make_move((Position((6, 0)), Position((8, 0))))

    with pytest.raises(ValueError):
        Board(size=4, shape="triangle").make_move((Position((3, 0)), Position((3, 2))))


def test_is_game_finished():
    assert not Board().is_game_finished()
    assert Board(
        size=4,
        shape="triangle",
        hole_positions=[
            Position((0, 0)),
            Position((0, 1)),
            Position((0, 2)),
            Position((0, 3)),
            Position((1, 0)),
            Position((1, 1)),
            Position((1, 2)),
            Position((2, 0)),
            Position((2, 1)),
        ],
    ).is_game_finished()


def test_get_game_score():
    assert (
        Board(
            size=4,
            shape="triangle",
            hole_positions=[
                Position((0, 0)),
                Position((0, 1)),
                Position((0, 2)),
                Position((0, 3)),
                Position((1, 0)),
                Position((1, 1)),
                Position((1, 2)),
                Position((2, 0)),
                Position((2, 1)),
            ],
        ).get_game_score()
        == 1
    )


def test_get_game_score_exceptions():
    with pytest.raises(RuntimeError):
        Board().get_game_score()


def test_get_edge_list():
    assert sorted(
        Board(
            size=3, shape="triangle", hole_positions=[Position((0, 0))]
        ).get_edge_list()
    ) == sorted(
        [
            (Position((0, 0)), Position((1, 0))),
            (Position((0, 0)), Position((0, 1))),
            (Position((0, 1)), Position((1, 0))),
            (Position((1, 0)), Position((1, 1))),
            (Position((1, 0)), Position((2, 0))),
            (Position((1, 1)), Position((2, 0))),
            (Position((0, 1)), Position((0, 2))),
            (Position((0, 1)), Position((1, 1))),
            (Position((0, 2)), Position((1, 1))),
        ],
    )


def test_get_all_holes():
    assert sorted(Board(size=3, hole_positions=[Position((0, 0))]).get_all_holes()) == [
        Position((0, 0))
    ]


def test_get_all_pieces():
    assert sorted(
        Board(
            size=3, shape="triangle", hole_positions=[Position((1, 1))]
        ).get_all_pieces()
    ) == sorted(
        [
            Position((0, 0)),
            Position((0, 1)),
            Position((0, 2)),
            Position((1, 0)),
            Position((2, 0)),
        ]
    )
