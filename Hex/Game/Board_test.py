import pytest
from functools import reduce
from itertools import product
from Game import Board


def test_board_constructor():
    Board(size=4)


def test_board_get_moves():
    assert len(Board(size=3).get_possible_moves()) == 9
    assert Board(size=3).get_possible_moves() == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]


def test_board_make_move():
    with pytest.raises(ValueError):
        Board(size=3).make_move((0, 0)).make_move((0, 0))

    assert Board(size=3).make_move((0, 0)).get_possible_moves() == [
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    ]


def test_vectorized_board():
    assert Board(size=3).get_tuple_representation() == (2, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert Board(size=3).make_move((0, 0)).make_move(
        (0, 1)
    ).get_tuple_representation() == (
        2,
        1,
        2,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )


def test_win_condition():
    with pytest.raises(Exception):
        reduce(
            lambda board, move: board.make_move(move),
            product(range(3), range(3)),
            Board(size=3),
        )

    assert not Board().is_finished()

    moves = [(2, 0), (2, 2), (2, 1), (3, 1), (1, 2), (1, 3), (0, 3)]
    assert reduce(
        lambda board, move: board.make_move(move),
        moves,
        Board(size=4),
    ).is_finished()
