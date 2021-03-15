import pytest
from functools import reduce
from itertools import product
from Game import Board


def test_board_constructor():
    Board(size=4)


def test_equality():
    assert Board() == Board()
    assert Board().perform_action((0, 0)) == Board().perform_action((0, 0))
    assert Board().perform_action((0, 0)) != Board().perform_action((0, 1))


def test_board_get_moves():
    assert len(Board(size=3).get_possible_actions()) == 9
    assert Board(size=3).get_possible_actions() == [
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


def test_board_perform_action():
    with pytest.raises(ValueError):
        Board(size=3).perform_action((0, 0)).perform_action((0, 0))

    assert Board(size=3).perform_action((0, 0)).get_possible_actions() == [
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
    assert Board(size=3).perform_action((0, 0)).perform_action(
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
            lambda board, move: board.perform_action(move),
            product(range(3), range(3)),
            Board(size=3),
        )

    assert Board().is_finished() == (False, None)

    moves = [(2, 0), (2, 2), (2, 1), (3, 1), (1, 2), (1, 3), (0, 3)]
    assert (
        reduce(
            lambda board, move: board.perform_action(move),
            moves,
            Board(size=4),
        ).is_finished()
        == (True, 1)
    )
