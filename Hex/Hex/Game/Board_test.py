import pytest
from functools import reduce
from itertools import product
from Hex.Game import Board


def test_board_constructor():
    board = Board(size=4)
    # state_size = board_size + 1 for player turn
    assert board.get_state_size() == 17
    assert board.get_action_space_size() == 16


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
    assert Board(size=3).get_tuple_representation() == (1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert Board(size=3).perform_action((0, 0)).perform_action(
        (0, 1)
    ).get_tuple_representation() == (
        1,
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


def test_from_tuple():
    actions = [(0, 1), (1, 2), (3, 1), (2, 2)]
    board = reduce(
        lambda board, action: board.perform_action(action), actions, Board(size=4)
    )

    state_tuple = board.get_tuple_representation()
    assert Board.from_tuple_representation(state_tuple) == board


def test_win_condition():
    with pytest.raises(Exception):
        reduce(
            lambda board, move: board.perform_action(move),
            product(range(3), range(3)),
            Board(size=3),
        )

    assert Board().is_finished() == (False, None)

    moves = [(0, 2), (2, 2), (1, 2), (1, 3), (2, 1), (3, 1), (3, 0)]
    assert (
        reduce(
            lambda board, move: board.perform_action(move),
            moves,
            Board(size=4),
        ).is_finished()
        == (True, 1)
    )


def test_index_to_action():
    size = 4
    board = Board(size=size)
    actions = [board.index_to_action(index) for index in range(size ** 2)]
    possible_actions = board.get_possible_actions()

    assert len(actions) == len(possible_actions)
    assert all(action in possible_actions for action in actions)


def test_action_to_index():
    size = 4
    board = Board(size=size)
    possible_actions = board.get_possible_actions()
    indeces = [board.action_to_index(action) for action in possible_actions]

    assert len(indeces) == len(possible_actions)
    assert all(index in indeces for index in range(size ** 2))


def test_get_player_turn():
    board = Board()

    assert board.get_player_turn() == 1
    assert board.perform_action(board.get_possible_actions()[0]).get_player_turn() == 2
