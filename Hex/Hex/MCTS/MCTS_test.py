from Hex.MCTS import MCTS, Tree
from Hex.MCTS.Nim import Nim
from Hex.Game import Board


def test_MCTS_constructor():
    MCTS(initial_state=Nim(n=40, k=2), search_games=100)


def test_MCTS_search():
    assert isinstance(
        MCTS(initial_state=Nim(n=40, k=2), search_games=100).search(), MCTS
    )


def test_MCTS_get_root_distribution():
    initial_board = Nim(n=40, k=2)
    possible_moves = [1, 2]
    assert (
        MCTS(initial_state=initial_board, search_games=100).get_root_distribution()
        == []
    )

    dist = (
        MCTS(initial_state=initial_board, search_games=100)
        .search()
        .get_root_distribution()
    )

    assert sum([visit_count for action, visit_count in dist]) == 100
    assert all([action in possible_moves for action, _ in dist])


def test_MCTS_update_root():
    assert isinstance(
        MCTS(initial_state=Nim(n=40, k=2), search_games=100).update_root(
            Nim(n=38, k=2)
        ),
        MCTS,
    )


def nim_playthrough_winner(n, k, search_games):
    state = Nim(n=n, k=k)
    mcts = MCTS(initial_state=state, search_games=search_games)
    player_turn = 1
    saps = []
    while True:
        mcts = mcts.search()
        best_action = mcts.get_best_action()
        saps.append((state, best_action))
        state = state.perform_action(best_action)
        if state.is_end_state_reached():
            return player_turn
        player_turn = 3 - player_turn
        mcts = mcts.update_root(state)


def test_nim_playthrough_player_1_wins():
    # n != 0 (mod k+1)
    assert nim_playthrough_winner(3, 1, 500) == 1
    assert nim_playthrough_winner(5, 2, 500) == 1
    assert nim_playthrough_winner(5, 3, 500) == 1
    assert nim_playthrough_winner(6, 4, 500) == 1
    assert nim_playthrough_winner(7, 5, 500) == 1
    assert nim_playthrough_winner(11, 6, 1000) == 1
    assert nim_playthrough_winner(17, 7, 1000) == 1


def test_nim_playthrough_player_2_wins():
    # n = 0 (mod k+1)
    assert nim_playthrough_winner(2, 1, 500) == 2
    assert nim_playthrough_winner(6, 2, 500) == 2
    assert nim_playthrough_winner(8, 3, 500) == 2
    assert nim_playthrough_winner(5, 4, 500) == 2
    assert nim_playthrough_winner(12, 5, 1000) == 2
    assert nim_playthrough_winner(14, 6, 1000) == 2
    assert nim_playthrough_winner(24, 7, 1000) == 2


def test_hex_playthrough():
    board_size = 4
    state = Board(size=board_size)
    mcts = MCTS(initial_state=state, search_games=100)
    player_turn = 1
    saps = []
    while True:
        mcts = mcts.search()
        best_action = mcts.get_best_action()
        saps.append((state, best_action))
        state = state.perform_action(best_action)
        if state.is_end_state_reached():
            saps.append((state, None))
            break
        player_turn = 3 - player_turn
        mcts = mcts.update_root(state)

    assert len(saps) >= 2 * board_size - 1
