from MCTS import MCTS, Tree
from MCTS.Nim import Nim


def test_MCTS_constructor():
    MCTS(initial_state=Nim(n=40, k=2), search_games=100)


def test_MCTS_search():
    assert isinstance(
        MCTS(initial_state=Nim(n=40, k=2), search_games=100).search(), MCTS
    )


def test_MCTS_get_root_distribution():
    initial_board = Nim(n=40, k=2)
    possible_next_boards = [Nim(n=39, k=2), Nim(n=38, k=2)]
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

    assert sum([visit_count for SAP, visit_count in dist]) == 100
    assert all(
        [SAP[0] in possible_next_boards and SAP[1] in possible_moves for SAP, _ in dist]
    )


def test_MCTS_update_root():
    assert isinstance(
        MCTS(initial_state=Nim(n=40, k=2), search_games=100).update_root(
            Nim(n=38, k=2)
        ),
        MCTS,
    )
