from Hex.Game import Board
from Hex.MCTS import MCTS


def benchmark_mcts_hex_playthrough():
    board_size = 5
    state = Board(size=board_size)
    mcts = MCTS(initial_state=state, search_games=2000, time_limit=1)
    saps = []
    while True:
        mcts = mcts.search(lambda state: state.get_possible_actions()[0])
        best_action = mcts.get_best_action()
        saps.append((state, best_action))
        state = state.perform_action(best_action)
        if state.is_end_state_reached():
            saps.append((state, None))
            break
        mcts = mcts.update_root(state)


if __name__ == "__main__":
    benchmark_mcts_hex_playthrough()
