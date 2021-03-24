from Game.Board import Board
from Utils import while_loop
from Player import Player


def run_single_episode(initial_board, player):
    def condition(game_state):
        state_sequence, _ = game_state
        return not state_sequence[-1].is_end_state_reached()

    def step(state):
        state_sequence, player = state
        current_state = state_sequence[-1]
        next_state = current_state.perform_action(player.get_action(current_state))
        return [*state_sequence, next_state], player.next_state(next_state)

    initial = ([initial_board], player)

    state_sequence, player = while_loop(condition, initial, step)

    return state_sequence, player.end_of_episode_update()


if __name__ == "__main__":
    run_single_episode(Board(), Player())