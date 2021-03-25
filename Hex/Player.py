from Game import GameBase
from Utils import while_loop
from AgentBase import AgentBase
from Agent import Agent


class Player:
    def __init__(self, _agent: AgentBase = None):
        self._agent = Agent(initial_state=None) if _agent is None else _agent

    def _play_single_episode(self, initial_game_board: GameBase):
        def condition(game_state):
            state_sequence, _ = game_state
            return not state_sequence[-1].is_end_state_reached()

        def step(state):
            state_sequence, agent = state
            current_state = state_sequence[-1]
            next_state = current_state.perform_action(agent.get_action(current_state))
            return [*state_sequence, next_state], agent.next_state(
                next_state, initial=False
            )

        initial = (
            [initial_game_board],
            self._agent.next_state(initial_game_board, initial=True),
        )

        state_sequence, agent = while_loop(condition, initial, step)

        return state_sequence, agent.end_of_episode_update()

    def play_episodes(self, initial_game_board: GameBase, episode_count=1):
        self._play_single_episode(initial_game_board)