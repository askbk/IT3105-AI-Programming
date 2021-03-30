from typing import Optional
from Hex.Game import GameBase, BoardDisplay
from Hex.Utils import while_loop
from Hex.AgentBase import AgentBase
from Hex.Agent import Agent


class Player:
    def __init__(self, game: GameBase, _agent: Optional[AgentBase] = None):
        self._game = game
        self._agent = (
            Agent(
                initial_state=game,
                state_size=game.get_state_size(),
                action_space_size=game.get_action_space_size(),
            )
            if _agent is None
            else _agent
        )

    def _play_single_episode(self):
        def condition(game_state):
            state_sequence, _ = game_state
            return not state_sequence[-1].is_end_state_reached()

        def step(state):
            state_sequence, agent = state
            current_state = state_sequence[-1]
            next_state = current_state.perform_action(agent.get_action(current_state))
            return [*state_sequence, next_state], agent.next_state(next_state)

        initial = (
            [self._game],
            self._agent,
        )

        state_sequence, agent = while_loop(condition, initial, step)

        return state_sequence, agent.end_of_episode_update()

    def play_single_episode(self, display_board: bool = True, time_interval: int = 1):
        boards, _ = self._play_single_episode()
        if display_board:
            BoardDisplay.display_board_sequence(boards, pause=time_interval)

    def play_episodes(self, episode_count: int = 1):
        self._play_single_episode()