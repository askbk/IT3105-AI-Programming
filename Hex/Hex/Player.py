from __future__ import annotations
import json
from functools import reduce
from typing import Optional, Tuple, Sequence, Union, Dict
from Hex.Game import GameBase, BoardDisplay
from Hex.Utils import while_loop
from Hex.AgentBase import AgentBase
from Hex.Agent import Agent

EpisodeState = Tuple[Sequence[GameBase], AgentBase]


class Player:
    def __init__(self, game: GameBase, _agent: Optional[AgentBase] = None):
        self._game = game
        self._agent = Agent(initial_state=game) if _agent is None else _agent

    @staticmethod
    def from_config(config: Dict, game: GameBase) -> Player:
        return Player(game=game, _agent=Agent.from_config(config, game))

    @staticmethod
    def from_config_path(path: str, game: GameBase) -> Player:
        with open(path, mode="r") as f:
            config = json.loads(f.read())

        return Player.from_config(config, game)

    def save_agent_nn(self, path: str):
        self._agent._actor._nn.save(path)

    @staticmethod
    def _play_single_episode(game: GameBase, initial_agent: Agent) -> EpisodeState:
        def condition(game_state: EpisodeState) -> bool:
            state_sequence, _ = game_state
            return not state_sequence[-1].is_end_state_reached()

        def step(state: EpisodeState) -> EpisodeState:
            state_sequence, agent = state
            current_state = state_sequence[-1]
            next_state = current_state.perform_action(agent.get_action())
            return [*state_sequence, next_state], agent.next_state(next_state)

        initial = (
            [game],
            initial_agent,
        )

        state_sequence, agent = while_loop(condition, initial, step)

        return state_sequence, agent.end_of_episode_update(game)

    def play_episodes(
        self,
        episode_count: int = 1,
        display_board: bool = True,
        time_interval: float = 1,
        silent=False,
    ):
        def reduce_func(
            agent: AgentBase, episode_number: int
        ) -> Union[Sequence[GameBase], AgentBase]:
            if not silent:
                print(f"Episode {episode_number+1}/{episode_count}")
            board_sequence, updated_agent = Player._play_single_episode(
                self._game, agent
            )
            if episode_number == episode_count - 1:
                self._agent = updated_agent
                return board_sequence

            return updated_agent

        last_episode_board_sequence = reduce(
            reduce_func, range(episode_count), self._agent
        )

        if display_board:
            BoardDisplay.display_board_sequence(
                last_episode_board_sequence, pause=time_interval
            )
