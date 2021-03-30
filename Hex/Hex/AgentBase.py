from __future__ import annotations
from Hex.Game import GameBase
from Hex.Types import Action


class AgentBase:
    def get_next_action(self) -> Action:
        """
        Returns next action to perform.
        """
        raise NotImplementedError

    def next_state(self, next_state: GameBase) -> AgentBase:
        """
        Move Agent to the next game state.
        """
        raise NotImplementedError

    def end_of_episode_update(self) -> AgentBase:
        """
        Perform any updates necessary at the end of the episode
        """
        raise NotImplementedError