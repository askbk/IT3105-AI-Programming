from __future__ import annotations
from Hex.Game import GameBase
from Hex.Types import Action


class AgentBase:
    def get_action(self) -> Action:
        """
        Returns next action to perform.
        """
        raise NotImplementedError

    def next_state(self, next_state: GameBase) -> AgentBase:
        """
        Move Agent to the next game state.
        """
        raise NotImplementedError

    def end_of_episode_update(self, initial_state: GameBase) -> AgentBase:
        """
        Perform any updates necessary at the end of the episode.
        """
        raise NotImplementedError

    def get_name(self):
        """
        Returns agent's name.
        """
        raise NotImplementedError
