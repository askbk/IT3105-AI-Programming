from __future__ import annotations


class GameBase:
    def get_possible_actions(self):
        """
        Returns an iterable of all possible actions in the current game state.
        """
        raise NotImplementedError

    def perform_action(self, action) -> GameBase:
        """
        Returns the new game state after performing an action in the current state.
        """
        raise NotImplementedError

    def is_end_state_reached(self) -> bool:
        """
        Returns a boolean indicating whether the game has reached an end state.
        """
        raise NotImplementedError

    def get_tuple_representation(self) -> tuple:
        """
        Returns a tuple representation of the current game state.
        """
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError
