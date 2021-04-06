from __future__ import annotations
from typing import Tuple, Optional
from collections.abc import Sequence
import Hex.Types as Types


class GameBase:
    def get_possible_actions(self) -> Sequence[Types.Action]:
        """
        Returns an iterable of all possible actions in the current game state.
        """
        raise NotImplementedError

    def perform_action(self, action: Types.Action) -> GameBase:
        """
        Returns the new game state after performing an action in the current state.
        """
        raise NotImplementedError

    def is_end_state_reached(self) -> bool:
        """
        Returns a boolean indicating whether the game has reached an end state.
        """
        raise NotImplementedError

    def is_finished(self) -> Tuple[bool, Optional[int]]:
        """
        Returns a (bool, int) tuple indicating whether an end state is reached and the winner.
        """
        raise NotImplementedError

    def get_tuple_representation(self) -> tuple:
        """
        Returns a tuple representation of the current game state.
        """
        raise NotImplementedError

    def get_state_size(self) -> int:
        """
        Returns the size of the state representation
        """
        raise NotImplementedError

    def get_action_space_size(self) -> int:
        """
        Returns the size of the action space
        """
        raise NotImplementedError

    def index_to_action(self, index: int) -> Types.Action:
        """
        Returns the action corresponding to the given index.
        """
        raise NotImplementedError

    def action_to_index(self, action: Types.Action) -> int:
        """
        Returns the index corresponding to the given action.
        """
        raise NotImplementedError

    def __eq__(self, other: GameBase) -> bool:
        raise NotImplementedError
