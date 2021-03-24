import random
from Game import GameBase


class Player:
    """
    docstring
    """

    def get_action(self, current_state: GameBase):
        """
        docstring
        """
        return random.choice(current_state.get_possible_actions())

    def next_state(self, next_state: GameBase):
        """
        docstring
        """
        return self

    def end_of_episode_update(self):
        """
        docstring
        """
        return self