import random


class RandomAgent:
    """
    An agent that acts randomly.
    """

    def choose_action(self, action_list):
        return random.choice(action_list)