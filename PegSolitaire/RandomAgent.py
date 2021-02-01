import random


class RandomAgent:
    """
    An agent that acts randomly.
    """

    def choose_action(self, state, possible_actions, reward=0):
        return random.choice(possible_actions)

    def end_state_reached(self, state, reward):
        pass