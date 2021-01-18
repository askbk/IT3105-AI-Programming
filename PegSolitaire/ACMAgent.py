import random
from typing import List


class ACMAgent:
    """
    An Actor-Critic model agent.
    """

    def __init__(
        self,
        actor_discount_factor=0.9,
        actor_eligibility_decay_rate=0.9,
        actor_learning_rate=0.01,
        critic_function="table",
        critic_nn_dimensions=None,
        critic_learning_rate=0.9,
        critic_eligibility_decay_rate=0.9,
        critic_discount_factor=0.9,
        initial_epsilon=0.05,
        epsilon_decay_rate=0.9,
    ):
        pass

    def choose_action(self, state, possible_actions: List, reward=0):
        """
        Choose next action to perform.
        """
        return random.choice(possible_actions)