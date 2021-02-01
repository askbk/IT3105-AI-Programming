import random
from typing import List
from Actor import Actor
from Critic import Critic


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
        self._actor = Actor(
            discount_factor=actor_discount_factor,
            eligibility_decay_rate=actor_eligibility_decay_rate,
            learning_rate=actor_learning_rate,
            initial_epsilon=initial_epsilon,
            epsilon_decay_rate=epsilon_decay_rate,
        )
        self._critic = Critic(
            critic_function="table",
            learning_rate=critic_learning_rate,
            eligibility_decay_rate=critic_eligibility_decay_rate,
            discount_factor=critic_discount_factor,
        )
        self._prev_action = None
        self._state_action_pairs = list()

    def _get_all_episode_state_action_pairs(self):
        return self._state_action_pairs

    def _get_all_episode_states(self):
        return list(
            map(
                lambda state_action: state_action[0],
                self._get_all_episode_state_action_pairs(),
            )
        )

    def _get_previous_state(self):
        return self._state_action_pairs[-1][0]

    def _store_state_action_pair(self, state, action):
        self._state_action_pairs.append((state, action))
        self._prev_action = action

    def _run_updates(self, state, reward):
        """
        Run updates on actor and critic
        """
        delta = self._critic.get_temporal_difference_error(
            self._get_previous_state(), state, reward
        )

        self._critic = self._critic.update(
            self._get_previous_state(), self._get_all_episode_states(), delta
        )
        self._actor = self._actor.update(
            self._get_all_episode_state_action_pairs(), delta
        )

    def choose_action(self, state, possible_actions: List, reward=0):
        """
        Choose next action to perform.
        """
        if self._prev_action is None:
            action = self._actor.get_action(state, possible_actions)
            self._store_state_action_pair(state, action)
            return action

        action = self._actor.get_action(state, possible_actions)

        self._run_updates(state, reward)

        self._store_state_action_pair(state, action)

        return action

    def end_state_reached(self, state, reward):
        """
        docstring
        """
        self._run_updates(state, reward)
        self._critic = self._critic.reset_eligibilities()
        self._actor = self._actor.reset_eligibilities_and_decay_epsilon()
        self._state_action_pairs = list()
        self._prev_action = None