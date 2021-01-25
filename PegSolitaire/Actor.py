from __future__ import annotations
import random
from functools import reduce
from itertools import product


class Actor:
    """
    The Actor part of the Actor-Critic model.
    """

    def __init__(
        self,
        discount_factor=0.9,
        eligibility_decay_rate=0.9,
        learning_rate=0.01,
        initial_epsilon=0.05,
        epsilon_decay_rate=0.9,
        use_random_values=False,
        _eligibilities=None,
        _policy=None,
    ):
        if _eligibilities is not None:
            self._eligibilities = _eligibilities
        else:
            self._eligibilities = dict()

        if _policy is not None:
            self._policy = _policy
        else:
            self._policy = dict()

        self._discount_factor = discount_factor
        self._eligibility_decay_rate = eligibility_decay_rate
        self._learning_rate = learning_rate
        self._epsilon = initial_epsilon
        self._epsilon_decay_rate = epsilon_decay_rate
        self._use_random_values = use_random_values

    @staticmethod
    def _update(old: Actor, policy, eligibilities):
        """
        Returns an updated version of old.
        """
        return Actor(
            discount_factor=old._discount_factor,
            eligibility_decay_rate=old._eligibility_decay_rate,
            learning_rate=old._learning_rate,
            initial_epsilon=old._epsilon,
            epsilon_decay_rate=old._epsilon_decay_rate,
            use_random_values=old._use_random_values,
            _eligibilities=eligibilities,
            _policy=policy,
        )

    @staticmethod
    def _new(old: Actor, eligibilities):
        return Actor(
            discount_factor=old._discount_factor,
            eligibility_decay_rate=old._eligibility_decay_rate,
            learning_rate=old._learning_rate,
            initial_epsilon=old._epsilon * old._epsilon_decay_rate,
            epsilon_decay_rate=old._epsilon_decay_rate,
            use_random_values=old._use_random_values,
            _eligibilities=eligibilities,
            _policy=old._policy,
        )

    def _get_eligibility(self, state_action, was_previous):
        """
        Get eligibility of a state-action pair
        """
        if was_previous:
            return 1

        return self._eligibilities[state_action]

    def _get_action_value(self, state, action):
        """
        Get current policy value for an action in the given state
        """
        try:
            return self._policy[(state, action)]
        except KeyError:
            if self._use_random_values:
                return random.random()

            return 0

    def get_action(self, current_state, possible_actions):
        """
        Returns the action recommended by the actor.
        """
        action_values = map(
            lambda action: (action, self._get_action_value(current_state, action)),
            possible_actions,
        )

        best_action = reduce(
            lambda best, current: best if best[1] > current[1] else current,
            action_values,
        )[0]

        if random.random() < self._epsilon:
            return random.choice(possible_actions)

        return best_action

    def update(self, state_actions, td_error):
        """
        Returns a new instance of Actor with updated policy and eligibilities.
        The most recent state-action pair of the current episode should be last in state_actions.
        """
        new_policy = {
            **self._policy,
            **{
                state_action: self._get_action_value(state_action[0], state_action[1])
                + self._learning_rate
                * td_error
                * self._get_eligibility(
                    state_action, was_previous=(state_action == state_actions[-1])
                )
                for state_action in state_actions
            },
        }

        new_eligibilities = {
            **self._eligibilities,
            **{
                state_action: self._discount_factor
                * self._eligibility_decay_rate
                * self._get_eligibility(
                    state_action, was_previous=(state_action == state_actions[-1])
                )
                for state_action in state_actions
            },
        }

        return Actor._update(self, new_policy, new_eligibilities)

    def reset_eligibilities_and_decay_epsilon(self):
        """
        Returns a new instance of the Actor with reset eligibilities and decayed epsilon.
        """
        return Actor._new(
            self,
            eligibilities=dict.fromkeys(self._eligibilities, 0),
        )
