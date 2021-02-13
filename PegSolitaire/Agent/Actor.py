from __future__ import annotations
import random
from functools import reduce
from itertools import product
from Agent.EligibilityTable import EligibilityTable


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
        _eligibility_table=None,
        _policy=None,
    ):
        self._eligibility_table = (
            EligibilityTable(discount_factor, eligibility_decay_rate)
            if _eligibility_table is None
            else _eligibility_table
        )

        if _policy is not None:
            self._policy = _policy
        else:
            self._policy = dict()

        self._learning_rate = learning_rate
        self._epsilon = initial_epsilon
        self._epsilon_decay_rate = epsilon_decay_rate
        self._use_random_values = use_random_values

    @staticmethod
    def _update(old: Actor, policy, eligibility_table):
        """
        Returns an updated version of old.
        """
        return Actor(
            learning_rate=old._learning_rate,
            initial_epsilon=old._epsilon,
            epsilon_decay_rate=old._epsilon_decay_rate,
            use_random_values=old._use_random_values,
            _eligibility_table=eligibility_table,
            _policy=policy,
        )

    @staticmethod
    def _new(old: Actor, eligibility_table):
        return Actor(
            learning_rate=old._learning_rate,
            initial_epsilon=old._epsilon * old._epsilon_decay_rate,
            epsilon_decay_rate=old._epsilon_decay_rate,
            use_random_values=old._use_random_values,
            _eligibility_table=eligibility_table,
            _policy=old._policy,
        )

    def _get_action_value(self, state, action):
        """
        Get current policy value for an action in the given state
        """
        try:
            return self._policy[str((state, action))]
        except KeyError:
            if self._use_random_values:
                return random.random()

            return 0

    def _get_action_deterministic(self, current_state, possible_actions):
        action_values = map(
            lambda action: (action, self._get_action_value(current_state, action)),
            possible_actions,
        )
        return reduce(
            lambda best, current: best if best[1] > current[1] else current,
            action_values,
        )[0]

    def get_action(self, current_state, possible_actions, epsilon_greedy=True):
        """
        Returns the action recommended by the actor.
        """
        best_action = self._get_action_deterministic(current_state, possible_actions)

        if epsilon_greedy and random.random() < self._epsilon:
            return random.choice(possible_actions)

        return best_action

    def update(self, state_actions, td_error):
        """
        Returns a new instance of Actor with updated policy and eligibilities.
        The most recent state-action pair of the current episode should be last in state_actions.
        """
        new_policy = {
            str(state_action): self._get_action_value(state_action[0], state_action[1])
            + self._learning_rate
            * td_error
            * self._eligibility_table.get_eligibility(
                state_action, was_previous=(state_action == state_actions[-1])
            )
            for state_action in state_actions
        }

        return Actor._update(
            self,
            self._policy | new_policy,
            self._eligibility_table.update_eligibilities(state_actions),
        )

    def reset_eligibilities_and_decay_epsilon(self):
        """
        Returns a new instance of the Actor with reset eligibilities and decayed epsilon.
        """
        return Actor._new(self, eligibility_table=self._eligibility_table.reset())
