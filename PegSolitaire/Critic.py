from __future__ import annotations
from typing import Tuple, List, Dict, Any
import random


class Critic:
    """
    The Critic part of the Actor-Critic model
    """

    def __init__(
        self,
        critic_function="table",
        critic_nn_dimensions=None,
        learning_rate=0.9,
        eligibility_decay_rate=0.9,
        discount_factor=0.9,
        use_random_values=False,
        _value_function=None,
        _eligibilities=None,
    ):
        if critic_function not in ["table", "neural_network"]:
            raise ValueError(
                "critic_function must be either 'table' or 'neural_network'."
            )

        if critic_function == "neural_network" and critic_nn_dimensions is None:
            raise ValueError(
                "Dimensions of neural network must be supplied when critic_function='neural_network'."
            )

        if _value_function is None:
            self._value_function = dict()
        else:
            self._value_function = _value_function

        if _eligibilities is None:
            self._eligibilities = dict()
        else:
            self._eligibilities = _eligibilities

        self._learning_rate = learning_rate
        self._critic_function = critic_function
        self._critic_nn_dimensions = critic_nn_dimensions
        self._eligibility_decay_rate = eligibility_decay_rate
        self._discount_factor = discount_factor
        self._use_random_values = use_random_values

    @staticmethod
    def _from_old_parameters(
        old: Critic,
        value_function: Dict[Any, float],
        eligibilities: Dict[Any, float],
    ):
        """
        Creates a new Critic instance with the same parameters as old, but with a new value function.
        """
        return Critic(
            critic_function=old._critic_function,
            critic_nn_dimensions=old._critic_nn_dimensions,
            learning_rate=old._learning_rate,
            eligibility_decay_rate=old._eligibility_decay_rate,
            discount_factor=old._discount_factor,
            use_random_values=old._use_random_values,
            _value_function=value_function,
            _eligibilities=eligibilities,
        )

    def _get_value(self, state: Any):
        """
        Get the value for the state.
        """
        try:
            return self._value_function[state]
        except KeyError:
            if self._use_random_values:
                return random.random()

            return 0

    def get_temporal_difference_error(
        self, old_state: Any, new_state: Any, reward: float
    ):
        """
        Calculates the temporal difference error.
        """
        return (
            reward
            + self._discount_factor * self._get_value(new_state)
            - self._get_value(old_state)
        )

    def _get_eligibility(self, state: Any, was_previous=False):
        """
        Get eligibility of a state.
        """
        if was_previous:
            return 1

        try:
            return self._eligibilities[state]
        except KeyError:
            return 0

    def update(self, old_state: Any, states: List, td_error: float):
        """
        Returns a new instance of the Critic with updated value function and eligibilities.
        """
        new_value_function = {
            **self._value_function,
            **{
                state: self._get_value(state)
                + self._learning_rate
                * td_error
                * self._get_eligibility(state, was_previous=(state == old_state))
                for state in states
            },
        }

        new_eligibilities = {
            **self._eligibilities,
            **{
                state: self._discount_factor
                * self._eligibility_decay_rate
                * self._get_eligibility(state, was_previous=(state == old_state))
                for state in states
            },
        }

        return Critic._from_old_parameters(self, new_value_function, new_eligibilities)

    def reset_eligibilities(self):
        """
        Returns a new instance of the Critic with reset eligibilities.
        """
        return Critic._from_old_parameters(
            self,
            value_function=self._value_function,
            eligibilities=dict.fromkeys(self._eligibilities, 0),
        )
