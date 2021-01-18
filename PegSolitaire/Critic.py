from __future__ import annotations
from typing import Tuple, List, Dict


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
        value_function=None,
        eligibilities=None,
    ):
        if critic_function not in ["table", "neural_network"]:
            raise ValueError(
                "critic_function must be either 'table' or 'neural_network'."
            )

        if critic_function == "neural_network" and critic_nn_dimensions is None:
            raise ValueError(
                "Dimensions of neural network must be supplied when critic_function='neural_network'."
            )

        if value_function is None:
            self._value_function = dict()
        else:
            self._value_function = value_function

        if eligibilities is None:
            self._eligibilities = dict()
        else:
            self._eligibilities = eligibilities

        self._learning_rate = learning_rate
        self._critic_function = critic_function
        self._critic_nn_dimensions = critic_nn_dimensions
        self._eligibility_decay_rate = eligibility_decay_rate
        self._discount_factor = discount_factor

    @staticmethod
    def _from_old_parameters(
        old: Critic,
        value_function: Dict[Tuple, float],
        eligibilities: Dict[Tuple, float],
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
            value_function=value_function,
            eligibilities=eligibilities,
        )

    def _get_value(self, state_action: Tuple):
        """
        Get the value for the state-action pair.
        """
        try:
            return self._value_function[state_action]
        except KeyError:
            return 0

    def get_temporal_difference_error(
        self, old_state_action: Tuple, new_state_action: Tuple, reward: float
    ):
        """
        Calculates the temporal difference error.
        """
        return (
            reward
            + self._discount_factor * self._get_value(new_state_action)
            - self._get_value(old_state_action)
        )

    def _get_eligibility(self, state_action: Tuple, was_previous=False):
        """
        Get eligibility of a state-action pair.
        """
        if was_previous:
            return 1

        try:
            return self._eligibilities[state_action]
        except KeyError:
            return 0

    def update(self, old_state_action, state_actions: List[Tuple], td_error: float):
        """
        Returns a new instance of the Critic with updated value function and eligibilities.
        """
        new_value_function = {
            **self._value_function,
            **{
                state_action: self._get_value(state_action)
                + self._learning_rate * td_error * self._get_eligibility(state_action)
                for state_action in state_actions
            },
        }

        new_eligibilities = {
            **self._eligibilities,
            **{
                state_action: self._discount_factor
                * self._eligibility_decay_rate
                * self._get_eligibility(
                    state_action, was_previous=(state_action == old_state_action)
                )
                for state_action in state_actions
            },
        }

        return Critic._from_old_parameters(self, new_value_function, new_eligibilities)