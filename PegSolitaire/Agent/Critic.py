from __future__ import annotations
import random
from typing import Tuple, List, Dict, Any, Optional
from Agent.TableFunctionApproximator import TableFunctionApproximator
from Agent.FunctionApproximator import FunctionApproximator


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
        _value_function: Optional[FunctionApproximator] = None,
        _random_initialization=True,
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
            if critic_function == "table":
                self._value_function = TableFunctionApproximator(
                    learning_rate=learning_rate,
                    discount_factor=discount_factor,
                    eligibility_decay_rate=eligibility_decay_rate,
                    random_initialization=_random_initialization,
                )
        else:
            self._value_function = _value_function

        self._discount_factor = discount_factor

    @staticmethod
    def _from_old_parameters(
        old: Critic,
        value_function: Dict[Any, float],
    ):
        """
        Creates a new Critic instance with the same parameters as old, but with a new value function.
        """
        return Critic(
            discount_factor=old._discount_factor,
            _value_function=value_function,
        )

    def get_temporal_difference_error(
        self, old_state: Any, new_state: Any, reward: float
    ):
        """
        Calculates the temporal difference error.
        """
        return (
            reward
            + self._discount_factor * self._value_function.get_value(new_state)
            - self._value_function.get_value(old_state)
        )

    def update(self, states: List, td_error: float):
        """
        Returns a new instance of the Critic with updated value function and eligibilities.
        The most recent state of the current episode should be last in states.
        """
        return Critic._from_old_parameters(
            self, value_function=self._value_function.update(states, td_error)
        )

    def reset_eligibilities(self):
        """
        Returns a new instance of the Critic with reset eligibilities.
        """
        return Critic._from_old_parameters(
            self,
            value_function=self._value_function.reset_eligibilities(),
        )
