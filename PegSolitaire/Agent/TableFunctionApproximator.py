import random
from typing import List, Any
from Agent.FunctionApproximator import FunctionApproximator


class TableFunctionApproximator(FunctionApproximator):
    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
        eligibility_decay_rate: float,
        _value_table=None,
        _eligibility_table=None,
        random_initialization=True,
    ):
        self._value_table = dict() if _value_table is None else _value_table
        self._eligibility_table = (
            dict() if _eligibility_table is None else _eligibility_table
        )
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._eligibility_decay_rate = eligibility_decay_rate
        self._random_initialisation = random_initialization

    def _get_eligibility(self, state: Any, was_previous=False):
        """
        Get eligibility of a state.
        """
        if was_previous:
            return 1

        try:
            return self._eligibility_table[state]
        except KeyError:
            return 0

    def get_value(self, state: Any):
        """
        Get the value for the state.
        """
        try:
            return self._value_table[state]
        except KeyError:
            if not self._random_initialisation:
                return 0
            return random.uniform(-0.1, 0.1)

    def update(self, states: List, td_error: float):
        new_value_function = {
            state: self.get_value(state)
            + self._learning_rate
            * td_error
            * self._get_eligibility(state, was_previous=(state == states[-1]))
            for state in states
        }

        new_eligibilities = {
            state: self._discount_factor
            * self._eligibility_decay_rate
            * self._get_eligibility(state, was_previous=(state == states[-1]))
            for state in states
        }

        return TableFunctionApproximator(
            learning_rate=self._learning_rate,
            discount_factor=self._discount_factor,
            eligibility_decay_rate=self._eligibility_decay_rate,
            _value_table=self._value_table | new_value_function,
            _eligibility_table=self._eligibility_table | new_eligibilities,
        )

    def reset_eligibilities(self):
        return TableFunctionApproximator(
            learning_rate=self._learning_rate,
            discount_factor=self._discount_factor,
            eligibility_decay_rate=self._eligibility_decay_rate,
            _value_table=self._value_table,
            _eligibility_table=dict.fromkeys(self._eligibility_table, 0),
        )
