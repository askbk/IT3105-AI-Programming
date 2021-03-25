import random
from typing import List, Any
from PegSolitaire.Agent import FunctionApproximator, EligibilityTable


class TableFunctionApproximator(FunctionApproximator):
    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
        eligibility_decay_rate: float = None,
        _value_table=None,
        _eligibility_table=None,
        random_initialization=True,
    ):
        self._value_table = dict() if _value_table is None else _value_table
        self._eligibility_table = (
            EligibilityTable(discount_factor, eligibility_decay_rate)
            if _eligibility_table is None
            else _eligibility_table
        )
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._random_initialisation = random_initialization

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
            * self._eligibility_table.get_eligibility(
                state, was_previous=(state == states[-1])
            )
            for state in states
        }

        return TableFunctionApproximator(
            learning_rate=self._learning_rate,
            discount_factor=self._discount_factor,
            _value_table=self._value_table | new_value_function,
            _eligibility_table=self._eligibility_table.update_eligibilities(states),
        )

    def reset_eligibilities(self):
        return TableFunctionApproximator(
            learning_rate=self._learning_rate,
            discount_factor=self._discount_factor,
            _value_table=self._value_table,
            _eligibility_table=self._eligibility_table.reset(),
        )
