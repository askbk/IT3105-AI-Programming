from __future__ import annotations


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

        self._learning_rate = learning_rate
        self._critic_function = critic_function
        self._critic_nn_dimensions = critic_nn_dimensions
        self._eligibility_decay_rate = eligibility_decay_rate
        self._discount_factor = discount_factor

    @staticmethod
    def _from_value_function(old: Critic, value_function):
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
        )

    def _get_value(self, state, action):
        """
        Get the value for the state-action pair.
        """
        try:
            return self._value_function[(state, action)]
        except KeyError:
            return 0

    def get_temporal_difference_error(self, state, action, reward):
        """
        Calculates the temporal difference error.
        """
        return (
            reward
            + self._discount_factor * self._get_value(state, action)
            - self._get_value(state, action)
        )

    def update_value_function(self, state, action, td_error):
        """
        Returns a new instance of the Critic with an updated value function.
        """
        old_value = self._get_value(state, action)
        new_value = old_value + self._learning_rate * td_error
        new_value_function = dict(self._value_function)

        return Critic()._from_value_function(
            self, {**new_value_function, (state, action): new_value}
        )
