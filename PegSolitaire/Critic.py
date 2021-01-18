class Critic:
    """
    The Critic part of the Actor-Critic model
    """

    def __init__(
        self,
        critic_function="table",
        critic_nn_dimensions=None,
        critic_learning_rate=0.9,
        critic_eligibility_decay_rate=0.9,
        critic_discount_factor=0.9,
    ):
        if critic_function not in ["table", "neural_network"]:
            raise ValueError(
                "critic_function must be either 'table' or 'neural_network'."
            )

        if critic_function == "neural_network" and critic_nn_dimensions is None:
            raise ValueError(
                "Dimensions of neural network must be supplied when critic_function='neural_network'."
            )
