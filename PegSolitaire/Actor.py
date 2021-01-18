class Actor:
    """
    The Actor part of the Actor-Critic model.
    """

    def __init__(
        self,
        actor_discount_factor=0.9,
        actor_eligibility_decay_rate=0.9,
        actor_learning_rate=0.01,
        initial_epsilon=0.05,
        epsilon_decay_rate=0.9,
    ):
        pass