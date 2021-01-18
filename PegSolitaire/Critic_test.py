from Critic import Critic
import pytest


def test_constructor():
    Critic()
    Critic(
        critic_function="table",
        critic_nn_dimensions=None,
        critic_learning_rate=0.9,
        critic_eligibility_decay_rate=0.9,
        critic_discount_factor=0.9,
    )


def test_constructor_exceptions():
    with pytest.raises(ValueError):
        Critic(critic_function="sigmoid")

    with pytest.raises(ValueError):
        Critic(critic_function="neural_network")