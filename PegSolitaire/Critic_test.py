from Critic import Critic
import pytest


def test_constructor():
    Critic()
    Critic(
        critic_function="table",
        critic_nn_dimensions=None,
        learning_rate=0.9,
        eligibility_decay_rate=0.9,
        discount_factor=0.9,
    )


def test_constructor_exceptions():
    with pytest.raises(ValueError):
        Critic(critic_function="sigmoid")

    with pytest.raises(ValueError):
        Critic(critic_function="neural_network")


def test_get_temporal_difference_error():
    td_error = Critic().get_temporal_difference_error((0, 1), (1, 1), 1)
    assert float(td_error) == td_error


def test_update_value_function():
    critic = Critic(
        learning_rate=0.1,
        discount_factor=0.9,
        eligibility_decay_rate=1,
    )
    initial_state_action = (0, 1)
    second_state_action = (1, 1)
    td_error = critic.get_temporal_difference_error(
        initial_state_action, second_state_action, 1
    )
    updated_critic = critic.update_value_function(initial_state_action, td_error)
    print(updated_critic._value_function)
    assert (
        updated_critic.get_temporal_difference_error(
            initial_state_action, second_state_action, 1
        )
        < td_error
    )


def test_update_eligibility():
    critic = Critic(init_state=0, init_action=1)

    updated_critic = critic.update_eligibility((0, 1))
