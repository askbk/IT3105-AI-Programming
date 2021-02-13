import pytest
import numpy as np
from Agent.Critic import Critic


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
    td_error = Critic().get_temporal_difference_error(0, 0, 1)
    assert float(td_error) == td_error


def test_update():
    state1 = 0
    state2 = 1
    reward1 = 0
    reward2 = 1

    critic = Critic(
        critic_function="table",
        learning_rate=0.1,
        discount_factor=0.9,
        eligibility_decay_rate=0.9,
        _random_initialization=False,
    )

    td_error = critic.get_temporal_difference_error(state1, state2, reward2)
    assert td_error == 1

    # move from state 1 to state 2 with reward 1
    critic = critic.update([state2, state1], td_error)

    td_error = critic.get_temporal_difference_error(state2, state1, reward1)
    assert round(td_error, 2) == 0.09

    # move from state 2 to state 1 with reward 0
    critic = critic.update([state1, state2], td_error)
    assert round(critic.get_temporal_difference_error(state2, state1, 0), 6) == 0.087561
    assert round(critic.get_temporal_difference_error(state1, state2, 1), 5) == 0.90081


def test_nn_critic():
    state1 = [0, 1]
    state2 = [1, 0]
    reward1 = [[0]]
    reward2 = [[1]]
    critic = Critic(
        critic_function="neural_network",
        critic_nn_dimensions=(2, 1),
        learning_rate=0.0001,
        discount_factor=0.9,
        eligibility_decay_rate=0.9,
    )

    td_error_1_2 = critic.get_temporal_difference_error(state1, state2, reward2)[0][0]

    # move from state 1 to state 2 with reward 1
    critic = critic.update([state2, state1], td_error_1_2)

    td_error_2_1 = critic.get_temporal_difference_error(state2, state1, reward1)[0][0]

    # move from state 2 to state 1 with reward 0
    critic = critic.update([state1, state2], td_error_2_1)
    assert (
        critic.get_temporal_difference_error(state2, state1, reward1)[0][0]
        != td_error_2_1
    )
    assert (
        critic.get_temporal_difference_error(state1, state2, reward2)[0][0]
        != td_error_1_2
    )
