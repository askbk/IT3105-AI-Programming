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


def test_update():
    state_action1 = (0, 1)
    state_action2 = (1, -1)

    critic = Critic(
        critic_function="table",
        learning_rate=0.1,
        discount_factor=0.9,
        eligibility_decay_rate=0.9,
        value_function={state_action1: 0.1, state_action2: -0.1},
        eligibilities={state_action1: 0.09, state_action2: 1},
    )

    td_error = critic.get_temporal_difference_error(state_action2, state_action1, 0)

    updated_critic = critic.update(
        state_action2, [state_action1, state_action2], td_error
    )

    assert (
        round(
            updated_critic.get_temporal_difference_error(
                state_action1, state_action2, 1
            ),
            5,
        )
        == 0.82539
    )
    assert (
        round(
            updated_critic.get_temporal_difference_error(
                state_action2, state_action1, 0
            ),
            6,
        )
        == 0.172539
    )
