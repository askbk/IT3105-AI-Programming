import numpy as np
from Agent.Actor import Actor


def test_constructor():
    Actor()
    Actor(
        discount_factor=0.9,
        eligibility_decay_rate=0.9,
        learning_rate=0.01,
        initial_epsilon=0.05,
        epsilon_decay_rate=0.9,
    )


def test_get_action():
    current_state = 0
    possible_actions = [0, 1]
    assert Actor().get_action(current_state, possible_actions) in possible_actions


def test_update():
    state_action1 = (0, 1)
    state_action2 = (1, -1)
    state_action3 = (1, 0)
    td_error = 1
    actor = Actor(
        initial_epsilon=0,
        _eligibilities={
            state_action1: 0.9,
            state_action2: 0.8,
            state_action3: 1,
        },
    )
    actor = actor.update([state_action1, state_action2, state_action3], td_error)

    assert actor.get_action(current_state=1, possible_actions=[-1, 0, 1]) == 0


def test_epsilon_greedy_actor():
    state_action1 = (0, 1)
    state_action2 = (1, -1)
    state_action3 = (1, 0)
    td_error = 1
    epsilon = 0.1
    actor = Actor(
        initial_epsilon=epsilon,
        epsilon_decay_rate=1,
        _eligibilities={
            state_action1: 0.9,
            state_action2: 0.8,
            state_action3: 1,
        },
    )
    actor = actor.update([state_action1, state_action2, state_action3], td_error)
    rounds = 10000
    possible_actions = [-1, 0, 1]
    actions = list(
        map(
            lambda _: actor.get_action(
                current_state=1, possible_actions=possible_actions
            ),
            range(rounds),
        )
    )

    assert np.isclose(
        actions.count(0) / rounds,
        (1 - epsilon) + epsilon * (1 / len(possible_actions)),
        rtol=0.01,
    )
