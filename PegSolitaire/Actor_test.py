from Actor import Actor


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
