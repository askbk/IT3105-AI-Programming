from Actor import Actor


def test_constructor():
    Actor()
    Actor(
        actor_discount_factor=0.9,
        actor_eligibility_decay_rate=0.9,
        actor_learning_rate=0.01,
        initial_epsilon=0.05,
        epsilon_decay_rate=0.9,
    )


def test_get_action():
    current_state = 0
    possible_actions = [0, 1]
    assert Actor().get_action(current_state, possible_actions) in possible_actions


def test_update():
    actor = Actor(initial_epsilon=0)
    action = actor.get_action(0, [0, 1])
    state_action1 = (0, 1)
    state_action2 = (1, -1)
    state_action3 = (0, 1)
    state_action4 = (1, 0)
    td_error = 1
    actor = actor.update(state_action1, [state_action1, state_action2], td_error)
