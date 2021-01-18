from ACMAgent import ACMAgent


def test_agent_constructor():
    ACMAgent()
    ACMAgent(
        actor_discount_factor=0.9,
        actor_eligibility_decay_rate=0.9,
        actor_learning_rate=0.01,
        critic_function="table",
        critic_nn_dimensions=None,
        critic_learning_rate=0.9,
        critic_eligibility_decay_rate=0.9,
        critic_discount_factor=0.9,
        initial_epsilon=0.05,
        epsilon_decay_rate=0.9,
    )


def test_agent_choose_action():
    possible_actions = [1, 2, 3, 4]
    assert ACMAgent().choose_action(0, possible_actions, 0) in possible_actions