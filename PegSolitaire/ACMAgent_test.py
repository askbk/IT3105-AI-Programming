from ACMAgent import ACMAgent
from itertools import accumulate
from functools import reduce
import random


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


def test_agent_performance_improves_over_time():
    possible_actions = {
        0: random.sample([0, 1], k=2),
        1: random.sample([-1, 0, 1], k=3),
        2: random.sample([-2, -1, 0, 1], k=4),
        3: random.sample([-3, -2, -1, 0], k=4),
    }
    rewards = {0: -1, 1: -1, 2: 1, 3: 3}
    agent = ACMAgent(
        actor_discount_factor=0.9,
        actor_eligibility_decay_rate=0.95,
        actor_learning_rate=0.1,
        critic_function="table",
        critic_learning_rate=0.1,
        critic_eligibility_decay_rate=0.95,
        critic_discount_factor=0.9,
        initial_epsilon=0.1,
        epsilon_decay_rate=0.95,
    )

    def sample_game_logic(state_actions, _index):
        current_state = state_actions[-1]
        action = agent.choose_action(
            current_state,
            possible_actions[current_state],
            reward=rewards[current_state],
        )
        next_state = current_state + action
        return [*state_actions[:-1], (current_state, action), next_state]

    episode_size = 50
    state_actions = list(
        reduce(
            sample_game_logic,
            range(episode_size),
            [0],
        )
    )[:-1]

    gained_rewards = list(
        map(lambda state_action: rewards[state_action[0]], state_actions)
    )

    accumulated_rewards = list(accumulate(gained_rewards))

    assert accumulated_rewards[0] < accumulated_rewards[-1]
    assert accumulated_rewards[episode_size // 2] < accumulated_rewards[-1]
    assert sum(gained_rewards[: episode_size // 2]) < sum(
        gained_rewards[episode_size // 2 :]
    )
