import random
from Hex.MCTS.Nim import Nim
from Hex.Agent import Agent


def test_agent_get_action():
    k = 2
    current_state = Nim(n=5, k=k)
    assert (
        Agent(
            initial_state=Nim(n=6, k=k), state_size=2, action_space_size=k
        ).get_action()
        in current_state.get_possible_actions()
    )


def test_agent_next_state():
    k = 2
    current_agent = Agent(
        initial_state=Nim(n=3, k=k), state_size=2, action_space_size=k
    )
    next_state = Nim(n=2, k=k)
    assert not current_agent.next_state(next_state) is current_agent


def test_end_of_episode_update():
    k = 2
    state = Nim(n=2, k=k)
    agent = Agent(initial_state=state, state_size=2, action_space_size=k)
    action = random.choice(state.get_possible_actions())
    agent = agent.next_state(state.perform_action(action))
    agent.end_of_episode_update()


def test_nim_playthrough_player_1_wins():
    k = 3
    state = Nim(n=7, k=k)
    agent = Agent(initial_state=state, state_size=2, action_space_size=k)
    while True:
        action = agent.get_action()
        state = state.perform_action(action)
        agent = agent.next_state(state)
        if state.is_end_state_reached():
            agent = agent.end_of_episode_update()
            break