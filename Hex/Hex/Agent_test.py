import random
from Hex.MCTS.Nim import Nim
from Hex.Agent import Agent


def test_agent_get_action():
    k = 2
    current_state = Nim(n=5, k=k)
    assert (
        Agent.from_config({}, game=Nim(n=6, k=k)).get_action()
        in current_state.get_possible_actions()
    )


def test_agent_next_state():
    k = 2
    current_agent = Agent.from_config({}, game=Nim(n=3, k=k))
    next_state = Nim(n=2, k=k)
    assert not current_agent.next_state(next_state) is current_agent


def test_end_of_episode_update():
    k = 2
    state1 = Nim(n=2, k=k)
    agent = Agent.from_config({}, game=state1)
    state2 = state1.perform_action(agent.get_action())
    agent = agent.next_state(state2)
    agent = agent.end_of_episode_update(state1)
    state2 = state1.perform_action(agent.get_action())
    agent = agent.next_state(state2)


def test_nim_playthrough_player_1_wins():
    k = 3
    initial_state = Nim(n=7, k=k)
    state = initial_state
    agent = Agent.from_config({}, game=state)
    while True:
        action = agent.get_action()
        state = state.perform_action(action)
        agent = agent.next_state(state)
        if state.is_end_state_reached():
            agent = agent.end_of_episode_update(initial_state)
            break