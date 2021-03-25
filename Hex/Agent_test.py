from MCTS.Nim import Nim
from Agent import Agent


def test_agent_get_action():
    current_state = Nim(n=5, k=2)
    assert Agent(initial_state=Nim(n=6, k=2)).get_action(current_state) in current_state.get_possible_actions()


def test_agent_next_state():
    current_agent = Agent(initial_state=Nim(n=3, k=2))
    next_state = Nim(n=2, k=2)
    assert not current_agent.next_state(next_state, initial=True) is current_agent


def test_end_of_episode_update():
    current_agent = Agent(initial_state=Nim(n=2, k=2))
    assert not current_agent.end_of_episode_update() is current_agent
