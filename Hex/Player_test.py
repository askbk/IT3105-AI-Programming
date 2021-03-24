from MCTS.Nim import Nim
from Player import Player

def test_player_get_action():
    current_state = Nim(n=5, k=2)
    assert Player().get_action(current_state) in current_state.get_possible_actions()

def test_player_next_state():
    current_player = Player()
    next_state = Nim(n=2, k=2)
    assert not current_player.next_state(next_state) is current_player

def test_end_of_episode_update():
    current_player = Player()
    assert not current_player.end_of_episode_update() is current_player
