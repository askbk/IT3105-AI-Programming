import random
from Game import GameBase
from Actor import Actor
from MCTS import MCTS


class Agent:
    def __init__(self, _replay_buffer=None, _actor=None):
        self._replay_buffer = [] if _replay_buffer is None else _replay_buffer
        self._actor = Actor() if _actor is None else _actor

    def get_action(self, current_state: GameBase):
        return random.choice(current_state.get_possible_actions())

    def next_state(self, next_state: GameBase):
        return Agent()

    def end_of_episode_update(self):
        return Agent()