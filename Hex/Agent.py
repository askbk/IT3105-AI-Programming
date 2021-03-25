import random
from Game import GameBase
from Actor import Actor
from MCTS import MCTS


class Agent:
    def __init__(self, initial_state, epsilon: float = 0, _replay_buffer=None, _actor=None, _mcts=None):
        self._replay_buffer = [] if _replay_buffer is None else _replay_buffer
        self._actor = Actor() if _actor is None else _actor
        self._epsilon = epsilon
        self._mcts = MCTS(initial_state=initial_state) if _mcts is None else _mcts
        self._initial_state = initial_state

    def get_action(self, current_state: GameBase):
        if random.random() < self._epsilon:
            return random.choice(current_state.get_possible_actions())

        updated_mcts = self._mcts.search()
        best_action = updated_mcts.get_best_action()
        return best_action

    def next_state(self, next_state: GameBase, initial:bool):
        return Agent(initial_state=next_state, _mcts=None if initial else self._mcts.update_root(next_state))

    def end_of_episode_update(self):
        return Agent(initial_state=self._initial_state)