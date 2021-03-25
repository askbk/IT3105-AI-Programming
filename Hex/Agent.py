import random
from Game import GameBase
from Actor import Actor
from MCTS import MCTS


class Agent:
    def __init__(
        self,
        initial_state=None,
        epsilon: float = 0,
        _replay_buffer=None,
        _actor=None,
        _mcts=None,
    ):
        self._replay_buffer = [] if _replay_buffer is None else _replay_buffer
        self._actor = Actor() if _actor is None else _actor
        self._epsilon = epsilon
        self._mcts = Agent._initialize_mcts(initial_state, _mcts)
        self._initial_state = initial_state

    @staticmethod
    def _initialize_mcts(initial_state, mcts):
        if mcts is not None:
            return mcts

        if initial_state is None:
            return None

        return MCTS(initial_state=initial_state).search()

    def get_action(self, current_state: GameBase):
        if random.random() < self._epsilon:
            return random.choice(current_state.get_possible_actions())
        return self._mcts.get_best_action()

    def next_state(self, next_state: GameBase, initial: bool):
        return Agent(
            initial_state=next_state,
            _mcts=None if initial else self._mcts.update_root(next_state).search(),
        )

    def end_of_episode_update(self):
        return Agent(initial_state=self._initial_state)