import random
from operator import getitem, itemgetter
from Hex.Game import GameBase
from Hex.Actor import Actor
from Hex.MCTS import MCTS


class Agent:
    def __init__(
        self,
        state_size: int,
        action_space_size: int,
        initial_state=None,
        epsilon: float = 0,
        _replay_buffer=None,
        _actor=None,
        _mcts=None,
    ):
        self._replay_buffer = [] if _replay_buffer is None else _replay_buffer
        self._actor = (
            Actor(input_size=state_size, output_size=action_space_size)
            if _actor is None
            else _actor
        )
        self._epsilon = epsilon
        self._mcts = Agent._initialize_mcts(initial_state, _mcts)
        self._initial_state = initial_state
        self._state_size = state_size
        self._action_space_size = action_space_size

    @staticmethod
    def _initialize_mcts(initial_state, mcts):
        if mcts is not None:
            return mcts

        if initial_state is None:
            return None

        return MCTS(initial_state=initial_state).search()

    @staticmethod
    def _rollout_policy(actor: Actor):
        def rollout(state: GameBase):
            probability_distribution = actor.rollout(state.get_tuple_representation())
            action_probabilities = [
                (probability, state.index_to_action(index))
                for index, probability in enumerate(probability_distribution)
            ]
            possible_actions = state.get_possible_actions()
            possible_action_probabilities = filter(
                lambda probability_action: getitem(probability_action, 1)
                in possible_actions,
                action_probabilities,
            )

            return getitem(max(possible_action_probabilities, key=itemgetter(0)), 1)

        return rollout

    def get_action(self, current_state: GameBase):
        if random.random() < self._epsilon:
            return random.choice(current_state.get_possible_actions())
        return self._mcts.get_best_action()

    def next_state(self, next_state: GameBase, initial: bool):
        return Agent(
            initial_state=next_state,
            state_size=self._state_size,
            action_space_size=self._action_space_size,
            _mcts=None
            if initial
            else self._mcts.update_root(next_state).search(
                rollout_policy=Agent._rollout_policy(self._actor)
            ),
        )

    def end_of_episode_update(self):
        return Agent(
            initial_state=self._initial_state,
            state_size=self._state_size,
            action_space_size=self._action_space_size,
        )
