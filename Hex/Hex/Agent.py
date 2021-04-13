from __future__ import annotations
import random
import math
from typing import Optional, Dict
from operator import getitem, itemgetter
from Hex.Game import GameBase
from Hex.Actor import Actor
from Hex.MCTS import MCTS
from Hex.Types import ReplayBuffer, Action, RolloutPolicy


class Agent:
    def __init__(
        self,
        initial_state: GameBase,
        epsilon: float = 0,
        _replay_buffer: Optional[ReplayBuffer] = None,
        _actor: Actor = None,
        _mcts: Optional[MCTS] = None,
    ):
        self._replay_buffer = [] if _replay_buffer is None else _replay_buffer
        self._actor = _actor
        self._epsilon = epsilon
        self._mcts = Agent._initialize_mcts(initial_state, _mcts)
        self._initial_state = initial_state

    @staticmethod
    def _initialize_mcts(initial_state: GameBase, mcts: MCTS) -> Optional[MCTS]:
        if mcts is not None:
            return mcts

        return MCTS(initial_state=initial_state).search()

    @staticmethod
    def from_config(config: Dict, game: GameBase) -> Agent:
        return Agent(
            initial_state=game,
            epsilon=config.get("epsilon", 0),
            _actor=Actor.from_config(
                input_size=game.get_state_size(),
                output_size=game.get_action_space_size(),
                actor_config=config.get("actor", {}),
            ),
            _mcts=MCTS.from_config(config.get("mcts", {}), initial_state=game).search(),
        )

    @staticmethod
    def _rollout_policy(actor: Actor) -> RolloutPolicy:
        def rollout(state: GameBase) -> Action:
            probability_distribution = actor.rollout(state.get_tuple_representation())
            action_probabilities = [
                (probability, state.index_to_action(index))
                for index, probability in enumerate(probability_distribution)
            ]
            possible_action_probabilities = filter(
                lambda probability_action: getitem(probability_action, 1)
                in state.get_possible_actions(),
                action_probabilities,
            )

            return getitem(max(possible_action_probabilities, key=itemgetter(0)), 1)

        return rollout

    def get_action(self) -> Action:
        if random.random() < self._epsilon:
            return random.choice(self._initial_state.get_possible_actions())
        return self._mcts.get_best_action()

    def next_state(self, next_state: GameBase) -> Agent:
        action_visit_count_distribution = self._mcts.get_root_distribution()
        action_distribution = [0] * self._initial_state.get_action_space_size()
        total_visit = sum(
            visit_count for _, visit_count in action_visit_count_distribution
        )
        for action, visit_count in action_visit_count_distribution:
            action_distribution[self._initial_state.action_to_index(action)] = (
                visit_count / total_visit
            )
        replay_buffer = [
            *self._replay_buffer,
            (self._initial_state.get_tuple_representation(), action_distribution),
        ]
        return Agent(
            initial_state=next_state,
            _replay_buffer=replay_buffer,
            _mcts=self._mcts.update_root(next_state).search(
                rollout_policy=Agent._rollout_policy(self._actor)
            ),
            _actor=self._actor,
        )

    def end_of_episode_update(self, initial_state: GameBase) -> Agent:
        subset_size = min(len(self._replay_buffer), 100)
        training_subset = random.sample(self._replay_buffer, subset_size)
        return Agent(
            initial_state=initial_state,
            _actor=self._actor.train(training_subset),
            _replay_buffer=self._replay_buffer,
        )
