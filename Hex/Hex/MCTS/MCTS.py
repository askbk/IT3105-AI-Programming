from __future__ import annotations
import math
import random
import operator
import time
from functools import reduce
from typing import Optional, Dict, Callable, Sequence
from Hex.Game import GameBase
from Hex.MCTS import Tree
from Hex.Types import RolloutPolicy, Action, TreePolicy


class MCTS:
    """
    Monte Carlo Tree Search
    """

    def __init__(
        self,
        initial_state: Optional[GameBase] = None,
        search_games: int = 100,
        exploration_coefficient: float = 3,
        time_limit: float = 1,
        _distribution=[],
        _tree: Tree = None,
    ):
        self._distribution = _distribution
        self._search_games = search_games
        self._tree = Tree(initial_state) if _tree is None else _tree
        self._exploration_coefficient = exploration_coefficient
        self._time_limit = time_limit

    @staticmethod
    def from_config(mcts_config: Dict, initial_state: GameBase):
        return MCTS(
            initial_state=initial_state,
            search_games=mcts_config.get("search_games", 100),
            exploration_coefficient=mcts_config.get("exploration_coefficient", 3),
            time_limit=mcts_config.get("time_limit", 1),
        )

    def get_root_distribution(self) -> list:
        return self._distribution

    @staticmethod
    def _upper_confidence_bound(
        player_turn: int, parent: Tree, exploration_coefficient=3.0
    ) -> Callable[[Tree], float]:
        value_multiplier = 1 if player_turn == 1 else -1
        return (
            lambda child: value_multiplier * child.get_value()
            + exploration_coefficient
            * math.sqrt(
                math.log(parent.get_visit_count()) / (0.01 + child.get_visit_count())
            )
        )

    @staticmethod
    def _tree_policy(exploration_coefficient: float) -> TreePolicy:
        def policy(parent: Tree, children: Sequence[Tree]) -> Tree:
            return max(
                children,
                key=MCTS._upper_confidence_bound(
                    parent.get_state().get_player_turn(),
                    parent,
                    exploration_coefficient,
                ),
            )

        return policy

    @staticmethod
    def _tree_search(
        rollout_policy: RolloutPolicy,
        tree_policy: TreePolicy,
        start_time: float,
        time_limit: float,
    ) -> Callable[[Tree, int], Tree]:
        def perform_rollout(state: GameBase) -> int:
            """
            Returns reward from rollout.
            """
            # 1. rollout until end state reached using actor
            # 2. return reward
            is_finished, winner = state.is_finished()
            if is_finished:
                return 1 if winner == 1 else -1

            return perform_rollout(state.perform_action(rollout_policy(state)))

        def perform_search(tree: Tree, game_number: int) -> Tree:
            """
            Return new tree with updated statistics.
            Stops if time limit is reached.
            """
            # 1. follow tree policy until unvisited node reached
            # 2. perform rollout from node
            # 3. update tree with reward from rollout
            if time.time() - start_time > time_limit:
                return tree

            if not tree.is_visited() or tree.is_end_state():
                reward = perform_rollout(tree.get_state())
                return tree.increment_visit_count(reward=reward)

            selected_child = tree_policy(tree, tree.get_children())
            updated_child = perform_search(selected_child, game_number)
            return tree.update_child_node(selected_child, updated_child)

        return perform_search

    def search(
        self,
        rollout_policy: RolloutPolicy = lambda state: random.choice(
            state.get_possible_actions()
        ),
    ) -> MCTS:
        if self._tree.is_end_state():
            return self

        new_tree = reduce(
            MCTS._tree_search(
                rollout_policy,
                MCTS._tree_policy(self._exploration_coefficient),
                start_time=time.time(),
                time_limit=self._time_limit,
            ),
            range(self._search_games + 1),
            self._tree,
        )
        distribution = [
            (child.get_action(), child.get_visit_count())
            for child in new_tree.get_children()
        ]

        return MCTS(
            search_games=self._search_games,
            _tree=new_tree,
            _distribution=distribution,
        )

    def get_best_action(self) -> Action:
        """
        Returns action most often taken from current state.
        """
        if len(self.get_root_distribution()) == 0:
            raise ValueError(
                "Root distribution is empty - did you remember to run search()?"
            )
        return operator.getitem(
            max(self.get_root_distribution(), key=operator.itemgetter(1)), 0
        )

    def update_root(self, new_root_state: GameBase) -> MCTS:
        new_tree = next(
            child
            for child in self._tree.get_children()
            if child.get_state() == new_root_state
        )
        return MCTS(
            search_games=self._search_games,
            _tree=new_tree,
        )

    def __repr__(self):
        return f"MCTS<root_state={self._tree.get_state()}>"
