from __future__ import annotations
import random
import math
from functools import reduce
from typing import Callable, Any
from Game import GameBase
from MCTS import Tree


class MCTS:
    """
    Monte Carlo Tree Search
    """

    def __init__(
        self,
        initial_state: Any = None,
        search_games=100,
        _distribution=[],
        _player=1,
        _tree: Tree = None,
    ):
        self._distribution = _distribution
        self._search_games = search_games
        self._tree = Tree(initial_state) if _tree is None else _tree
        self._player = _player

    @staticmethod
    def _get_next_player(player: int) -> int:
        return 3 - player

    def get_root_distribution(self) -> list:
        return self._distribution

    @staticmethod
    def _upper_confidence_bound(parent: Tree, child: Tree, c=1) -> float:
        return child.get_value() / (1 + child.get_visit_count()) + c * math.sqrt(
            math.log(parent.get_visit_count()) / (1 + child.get_visit_count())
        )

    @staticmethod
    def _select_node_tree_policy(parent: Tree, children: Tree, player: int) -> Tree:
        func = max if player == 1 else min
        return func(
            children, key=lambda child: MCTS._upper_confidence_bound(parent, child)
        )

    @staticmethod
    def _tree_search(rollout_policy: Callable) -> Tree:
        def perform_rollout(state: GameBase, player_turn: int) -> int:
            """
            Returns reward from rollout.
            """
            # 1. rollout until end state reached using actor
            # 2. return reward
            if not state.is_end_state_reached():
                action = rollout_policy(state.get_possible_actions())

                return perform_rollout(
                    state.perform_action(action), MCTS._get_next_player(player_turn)
                )

            return 1 if player_turn == 1 else -1

        def perform_search(tree: Tree, game_number: int, player_turn=1) -> Tree:
            """
            Return new tree with updated statistics.
            """
            # 1. follow tree policy until unvisited node reached
            # 2. perform rollout from node
            # 3. update tree with reward from rollout
            if not tree.is_visited() or tree.is_end_state():
                reward = perform_rollout(tree.get_state(), player_turn)
                return tree.increment_visit_count(reward=reward)

            selected_child = MCTS._select_node_tree_policy(
                tree, tree.get_children(), player_turn
            )
            updated_child = perform_search(
                selected_child,
                game_number,
                player_turn=MCTS._get_next_player(player_turn),
            )
            return tree.update_child_node(selected_child, updated_child)

        return perform_search

    def search(self) -> MCTS:
        new_tree = reduce(
            MCTS._tree_search(random.choice),
            range(self._search_games + 1),
            self._tree,
        )
        distribution = [
            ((child.get_state(), child.get_action()), child.get_visit_count())
            for child in new_tree.get_children()
        ]

        return MCTS(
            search_games=self._search_games,
            _tree=new_tree,
            _player=self._player,
            _distribution=distribution,
        )

    def update_root(self, new_root_state: Any) -> MCTS:
        new_tree = next(
            child
            for child in self._tree.get_children()
            if child.get_state() == new_root_state
        )
        return MCTS(
            initial_state=new_tree.get_state(),
            search_games=self._search_games,
            _player=MCTS._get_next_player(self._player),
            _tree=new_tree,
        )
