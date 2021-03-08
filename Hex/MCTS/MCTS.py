import random
import math
from functools import reduce
from Game import GameBase
from MCTS import Tree


class MCTS:
    """
    Monte Carlo Tree Search
    """

    def __init__(
        self,
        initial_state=None,
        search_games=100,
        _distribution=[],
        _player=1,
        _tree=None,
    ):
        self._distribution = _distribution
        self._search_games = search_games
        self._tree = Tree(initial_state) if _tree is None else _tree
        self._player = _player

    @staticmethod
    def _get_next_player(player):
        return 3 - player

    def get_root_distribution(self):
        return self._distribution

    @staticmethod
    def _upper_confidence_bound(parent, child, c=1):
        return child.get_value() / (1 + child.get_visit_count()) + c * math.sqrt(
            math.log(parent.get_visit_count()) / (1 + child.get_visit_count())
        )

    @staticmethod
    def _select_node_tree_policy(parent, children):
        return max(
            children, key=lambda child: MCTS._upper_confidence_bound(parent, child)
        )

    @staticmethod
    def _tree_search(rollout_policy) -> Tree:
        def perform_rollout(tree, recursively_called=False) -> int:
            """
            Returns new tree with updated statistics
            """
            # 1. rollout until end state reached using actor
            # 2. return reward
            if not tree.is_end_state():
                selected_child = rollout_policy(tree.get_children())

                return perform_rollout(selected_child)

            return 1

        def perform_search(tree, game_number) -> Tree:
            """
            Return new tree with updated statistics
            """
            # 1. follow tree policy until unvisited node reached
            # 2. perform rollout from node
            # 3. update tree with reward from rollout
            if not tree.is_visited():
                reward = perform_rollout(tree)
                return tree.increment_visit_count(reward=reward)

            selected_child = MCTS._select_node_tree_policy(tree, tree.get_children())
            updated_child = perform_search(selected_child, game_number)
            return tree.update_child_node(selected_child, updated_child)

        return perform_search

    def search(self):
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

    def update_root(self, new_root):
        return self
