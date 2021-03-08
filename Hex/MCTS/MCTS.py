import random
import math
from functools import reduce
from Game import GameBase


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
        return child.get_value() / child.get_visit_count() + c * math.sqrt(
            math.log(parent.get_visit_count()) / child.get_visit_count()
        )

    @staticmethod
    def _select_node_tree_policy(parent, children):
        return max(
            children, key=lambda child: MCTS._upper_confidence_bound(parent, child)
        )

    @staticmethod
    def _tree_search(rollout_policy):
        def perform_rollout(tree):
            """
            Returns new tree with updated statistics
            """
            # 1. rollout until end state reached
            # 2. backpropagate statistics to root node
            # 3. return updated tree
            return tree

        def perform_search(tree):
            """
            Return new tree with updated statistics
            """
            # 1. follow tree policy until unvisited node reached
            # 2. perform rollout from node
            # 3. return rollout subtree and update
            if not tree.is_visited():
                return perform_rollout(tree)

            # use Pt to search from root to leaf of MCT and update Bmc with each move
            # use actor to choose rollout actions from leaf to final state and update Bmc with each move
            # perform MCTS backprop from final state to root
            selected_child = MCTS._select_node_tree_policy(tree, tree.get_children)
            updated_child = perform_search(selected_child)
            return tree.update_child_node(selected_child, updated_child)

        return perform_search

    def search(self):
        new_tree = reduce(
            MCTS._tree_search(random.choice),
            range(self._search_games),
            self._tree,
        )

        return MCTS(
            search_games=self._search_games, _tree=new_tree, _player=self._player
        )

    def update_root(self, new_root):
        return self
