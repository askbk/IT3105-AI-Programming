class MCTS:
    """
    Monte Carlo Tree Search
    """

    def __init__(self, initial_state, search_games, _distribution=[]):
        self._distribution = _distribution
        self._initial_state = initial_state

    def get_root_distribution(self):
        return self._distribution

    def search(self):
        return MCTS(1, 1, [((self._initial_state, 1), 100)])

    def update_root(self, new_root):
        return self
