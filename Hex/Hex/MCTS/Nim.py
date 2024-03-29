import random
from Hex.Game import GameBase


class Nim(GameBase):
    def __init__(self, n, k, _player_turn=1):
        self._n = n
        self._k = k
        self._player_turn = _player_turn

    def get_player_turn(self):
        return self._player_turn

    def get_possible_actions(self):
        return list(range(1, min(self._n, self._k) + 1))

    def perform_action(self, rock_count):
        if self.is_end_state_reached():
            raise Exception

        if rock_count > self._k or rock_count > self._n:
            raise ValueError("Cannot take that many rocks")

        return Nim(
            n=self._n - rock_count, k=self._k, _player_turn=3 - self._player_turn
        )

    def is_end_state_reached(self):
        return self._n == 0

    def is_finished(self):
        if self.is_end_state_reached():
            return True, 3 - self._player_turn

        return False, None

    def get_tuple_representation(self):
        return (self._player_turn, self._n)

    def index_to_action(self, index):
        return index + 1

    def action_to_index(self, action):
        return action - 1

    def get_action_space_size(self):
        return self._k

    def get_state_size(self):
        return 2

    def __eq__(self, other):
        return self._n == other._n and self._k == other._k

    def __repr__(self):
        return f"Nim<n={self._n}, k={self._k}>"
