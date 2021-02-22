from itertools import product
from functools import reduce


class Board:
    """
    Hex board
    """

    def __init__(self, size=4, _board_state=None, _player_turn=None):
        self._size = size
        self._board_state = (0,) * size ** 2 if _board_state is None else _board_state
        self._player_turn = 1 if _player_turn is None else _player_turn

    @staticmethod
    def _get_valid_positions(size):
        return set(product(range(size), range(size)))

    @staticmethod
    def _get_next_player_turn(current_player_turn):
        return 3 - current_player_turn

    @staticmethod
    def _translate_index_to_coordinates(index, board_size):
        return (index // board_size, index % board_size)

    @staticmethod
    def _translate_coordinates_to_index(coordinates, board_size):
        return board_size * coordinates[0] + coordinates[1]

    def _is_position_occupied(self, position):
        return (
            self._board_state[
                Board._translate_coordinates_to_index(position, self._size)
            ]
            != 0
        )

    def _add_occupant(self, position):
        index = Board._translate_coordinates_to_index(position, self._size)

        return tuple(
            occupant if i != index else self._player_turn
            for i, occupant in enumerate(self._board_state)
        )

    def get_possible_moves(self):
        def vec2tuples(acc, index):
            if self._board_state[index]:
                return acc
            return [*acc, Board._translate_index_to_coordinates(index, self._size)]

        return reduce(vec2tuples, range(self._size ** 2), [])

    def make_move(self, position):
        if self._is_position_occupied(position):
            raise ValueError(f"Position {position} is already occupied.")

        if self.is_finished():
            raise Exception(f"Game is finished")

        return Board(
            size=self._size,
            _player_turn=self._get_next_player_turn(self._player_turn),
            _board_state=self._add_occupant(position),
        )

    def get_tuple_representation(self):
        return (self._get_next_player_turn(self._player_turn),) + tuple(
            self._board_state
        )

    def is_finished(self):
        raise NotImplementedError