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
    def _are_positions_adjacent(position_a, position_b):
        if position_a[0] == position_b[0]:
            return abs(position_a[1] - position_b[1]) == 1

        if position_a[1] == position_b[1]:
            return abs(position_a[0] - position_b[0]) == 1

        if position_a[0] + 1 == position_b[0]:
            return position_a[1] - 1 == position_b[1]

        if position_a[0] - 1 == position_b[0]:
            return position_a[1] + 1 == position_b[1]

        return False

    @staticmethod
    def _get_next_player_turn(current_player_turn):
        return 3 - current_player_turn

    @staticmethod
    def _translate_index_to_coordinates(index, board_size):
        return (index // board_size, index % board_size)

    @staticmethod
    def _translate_coordinates_to_index(coordinates, board_size):
        return board_size * coordinates[0] + coordinates[1]

    def _is_position_occupied(self, position, player=None):
        position_index = Board._translate_coordinates_to_index(position, self._size)
        if player is None:
            return self._board_state[position_index] != 0

        return self._board_state[position_index] == player

    def _add_occupant(self, position):
        index = Board._translate_coordinates_to_index(position, self._size)

        return tuple(
            occupant if i != index else self._player_turn
            for i, occupant in enumerate(self._board_state)
        )

    def _board_search_start_side(self, player):
        if player == 1:
            return product(range(self._size), [0])
        if player == 2:
            return product([0], range(self._size))

        raise ValueError("Invalid player")

    def _board_search_end_side(self, player):
        if player == 1:
            return product(range(self._size), [self._size - 1])
        if player == 2:
            return product([self._size - 1], range(self._size))

        raise ValueError("Invalid player")

    def _get_neighbors(self, position, player):
        return set(
            filter(
                lambda pos: Board._are_positions_adjacent(position, pos)
                and self._is_position_occupied(pos, player),
                Board._get_valid_positions(self._size),
            )
        )

    def _board_search(self, position, player, visited):
        if position in self._board_search_end_side(player=player):
            return True

        return any(
            [
                self._board_search(neighbor, player, visited | {neighbor})
                for neighbor in self._get_neighbors(position, player) - visited
            ]
        )

    def get_possible_moves(self):
        def vec2tuples(acc, index):
            if self._board_state[index]:
                return acc
            return [*acc, Board._translate_index_to_coordinates(index, self._size)]

        return reduce(vec2tuples, range(self._size ** 2), [])

    def make_move(self, position):
        if self._is_finished():
            raise Exception(f"Game is finished")

        if self._is_position_occupied(position):
            raise ValueError(f"Position {position} is already occupied.")

        return Board(
            size=self._size,
            _player_turn=self._get_next_player_turn(self._player_turn),
            _board_state=self._add_occupant(position),
        )

    def get_tuple_representation(self):
        return (self._get_next_player_turn(self._player_turn),) + tuple(
            self._board_state
        )

    def _is_finished(self):
        return self.is_finished()[0]

    def is_finished(self):
        for player in (1, 2):
            for position in self._board_search_start_side(player):
                if self._is_position_occupied(position, player):
                    if self._board_search(position, player=player, visited={position}):
                        return True, player

        return False, None

    def get_edge_list(self):
        """
        Returns the board as an edge list.
        """

        def is_valid_edge(edge) -> bool:
            position_a, position_b = edge

            if position_a == position_b:
                return False

            if position_b < position_a:
                return False

            try:
                return Board._are_positions_adjacent(position_a, position_b)
            except:
                return False

        return filter(
            is_valid_edge,
            product(
                Board._get_valid_positions(self._size),
                Board._get_valid_positions(self._size),
            ),
        )

    def get_all_holes(self):
        """
        Returns a list of all hole positions.
        """
        return list(
            filter(
                lambda pos: not self._is_position_occupied(pos),
                Board._get_valid_positions(self._size),
            )
        )

    def get_all_pieces(self):
        """
        Returns a list of all piece positions.
        """
        return tuple(
            list(
                filter(
                    lambda pos: self._is_position_occupied(pos, player),
                    Board._get_valid_positions(self._size),
                )
            )
            for player in (1, 2)
        )
