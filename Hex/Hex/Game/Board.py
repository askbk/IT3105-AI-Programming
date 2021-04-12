from __future__ import annotations
from itertools import product
from functools import reduce, cache
from Hex.Game.GameBase import GameBase
from typing import Optional, Tuple
import Hex.Types as Types

Position = Tuple[int, int]


class Board(GameBase):
    """
    Hex board
    """

    def __init__(self, size=4, _board_state=None, _player_turn: Optional[int] = None):
        self._size = size
        self._board_state = (0,) * size ** 2 if _board_state is None else _board_state
        self._player_turn = 1 if _player_turn is None else _player_turn
        is_finished, winner = self._is_end_state_get_winner()
        self._is_finished = is_finished
        self._winner = winner

    @staticmethod
    @cache
    def _get_valid_positions(size: int):
        return set(product(range(size), range(size)))

    @staticmethod
    @cache
    def _are_positions_adjacent(position_a: Position, position_b: Position) -> bool:
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
    def _get_next_player_turn(current_player_turn: int) -> int:
        return 3 - current_player_turn

    def index_to_action(self, index: int) -> Types.Action:
        return Board._translate_index_to_coordinates(index, self._size)

    def action_to_index(self, action: Position) -> int:
        return Board._translate_coordinates_to_index(action, self._size)

    def get_state_size(self) -> int:
        return self._size ** 2 + 1

    def get_action_space_size(self) -> int:
        return self._size ** 2

    @staticmethod
    def _translate_index_to_coordinates(index: int, board_size: int) -> Position:
        return (index // board_size, index % board_size)

    @staticmethod
    def _translate_coordinates_to_index(coordinates: Position, board_size: int) -> int:
        return board_size * coordinates[0] + coordinates[1]

    def _is_position_occupied(
        self, position: Position, player: Optional[int] = None
    ) -> bool:
        position_index = Board._translate_coordinates_to_index(position, self._size)
        if player is None:
            return self._board_state[position_index] != 0

        return self._board_state[position_index] == player

    def _add_occupant(self, position: Position):
        index = Board._translate_coordinates_to_index(position, self._size)

        return tuple(
            occupant if i != index else self._player_turn
            for i, occupant in enumerate(self._board_state)
        )

    @staticmethod
    @cache
    def _board_search_start_side(player: int, board_size: int):
        if player == 1:
            return list(product([0], range(board_size)))
        if player == 2:
            return list(product(range(board_size), [0]))

        raise ValueError("Invalid player")

    @staticmethod
    @cache
    def _board_search_end_side(player: int, board_size: int):
        if player == 1:
            return list(product([board_size - 1], range(board_size)))
        if player == 2:
            return list(product(range(board_size), [board_size - 1]))

        raise ValueError("Invalid player")

    def _get_neighbors(self, position: Position, player: int):
        return set(
            filter(
                lambda pos: Board._are_positions_adjacent(position, pos)
                and self._is_position_occupied(pos, player),
                Board._get_valid_positions(self._size),
            )
        )

    def _board_search(self, position: Position, player: int, visited: set) -> bool:
        if position in Board._board_search_end_side(
            player=player, board_size=self._size
        ):
            return True

        return any(
            self._board_search(neighbor, player, visited | {neighbor})
            for neighbor in self._get_neighbors(position, player) - visited
        )

    def get_possible_actions(self):
        def vec2tuples(acc, index):
            if self._board_state[index]:
                return acc
            return [*acc, Board._translate_index_to_coordinates(index, self._size)]

        return reduce(vec2tuples, range(self._size ** 2), [])

    def perform_action(self, position: Position) -> Board:
        if self._is_finished:
            raise Exception(f"Game is finished")

        if self._is_position_occupied(position):
            raise ValueError(f"Position {position} is already occupied.")

        return Board(
            size=self._size,
            _player_turn=self._get_next_player_turn(self._player_turn),
            _board_state=self._add_occupant(position),
        )

    def get_tuple_representation(self):
        return (self._player_turn,) + tuple(self._board_state)

    def _is_end_state_get_winner(self) -> Tuple[bool, Optional[int]]:
        for player in (1, 2):
            for position in Board._board_search_start_side(player, self._size):
                if self._is_position_occupied(position, player):
                    if self._board_search(position, player=player, visited={position}):
                        return True, player

        return False, None

    def is_finished(self):
        return self._is_finished, self._winner

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

    def is_end_state_reached(self) -> bool:
        return self._is_finished

    def get_player_turn(self) -> int:
        return self._player_turn

    def __eq__(self, other: Board) -> bool:
        return (
            self._size == other._size
            and self._player_turn == other._player_turn
            and self._board_state == other._board_state
        )

    def __repr__(self):
        return f"Board<size={self._size}, board_state={self._board_state}, player_turn={self._player_turn}>"
