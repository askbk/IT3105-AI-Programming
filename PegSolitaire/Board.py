from typing import List, Tuple, TYPE_CHECKING
from itertools import product, accumulate, chain
from functools import reduce
from Position import Position


class Board:
    def __init__(
        self, hole_count=1, size=4, shape="diamond", hole_positions=[Position((1, 1))]
    ):
        if hole_count < 1:
            raise ValueError("hole_count must be at least 1")
        if size < 4:
            raise ValueError("size must be at least 4")
        if shape != "triangle" and shape != "diamond":
            raise ValueError("shape must be either 'triangle' or 'diamond'")

        if any(
            map(
                lambda position: not Board._position_is_valid(shape, size, position),
                hole_positions,
            )
        ):
            raise ValueError("one or more invalid hole positions")

        if len(set(hole_positions)) != len(hole_positions):
            raise ValueError("one or more duplicate hole positions")

        self._size = size
        self._shape = shape
        self._hole_positions = hole_positions
        self._occupied_positions = list(
            set(self._get_all_valid_positions()) - set(self._hole_positions)
        )

    @staticmethod
    def _position_is_valid(
        board_shape: str, board_size: int, position: Position
    ) -> bool:
        """
        Determines whether a position is valid for a given board shape and size.
        """
        row, col = position.get_coordinates()
        if not 0 <= row < board_size:
            return False
        if not 0 <= col < board_size:
            return False
        if board_shape == "triangle" and not row + col <= board_size:
            return False

        return True

    def _is_position_occupied(self, position: Position) -> bool:
        """
        Determines if a position is occupied by a piece.
        """
        return position in self._occupied_positions

    def _get_all_valid_positions(self) -> List[Position]:
        """
        Returns a list of all valid positions on the game board.
        """
        return list(
            filter(
                lambda position: Board._position_is_valid(
                    self._shape, self._size, position
                ),
                map(Position, product(range(self._size), range(self._size))),
            )
        )

    def _can_move_to_hole(self, piece: Position, hole: Position) -> bool:
        """
        Determines whether a piece can move into a given hole.
        """
        if not (
            piece.is_on_same_column(hole)
            or piece.is_on_same_row(hole)
            or piece.is_on_same_diagonal(hole)
        ):
            return False

        if not piece.straight_distance(hole) == 2:
            return False

        return self._is_position_occupied(piece.get_middle_position(hole))

    def get_possible_moves(self) -> List[Tuple[Position]]:
        """
        Get currently possible moves.
        """
        return list(
            chain.from_iterable(
                map(
                    lambda position: reduce(
                        lambda moves, hole: moves + [(position, hole)]
                        if self._can_move_to_hole(position, hole)
                        else moves,
                        self._hole_positions,
                        list(),
                    ),
                    self._get_all_valid_positions(),
                )
            )
        )