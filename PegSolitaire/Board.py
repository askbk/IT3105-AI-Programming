from typing import List, Tuple, TYPE_CHECKING
from itertools import product, accumulate

Position = Tuple[int]


class Board:
    def __init__(self, hole_count=1, size=4, shape="diamond", hole_positions=[(1, 1)]):
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

    @staticmethod
    def _position_is_valid(
        board_shape: str, board_size: int, position: Position
    ) -> bool:
        """
        Determines whether a position is valid for a given board shape and size.
        """
        row, col = position
        if not 0 <= row < board_size:
            return False
        if not 0 <= col < board_size:
            return False
        if board_shape == "triangle" and not row + col <= board_size:
            return False

        return True

    def _get_all_valid_positions(self) -> List[Position]:
        """
        Returns a list of all valid positions on the game board.
        """
        return list(
            filter(
                lambda position: Board._position_is_valid(
                    self._shape, self._size, position
                ),
                product(range(self._size), range(self._size)),
            )
        )

    def _are_on_same_row(self, position_a: Position, position_b: Position) -> bool:
        return position_a[0] == position_b[0]

    def _are_on_same_column(self, position_a: Position, position_b: Position) -> bool:
        return position_a[1] == position_b[1]

    def _are_on_same_diagonal(self, position_a: Position, position_b: Position) -> bool:
        return position_a[0] + position_a[1] == position_b[0] + position_b[1]

    def _straight_distance(self, position_a, position_b):
        if self._are_on_same_column(position_a, position_b) or self._are_on_same_row(
            position_a, position_b
        ):
            return abs(position_a[0] - position_b[0]) + abs(
                position_a[1] - position_b[1]
            )

        if not self._are_on_same_diagonal(position_a, position_b):
            raise ValueError("there is no straight line between the positions")

        return abs(position_a[0] - position_b[0])

    def _can_move_to_hole(
        self, piece_position: Position, hole_position: Position
    ) -> bool:
        """
        Determines whether a piece can move into a given hole.
        """
        if not (
            self._are_on_same_column(piece_position, hole_position)
            or self._are_on_same_row(piece_position, hole_position)
            or self._are_on_same_diagonal(piece_position, hole_position)
        ):
            return False

        result = self._straight_distance(piece_position, hole_position) == 2

        if result:
            print(piece_position, "can move to", hole_position)

        return result

    def get_movable_pieces(self) -> List[Position]:
        """
        Get currently movable pieces as a list of tuples of their positions.
        """
        return list(
            filter(
                lambda position: any(
                    map(
                        lambda hole_position: self._can_move_to_hole(
                            position, hole_position
                        ),
                        self._hole_positions,
                    )
                ),
                self._get_all_valid_positions(),
            )
        )
