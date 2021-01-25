from typing import List, Tuple, Set
from itertools import product, accumulate, chain
from functools import reduce
from Position import Position


class Board:
    def __init__(
        self,
        size=4,
        shape="diamond",
        hole_positions=[Position((1, 1))],
    ):
        if len(hole_positions) < 1:
            raise ValueError("hole_count must be at least 1.")
        if size < 3:
            raise ValueError("size must be at least 3.")
        if shape != "triangle" and shape != "diamond":
            raise ValueError("shape must be either 'triangle' or 'diamond'.")

        if any(
            map(
                lambda position: not Board._is_position_valid(shape, size, position),
                hole_positions,
            )
        ):
            raise ValueError("One or more invalid positions given in hole_positions.")

        if len(set(hole_positions)) != len(hole_positions):
            raise ValueError("One or more duplicate positions in hole_positions.")

        self._size = size
        self._shape = shape
        self._hole_positions = set(hole_positions)

    @staticmethod
    def _is_position_valid(
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
        if board_shape == "triangle" and not row + col < board_size:
            return False

        return True

    def _get_occupied_positions(self) -> Set[Position]:
        """
        Returns list of occupied positions.
        """
        return self._get_all_valid_positions() - self._get_holes()

    def _get_holes(self) -> Set[Position]:
        """
        Returns list of holes.
        """
        return set(self._hole_positions)

    def _is_position_occupied(self, position: Position) -> bool:
        """
        Determines if a position is occupied by a piece.
        """
        return position in self._get_occupied_positions()

    def _get_all_valid_positions(self) -> Set[Position]:
        """
        Returns a list of all valid positions on the game board.
        """
        return set(
            filter(
                lambda position: Board._is_position_valid(
                    self._shape, self._size, position
                ),
                map(Position, product(range(self._size), range(self._size))),
            )
        )

    def _can_move_to_hole(self, piece: Position, hole: Position) -> bool:
        """
        Determines whether a piece can move into a given hole.
        """
        try:
            if not piece.straight_distance(hole) == 2:
                return False

            return self._is_position_occupied(piece) and self._is_position_occupied(
                piece.get_middle_position(hole)
            )
        except ValueError:
            return False

    def get_possible_moves(self) -> List[Tuple[Position]]:
        """
        Get currently possible moves.
        """
        return list(
            chain.from_iterable(
                map(
                    lambda position: reduce(
                        lambda moves, hole: moves | set([(position, hole)])
                        if self._can_move_to_hole(position, hole)
                        else moves,
                        self._get_holes(),
                        set(),
                    ),
                    self._get_all_valid_positions(),
                )
            )
        )

    def make_move(self, move: Tuple[Position]):
        """
        Returns a new Board where the given move has been made.
        """
        from_position, to_position = move

        if not Board._is_position_valid(self._shape, self._size, from_position):
            raise ValueError(f"Invalid board position: {from_position}")

        if not Board._is_position_valid(self._shape, self._size, to_position):
            raise ValueError(f"Invalid board position: {to_position}")

        new_hole_positions = (self._get_holes() - set([to_position])) | set(
            [
                from_position.get_middle_position(to_position),
                from_position,
            ]
        )

        return Board(
            size=self._size,
            shape=self._shape,
            hole_positions=new_hole_positions,
        )

    def is_game_finished(self):
        """
        Determines whether the game is finished, in essence if any further moves are possible.
        """
        return len(self.get_possible_moves()) == 0

    def get_game_score(self):
        """
        Scores the game when it is finished.
        """
        if not self.is_game_finished():
            raise RuntimeError("Cannot calculate score until game is finished")

        return len(self._get_occupied_positions())

    def get_edge_list(self) -> List[Tuple[Position]]:
        """
        Returns the board as an edge list.
        """

        def is_valid_edge(edge: Tuple[Position]) -> bool:
            position_a, position_b = edge

            if position_a == position_b:
                return False

            if position_b < position_a:
                return False

            try:
                return position_a.straight_distance(position_b) == 1
            except:
                return False

        return filter(
            is_valid_edge,
            product(self._get_all_valid_positions(), self._get_all_valid_positions()),
        )

    def get_all_holes(self) -> List[Position]:
        """
        Returns a list of all hole positions.
        """
        return list(self._get_holes())

    def get_all_pieces(self) -> List[Position]:
        """
        Returns a list of all piece positions.
        """
        return list(self._get_occupied_positions())

    def __repr__(self) -> str:
        return f"<Board {sorted(list(self._get_holes()))}>"

    def __hash__(self):
        return hash(self.__repr__())