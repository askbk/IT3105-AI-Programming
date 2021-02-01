from __future__ import annotations
from typing import Tuple


class Position:
    """
    Represents a position on a Board.
    """

    def __init__(self, row_col: Tuple[int]):
        row, col = row_col
        if row < 0 or col < 0:
            raise ValueError("Row and column must be non-negative integers.")

        self._row = row
        self._col = col

    def is_on_same_row(self, other: Position) -> bool:
        """
        Determines if this position is on the same Board row as other.
        """
        return self._row == other._row

    def is_on_same_column(self, other: Position) -> bool:
        """
        Determines if this position is on the same Board column as other.
        """
        return self._col == other._col

    def is_on_same_diagonal(self, other: Position) -> bool:
        """
        Determines if this position is on the same Board diagonal as other.
        """
        return self._row + self._col == other._row + other._col

    def straight_distance(self, other: Position) -> int:
        """
        Calculates the distance to other, assuming it is on the same row, column or diagonal.
        """
        if self.is_on_same_column(other) or self.is_on_same_diagonal(other):
            return abs(self._row - other._row)

        if self.is_on_same_row(other):
            return abs(self._col - other._col)

        raise ValueError(f"There is no straight line between {self} and {other}.")

    def get_middle_position(self, other: Position) -> Position:
        """
        Finds the position between this and other.
        """
        if self.straight_distance(other) != 2:
            raise ValueError(f"Distance between {self} and {other} must be 2.")

        row = self._row + (other._row - self._row) // 2
        col = self._col + (other._col - self._col) // 2

        return Position((row, col))

    def get_coordinates(self) -> Tuple[int]:
        return tuple((self._row, self._col))

    def __eq__(self, other: Position) -> bool:
        return self._row == other._row and self._col == other._col

    def __hash__(self):
        return hash((self._row, self._col))

    def __lt__(self, other: Position) -> bool:
        if self._row < other._row:
            return True

        if self._row == other._row and self._col < other._col:
            return True

        return False

    def __repr__(self) -> str:
        try:
            return f"Position(({self._row}, {self._col}))"
        except:
            return "Position()"
