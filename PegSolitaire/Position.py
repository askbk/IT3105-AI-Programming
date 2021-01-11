from __future__ import annotations


class Position:
    """
    Represents a position on a Board.
    """

    def __init__(self, row_col):
        row, col = row_col
        if row < 0 or col < 0:
            raise ValueError

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

    def straight_distance(self, other: Position):
        """
        Calculates the distance to other, assuming it is on the same row, column or diagonal.
        """
        if self.is_on_same_column(other) or self.is_on_same_row(other):
            return abs(self._row - other._row) + abs(self._col - other._col)

        if not self.is_on_same_diagonal(other):
            raise ValueError("there is no straight line between the positions")

        return abs(self._row - other._row)

    def get_middle_position(self, other: Position):
        """
        Finds the position between this and other.
        """
        if self.straight_distance(other) != 2:
            raise ValueError

        row = self._row + (other._row - self._row) // 2
        col = self._col + (other._col - self._col) // 2

        return Position((row, col))

    def get_coordinates(self):
        return tuple((self._row, self._col))

    def __eq__(self, other: Position):
        return self._row == other._row and self._col == other._col

    def __hash__(self):
        return hash((self._row, self._col))

    def __lt__(self, other: Position):
        if self._row < other._row:
            return True

        if self._row == other._row and self._col < other._col:
            return True

        return False

    def __repr__(self):
        try:
            return f"Position(({self._row}, {self._col}))"
        except:
            return "Position()"
