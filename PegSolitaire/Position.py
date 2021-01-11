class Position:
    """
    Represents a position on a Board.
    """

    def __init__(self, row, col):
        if row < 0 or col < 0:
            raise ValueError

        self._row = row
        self._col = col

    def is_on_same_row(self, other):
        """
        Determines if this position is on the same Board row as other.
        """
        return self._row == other._row

    def is_on_same_column(self, other):
        """
        Determines if this position is on the same Board column as other.
        """
        return self._col == other._col

    def is_on_same_diagonal(self, other):
        """
        Determines if this position is on the same Board diagonal as other.
        """
        return self._row + self._col == other._row + other._col
