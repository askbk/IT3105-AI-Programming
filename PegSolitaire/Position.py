class Position:
    """
    Represents a position on a Board.
    """

    def __init__(self, row, col):
        if row < 0 or col < 0:
            raise ValueError

        self._row = row

    def is_on_same_row(self, other):
        """
        Determines if this position is on the same Board row as other.
        """
        return self._row == other._row