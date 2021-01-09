from typing import List, Tuple


class Board:
    def __init__(self, hole_count=1, size=4, shape="diamond", hole_positions=[(1, 1)]):
        if hole_count < 1:
            raise ValueError("hole_count must be at least 1")
        if size < 4:
            raise ValueError("size must be at least 4")
        if shape != "triangle" and shape != "diamond":
            raise ValueError("shape must be either 'triangle' or 'diamond'")

        for position in hole_positions:
            if not Board._position_is_valid(shape, size, position):
                raise ValueError

        self.size = size

    @staticmethod
    def _position_is_valid(board_shape, board_size, position):
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

    # def get_movable_pieces(self) -> List[Tuple[int]]:
    #     """
    #     Get currently movable pieces as a list of tuples of their positions.
    #     """
    #     result = []
    #     for row in range(self.size):
    #         for col in range(self.size):
    #             if can_move_to_hole((row, col)):
    #                 result += (row, col)

    #     return result
