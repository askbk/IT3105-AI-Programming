class Board:
    def __init__(self, hole_count=1, size=4):
        if hole_count < 1:
            raise ValueError("hole_count must be at least 1")
        if size < 4:
            raise ValueError("size must be at least 4")