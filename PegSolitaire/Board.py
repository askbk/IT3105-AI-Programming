class Board:
    def __init__(self, hole_count=1, size=4, shape="triangle"):
        if hole_count < 1:
            raise ValueError("hole_count must be at least 1")
        if size < 4:
            raise ValueError("size must be at least 4")
        if shape != "triangle" and shape != "diamond":
            raise ValueError("shape must be either 'triangle' or 'diamond'")