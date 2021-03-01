import random


class Nim:
    def __init__(self, n, k):
        self._n = n
        self._k = k

    def get_possible_moves(self):
        return list(range(1, min(self._n, self._k) + 1))

    def take_rocks(self, number):
        if self.is_finished():
            raise Exception

        if number > self._k or number > self._n:
            raise ValueError("Cannot take that many rocks")

        return Nim(n=self._n - number, k=self._k)

    def opponent_take_rocks(self):
        if self.is_finished():
            raise Exception

        if self._k <= self._n:
            return Nim(n=0, k=self._k)

        return Nim(n=self._n - random.randint(1, self._k), k=self._k)

    def is_finished(self):
        return self._n == 0

    def __eq__(self, other):
        return self._n == other._n and self._k == other._k
