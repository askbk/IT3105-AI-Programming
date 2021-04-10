from typing import Sequence
from Hex.Player import Player


def train_progressive_policies(episodes: int, save_interval: int):
    """
    Save progressively more trained models with given interval.
    """
    pass


def play_tournament(players: Sequence[Player], games_per_pair: int):
    pass


def train_and_play_tournament(episodes: int, save_interval: int, games_per_pair: int):
    pass


if __name__ == "__main__":
    train_and_play_tournament(50, 5, 3)
