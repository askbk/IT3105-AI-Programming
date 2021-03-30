from Hex.Game import Board
from Hex.Player import Player

if __name__ == "__main__":
    Player(Board(size=3)).play_episodes(10, display_board=False, time_interval=1)