from Hex.Game import Board
from Hex.Player import Player

if __name__ == "__main__":
    Player.from_config_path("config.json", game=Board(size=3)).play_episodes(5)
