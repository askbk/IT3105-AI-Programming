import json
from Hex.Game import Board
from Hex.Player import Player


def load_config(path: str):
    with open(path, mode="r") as f:
        return json.loads(f.read())


if __name__ == "__main__":
    Player.from_config(load_config("config.json"), game=Board(size=3)).play_episodes(1)
    # Player(Board(size=3)).play_episodes(10, display_board=False, time_interval=1)