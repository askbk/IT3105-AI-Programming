import json
from Hex.Game import Board
from Hex.Player import Player

if __name__ == "__main__":
    with open("./config.json", mode="r") as f:
        config = json.loads(f.read())
    play_config = config.get("simple_playthrough")
    Player.from_config(
        config, game=Board(size=play_config.get("board_size"))
    ).play_episodes(
        play_config.get("episodes"),
        display_board=play_config.get("display_board"),
        time_interval=play_config.get("time_interval"),
    )
