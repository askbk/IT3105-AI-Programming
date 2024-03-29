import numpy as np
from PegSolitaire.Agent.ACMAgent import (
    ACMAgent,
    agent_diamond_4,
    agent_triangle_5,
    agent_nn_diamond_4,
    agent_nn_triangle_5,
)
from PegSolitaire.Game.Board import (
    diamond_3,
    diamond_4,
    diamond_5,
    triangle_3,
    triangle_4,
    triangle_5,
)
from PerformanceDisplay import graph_performance
from PegSolitaire.Player import Player

if __name__ == "__main__":
    # graph_performance(Player(agent_diamond_4).play_multiple_episodes(diamond_4, 100))
    # graph_performance(Player(agent_nn_diamond_4).play_multiple_episodes(diamond_4, 100))
    # graph_performance(Player(agent_triangle_5).play_multiple_episodes(triangle_5, 100))
    graph_performance(
        Player(agent_nn_triangle_5).play_multiple_episodes(triangle_5, 200)
    )
