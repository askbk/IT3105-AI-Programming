from RandomAgent import RandomAgent
from ACMAgent import ACMAgent
from Board import Board, diamond_4, triangle_5
from Position import Position
from PerformanceDisplay import graph_performance
from Player import Player


diamond_4_params_table = {
    "actor_discount_factor": 0.9,
    "actor_eligibility_decay_rate": 0.9,
    "actor_learning_rate": 0.001,
    "critic_discount_factor": 0.5,
    "critic_eligibility_decay_rate": 0.5,
    "critic_learning_rate": 0.001,
    "epsilon_decay_rate": 0.9,
    "initial_epsilon": 1,
}

diamond_4_params_table_2 = {
    "actor_discount_factor": 0.7,
    "actor_eligibility_decay_rate": 0.8,
    "actor_learning_rate": 0.01,
    "critic_discount_factor": 0.4,
    "critic_eligibility_decay_rate": 0.4,
    "critic_learning_rate": 0.001,
    "epsilon_decay_rate": 0.96,
    "initial_epsilon": 1,
}


if __name__ == "__main__":
    agent = ACMAgent(**diamond_4_params_table_2)
    player = Player(agent)

    graph_performance(player.play_multiple_episodes(diamond_4, 100))
    # graph_performance(Player(ACMAgent(**a)).play_multiple_episodes(triangle_5, 400))
