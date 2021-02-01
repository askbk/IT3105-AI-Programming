from RandomAgent import RandomAgent
from ACMAgent import ACMAgent
from Board import Board
from Position import Position
from PerformanceDisplay import graph_performance
from Player import Player


if __name__ == "__main__":
    agent = ACMAgent(
        actor_discount_factor=0.9,
        actor_eligibility_decay_rate=0.8,
        actor_learning_rate=0.05,
        critic_function="table",
        critic_learning_rate=0.1,
        critic_discount_factor=0.9,
        critic_eligibility_decay_rate=0.9,
        initial_epsilon=1,
        epsilon_decay_rate=0.95,
    )
    player1 = Player(agent)
    # player2 = Player(RandomAgent())
    # small_triangle_board = Board(
    #     size=4, shape="triangle", hole_positions=[Position((2, 0))]
    # )
    # small_diamond_board = Board(
    #     size=3, shape="diamond", hole_positions=[Position((2, 0))]
    # )
    # medium_triangle_board = Board(size=5)
    # large_triangle_board = Board(size=8)
    # large_diamond_board = Board(size=6)
    remaining1 = player1.play_multiple_episodes(200)
    print(len(player1._agent._actor._policy))
    # graph_performance(remaining1)