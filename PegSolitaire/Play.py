from RandomAgent import RandomAgent
from ACMAgent import ACMAgent
from Board import Board
from Position import Position
from PerformanceDisplay import graph_performance

# from BoardDisplay import BoardDisplay


def play_many_episodes(agent, board, episode_count):
    def play_one_episode(start_board):
        boards = [start_board]
        moves = []
        while not boards[-1].is_game_finished():
            possible_actions = boards[-1].get_possible_moves()
            moves.append(agent.choose_action(boards[-1], possible_actions))
            boards.append(boards[-1].make_move(moves[-1]))

        remaining = boards[-1].get_game_score()
        agent.end_state_reached(boards[-1], reward=board._size - remaining)
        return remaining
        # BoardDisplay.display_board_move_sequence(boards[1:], moves, 4)

    return list(map(lambda _: play_one_episode(board), range(episode_count)))


if __name__ == "__main__":
    agent = ACMAgent(
        actor_discount_factor=0.9,
        actor_eligibility_decay_rate=0.7,
        actor_learning_rate=0.1,
        critic_function="table",
        critic_learning_rate=0.1,
        critic_eligibility_decay_rate=0.7,
        critic_discount_factor=0.9,
        initial_epsilon=0.9,
        epsilon_decay_rate=0.99,
    )
    small_triangle_board = Board(
        size=4, shape="triangle", hole_positions=[Position((2, 0))]
    )
    small_diamond_board = Board(
        size=3, shape="diamond", hole_positions=[Position((2, 0))]
    )
    medium_triangle_board = Board(size=5)
    large_triangle_board = Board(size=8)
    large_diamond_board = Board(size=6)
    remaining = play_many_episodes(agent, medium_triangle_board, 50)
    # graph_performance(remaining)