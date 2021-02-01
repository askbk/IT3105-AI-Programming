from Position import Position
from Board import Board


class Player:
    """
    A player class for playing board games with reinforcement learning agents.
    """

    def __init__(self, agent):
        self._agent = agent

    def play_single_episode(self, start_state):
        """
        Play a single episode of a game.
        """
        boards = [start_state]
        actions = []
        while not boards[-1].is_game_finished():
            possible_actions = boards[-1].get_possible_moves()
            actions.append(self._agent.choose_action(boards[-1], possible_actions))
            boards.append(boards[-1].make_move(actions[-1]))

        remaining = boards[-1].get_game_score()
        self._agent.end_state_reached(boards[-1], reward=1 / remaining ** 2)
        return remaining

    def play_multiple_episodes(self, episode_count):
        """
        Play multiple episodes of a game.
        """
        start_board = Board(shape="diamond", size=4, hole_positions=[Position((1, 2))])
        return list(
            map(lambda _: self.play_single_episode(start_board), range(episode_count))
        )
