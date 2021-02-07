from Position import Position
from Board import Board


class Player:
    """
    A player class for playing board games with reinforcement learning agents.
    """

    def __init__(self, agent):
        self._agent = agent

    def _get_reward(self, remaining):
        return 10 / remaining - remaining / 2

    def play_single_episode(self, board):
        """
        Play a single episode of a game.
        """
        boards = [board]
        actions = []
        while not boards[-1].is_game_finished():
            possible_actions = boards[-1].get_possible_moves()
            actions.append(self._agent.choose_action(boards[-1], possible_actions))
            boards.append(boards[-1].make_move(actions[-1]))

        remaining = boards[-1].get_game_score()
        self._agent.end_state_reached(boards[-1], reward=self._get_reward(remaining))
        return remaining

    def play_multiple_episodes(self, board, episode_count):
        """
        Play multiple episodes of a game.
        """
        return list(
            map(lambda _: self.play_single_episode(board), range(episode_count))
        )
