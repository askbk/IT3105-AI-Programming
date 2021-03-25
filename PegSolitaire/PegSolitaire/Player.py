from PegSolitaire.Game import Position, Board, BoardDisplay


class Player:
    """
    A player class for playing board games with reinforcement learning agents.
    """

    def __init__(self, agent):
        self._agent = agent

    def _get_reward(self, remaining):
        return 10 / remaining - 2.5 * remaining

    def play_single_episode(self, board, replay=False, replay_interval=1):
        """
        Play a single episode of a game.
        """
        boards = [board]
        actions = []
        while not boards[-1].is_game_finished():
            possible_actions = boards[-1].get_possible_moves()
            actions.append(
                self._agent.choose_action(
                    boards[-1], possible_actions, epsilon_greedy=(not replay)
                )
            )
            boards.append(boards[-1].make_move(actions[-1]))

        remaining = boards[-1].get_game_score()
        self._agent.end_state_reached(boards[-1], reward=self._get_reward(remaining))
        if replay:
            BoardDisplay.display_board_move_sequence(
                boards[1:], actions, pause=replay_interval
            )
        return remaining

    def play_multiple_episodes(self, board, episode_count, replay_after_training=False):
        """
        Play multiple episodes of a game.
        """
        remaining = list(
            map(lambda _: self.play_single_episode(board), range(episode_count))
        )
        if replay_after_training:
            self.play_single_episode(board, replay=True, replay_interval=1)

        return remaining