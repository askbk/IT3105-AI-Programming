import random
import pickle
import time
import os.path as path
from Hex.MCTS import MCTS
from Hex.Game import Board
from Hex.Player import Player
from Hex.GreedyNNAgent import GreedyNNAgent


class RandomMCTSAgent:
    def __init__(
        self,
        initial_state,
        epsilon: float = 0,
        _replay_buffer=None,
        _mcts=None,
    ):
        self._replay_buffer = {} if _replay_buffer is None else _replay_buffer
        self._epsilon = epsilon
        self._mcts = RandomMCTSAgent._initialize_mcts(initial_state, _mcts)
        self._initial_state = initial_state

    @staticmethod
    def _initialize_mcts(initial_state, mcts):
        if mcts is not None:
            return mcts

        return MCTS(
            initial_state=initial_state, search_games=1000, time_limit=2
        ).search()

    def get_action(self):
        if random.random() < self._epsilon:
            return random.choice(self._initial_state.get_possible_actions())
        return self._mcts.get_best_action()

    def next_state(self, next_state):
        action_visit_count_distribution = self._mcts.get_root_distribution()
        action_distribution = [0] * self._initial_state.get_action_space_size()
        total_visit = sum(
            visit_count for _, visit_count in action_visit_count_distribution
        )
        for action, visit_count in action_visit_count_distribution:
            action_distribution[self._initial_state.action_to_index(action)] = (
                visit_count / total_visit
            )
        replay_buffer = {
            **self._replay_buffer,
            self._initial_state.get_tuple_representation(): action_distribution,
        }
        return RandomMCTSAgent(
            initial_state=next_state,
            _replay_buffer=replay_buffer,
            _mcts=self._mcts.update_root(next_state).search(),
        )

    def end_of_episode_update(self, initial_state):
        return RandomMCTSAgent(
            initial_state=initial_state, _replay_buffer=self._replay_buffer
        )


def generate_synthetic_training_data(n_test_cases, game, time_limit=None):
    save_path = f"./replay_buffers/size_{game._size}"
    replay_buffer = (
        load_synthetic_training_data(game._size) if path.exists(save_path) else {}
    )
    agent = RandomMCTSAgent(initial_state=game, epsilon=0.5)
    player = Player(game, _agent=agent)
    player._agent._replay_buffer = replay_buffer
    start_time = time.time()
    while len(player._agent._replay_buffer) < n_test_cases:
        if time_limit is not None and time.time() - start_time > time_limit:
            break
        player.play_episodes(episode_count=1, display_board=False, silent=True)
        print(f"{len(player._agent._replay_buffer)}/{n_test_cases}")

    mode = "b+w" if path.exists(save_path) else "x+b"

    with open(save_path, mode=mode) as f:
        pickle.dump(player._agent._replay_buffer, f)


def load_synthetic_training_data(size):
    with open(f"./replay_buffers/size_{size}", mode="r+b") as f:
        return pickle.load(f)


def train_nn_on_data(size):
    data = load_synthetic_training_data(size).items()
    print("loaded", len(data), "saved training cases")
    agent = GreedyNNAgent.from_training_data(
        {
            "actor": {
                "learning_rate": 0.01,
                "optimizer": "adam",
                "output_activation": "sigmoid",
                "loss": "mean_squared_error",
                "layers": [
                    {"units": 200, "activation": "relu"},
                    {"units": 50, "activation": "relu"},
                ],
            }
        },
        Board(size=size),
        data,
        {"batch_size": 64, "epochs": 50, "validation_split": 0.15},
    )
    agent._actor._nn.save(f"./models/size_{size}")


if __name__ == "__main__":
    size = 5
    generate_synthetic_training_data(100000, Board(size=size), time_limit=3600 * 3.5)
    # train_nn_on_data(size)
