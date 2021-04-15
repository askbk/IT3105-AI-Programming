import random
import json
from operator import itemgetter
from typing import Sequence, Tuple
from functools import reduce
from Hex.Player import Player
from Hex.Game import Board, GameBase
from Hex.AgentBase import AgentBase
from Hex.GreedyNNAgent import GreedyNNAgent
from Hex.Utils import while_loop


def train_progressive_policies(
    episodes: int, save_interval: int, game: GameBase
) -> Sequence[str]:
    """
    Save progressively more trained models with given interval.
    Returns file paths of saved models.
    """
    paths = []
    player = Player.from_config_path("./config.json", game)
    player.save_agent_nn(f"./models/TOPP/{game._size}/0")
    paths.append(f"./models/TOPP/{game._size}/0")
    for episode in range(save_interval, episodes + 1, save_interval):
        print(f"{episode}/{episodes}")
        player.play_episodes(
            episode_count=save_interval, display_board=False, silent=True
        )
        player.save_agent_nn(f"./models/TOPP/{game._size}/{episode}")
        paths.append(f"./models/TOPP/{game._size}/{episode}")

    return paths


def play_single_game(agent1: AgentBase, agent2: AgentBase, game: GameBase) -> AgentBase:
    def condition(state: Tuple[GameBase, AgentBase, AgentBase]) -> bool:
        game_state, *_ = state
        return not game_state.is_end_state_reached()

    def step(state: Tuple[GameBase, AgentBase, AgentBase]):
        game_state, current_agent, next_agent = state
        next_game_state = game_state.perform_action(current_agent.get_action())
        return (
            next_game_state,
            next_agent.next_state(next_game_state),
            current_agent.next_state(next_game_state),
        )

    initial = (game, agent1, agent2)

    end_state, *_ = while_loop(condition, initial, step)
    _, winner = end_state.is_finished()

    return agent1 if winner == 1 else agent2


def aggregate_results(raw_results, agents: Sequence[AgentBase]):
    def get_series_winner(series_results):
        (agent1, agent1_wins), (agent2, agent2_wins) = series_results
        if agent1_wins > agent2_wins:
            return agent1
        if agent2_wins > agent1_wins:
            return agent2
        return None

    def get_series_won_games(series_results, agent):
        (agent1, agent1_wins), (agent2, agent2_wins) = series_results
        if agent == agent1:
            return agent1_wins
        if agent == agent2:
            return agent2_wins
        return 0

    def aggregate_reducer(results, agent: AgentBase):
        won_series = sum(
            1 if get_series_winner(series_results) == agent.get_name() else 0
            for series_results in raw_results
        )
        won_games = sum(
            get_series_won_games(series_results, agent.get_name())
            for series_results in raw_results
        )
        return {**results, agent.get_name(): (won_games, won_series)}

    return reduce(aggregate_reducer, agents, dict())


def play_tournament(agents: Sequence[AgentBase], games_per_pair: int, game: GameBase):
    def play_series(agent_pair: Tuple[AgentBase, AgentBase]):
        agent1, agent2 = agent_pair
        series_results = [
            play_single_game(*random.sample(agent_pair, k=2), game)
            for _ in range(games_per_pair)
        ]
        return (agent1.get_name(), series_results.count(agent1)), (
            agent2.get_name(),
            series_results.count(agent2),
        )

    agent_pairs = [
        (agents[i], agents[j])
        for i in range(len(agents))
        for j in range(i + 1, len(agents))
    ]

    tournament_results = [play_series(agent_pair) for agent_pair in agent_pairs]
    aggregated = aggregate_results(tournament_results, agents)
    sorted_by_games = sorted(
        ((name, won[0]) for name, won in aggregated.items()),
        key=itemgetter(1),
        reverse=True,
    )
    sorted_by_series = sorted(
        ((name, won[1]) for name, won in aggregated.items()),
        key=itemgetter(1),
        reverse=True,
    )
    print("Sorted by games won:", *sorted_by_games, sep="\n")
    print("Sorted by series won:", *sorted_by_series, sep="\n")


def load_greedy_agents_from_saved(paths, game):
    return [GreedyNNAgent.from_saved_nn(path, game) for path in paths]


def train_and_play_tournament(
    episodes: int, save_interval: int, games_per_series: int, game: GameBase
):
    saved_nn_paths = train_progressive_policies(episodes, save_interval, game)

    agents = load_greedy_agents_from_saved(saved_nn_paths, game)

    play_tournament(agents, games_per_series, game)


if __name__ == "__main__":
    with open("./config.json", mode="r") as f:
        config = json.loads(f.read()).get("TOPP", {})
    game = Board(size=config.get("board_size", 4))
    play_tournament(
        load_greedy_agents_from_saved(
            [
                "./models/TOPP/4/0",
                "./models/TOPP/4/50",
                "./models/TOPP/4/100",
                "./models/TOPP/4/150",
                "./models/TOPP/4/200",
            ],
            game,
        ),
        config.get("games_per_series", 9),
        game,
    )
    # train_and_play_tournament(
    #     episodes=config.get("training_episodes", 200),
    #     save_interval=config.get("save_interval", 50),
    #     games_per_series=config.get("games_per_series", 9),
    #     game=Board(size=config.get("board_size", 4)),
    # )
