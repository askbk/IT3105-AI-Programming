import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer
from Agent.ACMAgent import ACMAgent
from Player import Player

from sklearn.utils.fixes import loguniform
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator

from Game.Board import diamond_4, triangle_5


def make_param_grid(dictionary):
    return {key: [value] for key, value in dictionary.items()}


def make_dict(param_grid):
    return {key: value[0] for key, value in param_grid.items()}


def tune_params(param_grid, board):
    convergence_list = list()
    best_median = 100000
    best_mean = 10000
    best_params = {}
    grid = ParameterGrid(param_grid)
    print(f"testing {len(grid)} param variations")
    for g in grid:
        remaining = Player(ACMAgent(**g)).play_multiple_episodes(board, 200)
        median = np.median(remaining[-30:])
        mean = np.mean(remaining[-30:])
        if median <= best_median and mean < best_mean:
            print(median, mean, g)
            best_median = median
            best_mean = mean
            best_params = g
        if median == 1:
            convergence_list.append(g)
    print(f"reached convergence: {convergence_list}")
    return best_params


def filter_dict_keys(dictionary, keys):
    return {key: dictionary[key] for key in keys}


def tuning_round(params, param_keys, board):
    return filter_dict_keys(make_param_grid(tune_params(params, board)), param_keys)


def validate(params, board):
    return np.mean(Player(ACMAgent(**params)).play_multiple_episodes(board, 500)[-30:])


def tune_params_for_board(board, rounds=2):
    actor_keys = [
        "actor_discount_factor",
        "actor_eligibility_decay_rate",
        "actor_learning_rate",
        "initial_epsilon",
        "epsilon_decay_rate",
    ]
    critic_keys = [
        "critic_learning_rate",
        "critic_discount_factor",
        "critic_eligibility_decay_rate",
    ]
    actor_params = {
        "actor_discount_factor": [1],
        "actor_eligibility_decay_rate": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "actor_learning_rate": [0.00001, 0.0001, 0.001, 0.01],
        "initial_epsilon": [1],
        "epsilon_decay_rate": [0.995],
    }
    critic_params = {
        "critic_learning_rate": [0.00001, 0.0001, 0.001, 0.01],
        "critic_discount_factor": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "critic_eligibility_decay_rate": [0.4, 0.5, 0.6, 0.7, 0.8],
    }
    best = tune_params(
        {
            "actor_discount_factor": [0.9],
            "actor_eligibility_decay_rate": [0.9],
            "actor_learning_rate": [1e-2, 1e-3, 1e-4, 1e-5],
            "critic_function": ["neural_network"],
            "critic_nn_dimensions": [(15, 8, 1), (15, 15, 8, 1)],
            "critic_discount_factor": [0.5],
            "critic_eligibility_decay_rate": [0.5],
            "critic_learning_rate": [1e-3, 1e-4, 1e-5, 1e-6],
            "epsilon_decay_rate": [0.9],
            "initial_epsilon": [1],
        },
        board,
    )
    print("best:", best)
    return best
    # round_best = dict()
    # round_best_score = validate(round_best, board)
    # for i in range(rounds):
    #     print(f"----Round {i}----")
    #     round_actor_params = tuning_round(round_best | actor_params, actor_keys, board)
    #     round_critic_params = tuning_round(
    #         critic_params | round_actor_params, critic_keys, board
    #     )
    #     temp = round_actor_params | round_critic_params
    #     score = validate(make_dict(temp), board)
    #     print("avg of 100", score)
    #     if score < round_best_score:
    #         round_best = temp
    #         round_best_score = score
    #     elif score > round_best_score:
    #         break

    # print("best:", round_best)
    # return round_best

    # round1_actor_params = tuning_round(actor_params, actor_keys, board)
    # print("round 1 actor params:", round1_actor_params)
    # round1_critic_params = tuning_round(
    #     critic_params | round1_actor_params, critic_keys, board
    # )
    # print("round 1 critic params:", round1_critic_params)
    # round2_params = tuning_round(
    #     round1_critic_params | round1_actor_params, [*actor_keys, *critic_keys], board
    # )
    # print("best:", round2_params)


if __name__ == "__main__":
    # tune_params()
    tune_params_for_board(triangle_5, 5)
