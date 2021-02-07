import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer
from ACMAgent import ACMAgent
from Player import Player

from sklearn.utils.fixes import loguniform
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator

from Board import diamond_4, triangle_5


def make_param_grid(dictionary):
    return {key: [value] for key, value in dictionary.items()}


def custom_scoring(remaining):
    return np.median(remaining[-30:])


def tune_params(param_grid):
    best_median = 100000
    best_mean = 10000
    best_params = {}
    for g in ParameterGrid(param_grid):
        remaining = Player(ACMAgent(**g)).play_multiple_episodes(triangle_5, 400)
        median = np.median(remaining[-30:])
        mean = np.mean(remaining[-30:])
        if median < best_median:
            print(median, mean, g)
            best_median = median
            best_mean = mean
            best_params = g
        elif median == best_median and mean < best_mean:
            print(median, mean, g)
            best_median = median
            best_mean = mean
            best_params = g

    return best_params


if __name__ == "__main__":
    actor_params = {
        "actor_discount_factor": [0.6, 0.7, 0.8],
        "actor_eligibility_decay_rate": [0.5, 0.6, 0.7, 0.8],
        "actor_learning_rate": [0.001, 0.01, 0.1],
        "initial_epsilon": [1],
        "epsilon_decay_rate": [0.9, 0.99],
    }
    critic_params = {
        "critic_learning_rate": [
            0.0001,
            0.001,
            0.01,
        ],
        "critic_discount_factor": [0.4, 0.5, 0.6, 0.7, 0.8],
        "critic_eligibility_decay_rate": [0.4, 0.5, 0.6, 0.7, 0.8],
    }

    round1_actor_params = make_param_grid(tune_params(actor_params))
    print("round 1 actor params:", round1_actor_params)
    round1_critic_params = make_param_grid(
        tune_params(critic_params | round1_actor_params)
    )
    print("round 1 critic params:", round1_critic_params)
    round2_params = make_param_grid(tune_params(round1_critic_params | actor_params))
    print("best:", round2_params)