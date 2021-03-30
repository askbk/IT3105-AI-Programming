from __future__ import annotations
from warnings import simplefilter
from functools import reduce
import numpy as np
from Hex.Types import StateVector, ProbabilityDistribution, ReplayBuffer

simplefilter(action="ignore", category=DeprecationWarning)
import tensorflow.keras as keras


class Actor:
    def __init__(self, input_size: int, output_size: int):
        inputs = keras.layers.Input(shape=(input_size,))
        self._nn = keras.Model(
            inputs=inputs,
            outputs=reduce(
                lambda prev_layer, curr_layer: curr_layer(prev_layer),
                [
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dense(output_size, activation="softmax"),
                ],
                inputs,
            ),
        )

        self._nn.compile(optimizer="adam", loss="mean_squared_error")

    def rollout(self, state_vector: StateVector) -> np.array:
        return self._nn(np.atleast_2d(np.array(state_vector))).numpy().flatten()

    def train(self, replay_buffer: ReplayBuffer) -> Actor:
        root_states, probability_distributions = zip(*replay_buffer)
        self._nn.train_on_batch(
            x=np.atleast_2d(np.array(root_states)),
            y=np.atleast_2d(np.array(probability_distributions)),
        )
        return self