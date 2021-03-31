from __future__ import annotations
from warnings import simplefilter
from typing import Dict
from functools import reduce
import numpy as np
from Hex.Types import StateVector, ProbabilityDistribution, ReplayBuffer

simplefilter(action="ignore", category=DeprecationWarning)
import tensorflow as tf
import tensorflow.keras as keras


class Actor:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        optimizer="adam",
        loss="mean_squared_error",
        learning_rate=0.001,
    ):
        self._input_size = input_size
        self._output_size = output_size
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

        self._nn.compile(
            optimizer=Actor._initialize_optimizer(
                name=optimizer, learning_rate=learning_rate
            ),
            loss=loss,
        )

    @staticmethod
    def _initialize_optimizer(name, learning_rate):
        if name == "adam":
            optimizer = keras.optimizers.Adam

        if name == "rms":
            optimizer = keras.optimizers.RMSprop

        if name == "adagrad":
            optimizer = keras.optimizers.Adagrad

        if name == "sgd":
            optimizer = keras.optimizers.SGD

        return optimizer(learning_rate=learning_rate)

    @staticmethod
    def from_config(input_size: int, output_size: int, config: Dict) -> Actor:
        return Actor(
            input_size=input_size, output_size=output_size, **config.get("actor", {})
        )

    def rollout(self, state_vector: StateVector) -> np.array:
        x = np.atleast_2d(np.array(state_vector))
        if x.shape != (1, self._input_size):
            raise ValueError(
                f"Input must be a vector of length {self._input_size}, length was {x.shape[1]}"
            )
        return self._nn(tf.convert_to_tensor(x)).numpy().flatten()

    def train(self, replay_buffer: ReplayBuffer) -> Actor:
        root_states, probability_distributions = zip(*replay_buffer)
        x = np.atleast_2d(np.array(root_states))
        y = np.atleast_2d(np.array(probability_distributions))
        if x.shape[1] != self._input_size:
            raise ValueError(f"Input vectors must have length {self._input_size}")

        if y.shape[1] != self._output_size:
            raise ValueError(f"Output vectors must have length {self._output_size}")

        self._nn.fit(x=x, y=y, batch_size=len(replay_buffer))

        return self