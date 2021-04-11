from __future__ import annotations
from warnings import simplefilter
from typing import Dict, Sequence
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
        network: keras.Model,
        optimizer="adam",
        loss="mean_squared_error",
        learning_rate=0.001,
        **kwargs,
    ):
        self._input_size = input_size
        self._output_size = output_size
        self._nn = network
        self._nn.compile(
            optimizer=Actor._initialize_optimizer(
                name=optimizer, learning_rate=learning_rate
            ),
            loss=loss,
        )

    @staticmethod
    def _initialize_optimizer(
        name: str, learning_rate: float
    ) -> keras.optimizers.Optimizer:
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
    def _initialize_network(
        actor_config: Dict, input_size: int, output_size: int
    ) -> keras.Model:
        def create_layer(prev_layer, curr_layer_config):
            return keras.layers.Dense(**curr_layer_config)(prev_layer)

        inputs = keras.layers.Input(shape=(input_size,))

        return keras.Model(
            inputs=inputs,
            outputs=keras.layers.Dense(
                units=output_size,
                activation=actor_config.get("output_activation", "softmax"),
            )(reduce(create_layer, actor_config.get("layers", []), inputs)),
        )

    @staticmethod
    def from_config(input_size: int, output_size: int, actor_config: Dict) -> Actor:

        return Actor(
            input_size=input_size,
            output_size=output_size,
            network=Actor._initialize_network(actor_config, input_size, output_size),
            **actor_config,
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

        self._nn.train_on_batch(x=x, y=y)

        return self
