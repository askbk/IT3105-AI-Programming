from functools import reduce
import tensorflow.keras as keras
import numpy as np


class Actor:
    def __init__(self, input_size, output_size):
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

    def rollout(self, state_vector):
        return self._nn(np.atleast_2d(np.array(state_vector)))