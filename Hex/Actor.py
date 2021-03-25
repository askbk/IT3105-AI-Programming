from functools import reduce
import tensorflow.keras as keras


class Actor:
    def __init__(self):
        inputs = keras.layers.Input(shape=(10,))
        self._nn = keras.Model(
            inputs=inputs,
            outputs=reduce(
                lambda prev_layer, curr_layer: curr_layer(prev_layer),
                [
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dense(9, activation="softmax"),
                ],
                inputs,
            ),
        )
