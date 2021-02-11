import numpy as np
from functools import reduce
from tensorflow import keras
from tensorflow.keras import layers
from Agent.SplitGD import SplitGD
from Agent.FunctionApproximator import FunctionApproximator


class NN(SplitGD):
    def __init__(self, dimensions):
        super().__init__(NN._build_network(dimensions))

    def feed_forward(self, inputs):
        return self.model(inputs).numpy()

    def _modify_gradients(self, gradients):
        return gradients

    @staticmethod
    def _build_network(dimensions):
        inputs = keras.Input(shape=(None, dimensions[0]))
        network = reduce(
            lambda previous_layer, layer: layer(previous_layer),
            [layers.Dense(units=n, activation="relu") for n in dimensions[1:]],
            inputs,
        )

        model = keras.Model(inputs=inputs, outputs=network)
        model.compile(
            optimizer="adam", loss=keras.losses.MeanSquaredError(), metrics=["mse"]
        )
        return model


class NNFunctionApproximator(FunctionApproximator):
    def __init__(
        self,
        learning_rate,
        discount_factor,
        eligibility_decay_rate,
        dimensions=None,
        _network=None,
    ):
        if _network is None:
            self._network = NN(dimensions)
        else:
            self._network = _network

    def get_value(self, state):
        return self._network.feed_forward(state)

    def update(self, states, td_error):
        batch = np.array(states)
        targets = self.get_value(batch) + td_error
        self._network.fit(batch, targets)
        return NNFunctionApproximator(
            learning_rate=None,
            discount_factor=None,
            eligibility_decay_rate=None,
            _network=self._network,
        )

    @staticmethod
    def _get_loss_derivative(td_error, value_derivative):
        return -2 * td_error * value_derivative