from functools import reduce
from tensorflow import keras
from tensorflow.keras import layers
from Agent.SplitGD import SplitGD
from Agent.FunctionApproximator import FunctionApproximator


class NN(SplitGD):
    def __init__(self, dimensions):
        super(NN, self).__init__(NN._build_network(dimensions))

    @staticmethod
    def _build_network(dimensions):
        return reduce(
            lambda previous_layer, layer: layer(previous_layer),
            [layers.Dense(units=(n,), activation="relu") for n in dimensions[1:]],
            keras.Input(shape=(dimensions[0],)),
        )


class NNFunctionApproximator(FunctionApproximator):
    def __init__(self, dimensions=None, _network=None):
        if _network is None:
            self._network = NN(dimensions)
        else:
            self._network = _network

    @staticmethod
    def _get_loss_derivative(td_error, value_derivative):
        return -2 * td_error * value_derivative