import numpy as np
import tensorflow as tf
from functools import reduce
from tensorflow import keras
from tensorflow.keras import layers
from Agent.SplitGD import SplitGD
from Agent.FunctionApproximator import FunctionApproximator


class NN(SplitGD):
    def __init__(self, dimensions, learning_rate, calculate_eligibilities):
        self._dimensions = dimensions
        super().__init__(NN._build_network(dimensions, learning_rate))
        self._eligibility_shapes = [
            tensor.shape.as_list() for tensor in self.model.trainable_weights
        ]
        self.reset_eligibilities()
        self._learning_rate = learning_rate
        self._calculate_eligibilities = calculate_eligibilities

    def feed_forward(self, inputs):
        return self.model(inputs).numpy()

    def fit(self, batch, targets, td_error):
        self._td_error = td_error
        super().fit(batch, targets)

    def _modify_gradients(self, gradients):
        gradient_matrices = [tensor.numpy() for tensor in gradients]
        modified_gradients = [
            tf.convert_to_tensor(
                self._modify_gradient(gradient, eligibility), dtype=tf.float32
            )
            for gradient, eligibility in zip(gradient_matrices, self._eligibilities)
        ]
        return modified_gradients

    def _modify_gradient(self, gradient, eligibility):
        new_gradient = np.squeeze(self._td_error) * self._calculate_eligibilities(
            eligibility, gradient
        )
        return new_gradient

    def reset_eligibilities(self):
        self._eligibilities = [
            np.zeros(tuple(shape) if len(shape) > 1 else shape[0])
            for shape in self._eligibility_shapes
        ]

    @staticmethod
    def _build_network(dimensions, learning_rate):
        inputs = keras.Input(shape=(dimensions[0],))
        network = reduce(
            lambda previous_layer, layer: layer(previous_layer),
            [layers.Dense(units=n, activation="relu") for n in dimensions[1:]],
            inputs,
        )

        model = keras.Model(inputs=inputs, outputs=network)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=keras.losses.MeanSquaredError(),
            metrics=["mse"],
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
        self._discount_factor = discount_factor
        self._eligibility_decay_rate = eligibility_decay_rate
        if _network is None:
            self._network = NN(dimensions, learning_rate, self._calculate_eligibilities)
        else:
            self._network = _network

    def get_value(self, state, _batch=False):
        if _batch:
            return self._network.feed_forward(state)
        return self._network.feed_forward(np.array([eval(repr(state))]))

    def update(self, states, td_error):
        batch = np.array([np.array(eval(repr(state))) for state in states])
        targets = self.get_value(batch, _batch=True) + td_error
        self._network.fit(batch, targets, td_error)
        return NNFunctionApproximator(
            learning_rate=None,
            discount_factor=self._discount_factor,
            eligibility_decay_rate=self._eligibility_decay_rate,
            _network=self._network,
        )

    @staticmethod
    def _get_loss_derivative(td_error, value_derivative):
        return -2 * td_error * value_derivative

    def _calculate_eligibilities(self, eligibility, gradient):
        return (
            self._discount_factor * self._eligibility_decay_rate * eligibility
            + gradient
        )

    def reset_eligibilities(self):
        self._network.reset_eligibilities()
        return self