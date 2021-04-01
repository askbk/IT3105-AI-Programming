import numpy as np
from Hex.Actor import Actor
from Hex.Game import Board


def test_rollout():
    state = Board()
    actor = Actor.from_config(
        input_size=len(state.get_tuple_representation()),
        output_size=state._size ** 2,
        actor_config={
            "output_activation": "softmax",
            "layers": [{"units": 10, "activation": "sigmoid"}],
        },
    )
    print(actor._nn.summary())

    distribution = actor.rollout(state.get_tuple_representation())

    assert len(distribution) == state._size ** 2
    assert np.isclose(sum(distribution), 1)


def test_train():
    state = Board(size=2)
    untrained = Actor.from_config(
        input_size=len(state.get_tuple_representation()),
        output_size=state._size ** 2,
        actor_config={
            "output_activation": "softmax",
            "layers": [{"units": 10, "activation": "sigmoid"}],
        },
    )
    print(untrained._nn.summary())

    replay_buffer = [
        ((1, 0, 0, 0, 0), [0.1, 0.2, 0.3, 0.4]),
        ((2, 1, 0, 0, 0), [0.2, 0.2, 0.2, 0.4]),
        ((1, 1, 2, 0, 0), [0.3, 0.4, 0.2, 0.1]),
    ]

    trained = untrained.train(replay_buffer)
    distribution = trained.rollout(state.get_tuple_representation())
    assert len(distribution) == state._size ** 2
    assert np.isclose(sum(distribution), 1)
