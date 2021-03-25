import numpy as np
from Hex.Actor import Actor
from Hex.Game import Board


def test_rollout():
    state = Board()
    distribution = Actor(
        input_size=len(state.get_tuple_representation()),
        output_size=state._size ** 2,
    ).rollout(state.get_tuple_representation())

    assert len(distribution) == state._size ** 2
    assert np.isclose(sum(distribution), 1)


def test_train():
    state = Board(size=2)
    untrained = Actor(
        input_size=len(state.get_tuple_representation()), output_size=state._size ** 2
    )

    replay_buffer = [
        ((1, 0, 0, 0, 0), [0.1, 0.2, 0.3, 0.4]),
        ((2, 1, 0, 0, 0), [0.2, 0.2, 0.2, 0.4]),
        ((1, 1, 2, 0, 0), [0.3, 0.4, 0.2, 0.1]),
    ]

    trained = untrained.train(replay_buffer)
    distribution = trained.rollout(state.get_tuple_representation())
    assert len(distribution) == state._size ** 2
    assert np.isclose(sum(distribution), 1)
