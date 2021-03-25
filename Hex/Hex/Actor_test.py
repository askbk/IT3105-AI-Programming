import numpy as np
from Hex.Actor import Actor
from Hex.Game import Board


def test_rollout():
    state = Board()
    distribution = (
        Actor(
            input_size=len(state.get_tuple_representation()),
            output_size=state._size ** 2,
        )
        .rollout(state.get_tuple_representation())
        .numpy()
        .flatten()
    )

    assert len(distribution) == state._size ** 2
    assert np.isclose(sum(distribution), 1)
