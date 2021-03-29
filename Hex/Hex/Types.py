import numpy as np
from Hex.Game import GameBase
from typing import Tuple, Sequence, Union, Any, Callable

StateVector = Union[np.array, Sequence]
ProbabilityDistribution = Union[np.array, Sequence]
ReplayBuffer = Tuple[StateVector, ProbabilityDistribution]
Action = Any
RolloutPolicy = Callable[[GameBase], Action]