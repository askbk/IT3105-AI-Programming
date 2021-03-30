import numpy as np
import Hex.Game.GameBase as GameBase
from typing import Tuple, Sequence, Union, Any, Callable

StateVector = Union[np.array, Sequence]
ProbabilityDistribution = Union[np.array, Sequence]
ReplayBuffer = Tuple[StateVector, ProbabilityDistribution]
Action = Any
RolloutPolicy = Callable[[GameBase], Action]