from __future__ import annotations
import tensorflow as tf
from Hex.Types import Action
from Hex.Game import GameBase
from Hex.AgentBase import AgentBase
from Hex.Agent import Agent
from Hex.Actor import Actor


class GreedyNNAgent(AgentBase):
    def __init__(self, policy, state, name):
        self._state = state
        self._policy = policy
        self._name = name

    @staticmethod
    def from_saved_nn(path: str, initial_state: GameBase) -> GreedyNNAgent:
        nn = tf.keras.models.load_model(path)
        actor = Actor(
            initial_state.get_state_size(),
            initial_state.get_action_space_size(),
            network=nn,
        )
        return GreedyNNAgent(Agent._rollout_policy(actor), initial_state, path)

    def get_action(self) -> Action:
        return self._policy(self._state)

    def next_state(self, next_state: GameBase) -> Agent:
        return GreedyNNAgent(self._policy, next_state, self._name)

    def get_name(self):
        return self._name
