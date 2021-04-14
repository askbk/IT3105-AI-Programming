from __future__ import annotations
import tensorflow as tf
from Hex.Types import Action
from Hex.Game import GameBase
from Hex.AgentBase import AgentBase
from Hex.Agent import Agent
from Hex.Actor import Actor


class GreedyNNAgent(AgentBase):
    def __init__(self, policy, state, name):
        self._initial_state = state
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

    @staticmethod
    def from_training_data(config, game: GameBase, data, train_config):
        actor = Actor.from_config(
            input_size=game.get_state_size(),
            output_size=game.get_action_space_size(),
            actor_config=config.get("actor", {}),
        ).train(data, train_config=train_config)
        return Agent(
            initial_state=game,
            epsilon=config.get("epsilon", 0),
            _actor=actor,
        )

    def get_action(self) -> Action:
        return self._policy(self._initial_state)

    def next_state(self, next_state: GameBase) -> Agent:
        return GreedyNNAgent(self._policy, next_state, self._name)

    def get_name(self):
        return self._name
