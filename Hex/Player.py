import random
from Game import GameBase


class Player:

    def get_action(self, current_state: GameBase):
        return random.choice(current_state.get_possible_actions())

    def next_state(self, next_state: GameBase):
        return Player()

    def end_of_episode_update(self):
        return Player()