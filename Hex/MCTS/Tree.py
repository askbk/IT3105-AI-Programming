from Game import GameBase


class Tree:
    def __init__(
        self, state: GameBase, children=None, visit_count=0, value=0, action=None
    ):
        self._state = state
        self._is_end_state = state.is_end_state_reached()
        self._visit_count = visit_count
        self._value = value
        self._children = children
        self._action = action

    @staticmethod
    def _create_children(root_state: GameBase):
        if root_state.is_end_state_reached():
            return None

        return [
            Tree(root_state.perform_action(action), action=action)
            for action in root_state.get_possible_actions()
        ]

    def increment_visit_count(self, reward=0):
        return Tree(
            self._state,
            self.get_children(),
            self._visit_count + 1,
            self._value + reward,
        )

    def get_state(self):
        return self._state

    def get_action(self):
        return self._action

    def get_value(self):
        return self._value

    def get_visit_count(self):
        return self._visit_count

    def get_children(self):
        if self.is_end_state():
            return None
        if self._children is not None:
            return self._children
        return Tree._create_children(self._state)

    def update_child_node(self, old, updated):
        new_children = [
            *filter(lambda child: child is not old, self.get_children()),
            updated,
        ]
        new_value = sum([child.get_value() for child in new_children])
        return Tree(self._state, new_children, self._visit_count + 1, new_value)

    def is_visited(self):
        return self._visit_count > 0

    def is_fully_expanded(self):
        return self.is_end_state() or all(
            [child.is_visited() for child in self.get_children()]
        )

    def is_end_state(self):
        return self._is_end_state

    def __repr__(self):
        return f"Tree<state={self._state}, is_end_state={self.is_end_state()}, visit_count={self.get_visit_count()}, value={self.get_value()}>"
