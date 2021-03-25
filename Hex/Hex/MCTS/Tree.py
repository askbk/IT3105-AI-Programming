from __future__ import annotations
from typing import Any, List, Optional
from Hex.Game import GameBase


class Tree:
    def __init__(
        self,
        state: GameBase,
        children: Tree = None,
        visit_count=0,
        value=0,
        action: Any = None,
    ):
        self._state = state
        self._is_end_state = state.is_end_state_reached()
        self._visit_count = visit_count
        self._value = value
        self._children = children
        self._action = action

    @staticmethod
    def _create_children(root_state: GameBase) -> List[Tree]:
        if root_state.is_end_state_reached():
            return None

        return [
            Tree(root_state.perform_action(action), action=action)
            for action in root_state.get_possible_actions()
        ]

    def increment_visit_count(self, reward=0) -> Tree:
        return Tree(
            self.get_state(),
            self.get_children(),
            self.get_visit_count() + 1,
            self.get_value() + reward,
            action=self.get_action(),
        )

    def get_state(self) -> Any:
        return self._state

    def get_action(self) -> Any:
        return self._action

    def get_value(self) -> int:
        return self._value

    def get_visit_count(self) -> int:
        return self._visit_count

    def get_children(self):
        if self.is_end_state():
            return None
        if self._children is not None:
            return self._children
        return Tree._create_children(self.get_state())

    def update_child_node(self, old: Tree, updated: Tree) -> Tree:
        new_children = [
            *filter(lambda child: child is not old, self.get_children()),
            updated,
        ]
        new_value = sum(child.get_value() for child in new_children)
        return Tree(
            self.get_state(),
            new_children,
            self.get_visit_count() + 1,
            new_value,
            action=self.get_action(),
        )

    def is_visited(self) -> bool:
        return self.get_visit_count() > 0

    def is_fully_expanded(self) -> bool:
        return self.is_end_state() or all(
            child.is_visited() for child in self.get_children()
        )

    def is_end_state(self) -> bool:
        return self._is_end_state

    def __repr__(self):
        return f"Tree<state={self.get_state()}, is_end_state={self.is_end_state()}, visit_count={self.get_visit_count()}, value={self.get_value()}>"
