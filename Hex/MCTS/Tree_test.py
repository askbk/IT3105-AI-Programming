from MCTS import Tree
from MCTS.Nim import Nim


def test_get_value():
    assert Tree(Nim(n=0, k=0)).get_value() == 0


def test_is_visited():
    assert not Tree(Nim(n=0, k=0)).is_visited()


def test_is_fully_expanded():
    assert Tree(Nim(n=0, k=1)).is_fully_expanded()
    assert not Tree(Nim(n=10, k=1)).is_fully_expanded()
    state = Nim(n=10, k=2)
    unvisited_children = [
        Tree(Nim(n=9, k=1), visit_count=1),
        Tree(Nim(n=8, k=1), visit_count=0),
    ]

    assert not Tree(
        state, children=unvisited_children, visit_count=1
    ).is_fully_expanded()
    visited_children = [
        Tree(Nim(n=9, k=1), visit_count=1),
        Tree(Nim(n=8, k=1), visit_count=2),
    ]
    assert Tree(state, visit_count=1, children=visited_children).is_fully_expanded()


def test_is_end_state():
    assert not Tree(Nim(n=1, k=1)).is_end_state()
    assert Tree(Nim(n=0, k=1)).is_end_state()


def test_increment_visit_count():
    tree = Tree(Nim(n=0, k=0))
    incremented = tree.increment_visit_count()
    assert incremented.get_value() == 0
    assert incremented.get_visit_count() == 1
    incremented2 = incremented.increment_visit_count(1)
    assert incremented2.get_visit_count() == 2
    assert incremented2.get_value() == 1
