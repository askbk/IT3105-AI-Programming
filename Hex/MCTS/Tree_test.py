from MCTS import Tree
from MCTS.Nim import Nim


def test_get_value():
    assert Tree(Nim(n=0, k=0)).get_value() == 0


def test_get_state():
    assert Tree(Nim(n=1, k=1)).get_state() == Nim(n=1, k=1)


def test_get_action():
    assert Tree(Nim(n=1, k=1), action=2).get_action() == 2


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


def test_update_child_node():
    child1 = Tree(Nim(n=9, k=1), visit_count=1, value=1)
    child2 = Tree(Nim(n=8, k=1), visit_count=0)
    tree = Tree(Nim(n=10, k=2), children=[child1, child2], value=1)
    assert not tree.is_fully_expanded()
    assert tree.get_visit_count() == 0
    assert tree.get_value() == 1
    updated_tree = tree.update_child_node(
        child2, child2.increment_visit_count(reward=2)
    )
    assert updated_tree.is_fully_expanded()
    assert updated_tree.get_value() == 3
    assert updated_tree.get_visit_count() == 1


def test_get_children():
    child1 = Tree(Nim(n=9, k=1), visit_count=1, value=1)
    child2 = Tree(Nim(n=8, k=1), visit_count=0)
    tree = Tree(Nim(n=10, k=2), children=[child1, child2], value=1)
    children = tree.get_children()
    assert all([isinstance(child, Tree) for child in children])
    assert child1 in children and child2 in children
    assert len(children) == 2
