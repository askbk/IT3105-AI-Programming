from typing import Callable, Any


def while_loop(condition: Callable[Any, bool], initialState: Any, body: Callable):
    if not condition(initialState):
        return initialState

    return while_loop(condition, body(initialState), body)
