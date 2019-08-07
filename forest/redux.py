"""Redux design pattern

Callbacks and MV* patterns are very common solutions to event-driven
design and indeed are a natural fit in most circumstances, but as
applications grow in complexity it becomes harder to keep track of state.
Centralising state into a singleton and only allowing updates to be
performed through reducers is one way to tackle callback hell and
complex model view controller relationships

"""
from collections import OrderedDict
from forest.observe import Observable
from forest import actions

__all__ = [
    "Store",
    "reducer"
]


class Store(Observable):
    def __init__(self, reducer, middlewares=None, state=None):
        self.reducer = reducer
        if state is None:
            state = {}
        self.state = state
        if middlewares is not None:
            mws = [m(self) for m in middlewares]
            f = self.dispatch
            for mw in reversed(mws):
                f = mw(f)
            self.dispatch = f
        super().__init__()

    def dispatch(self, action):
        self.state = self.reducer(self.state, action)
        self.notify(self.state)


def reducer(state, action):
    kind = action["kind"]
    if kind == actions.SET_ITEM:
        return set_reducer(state, action)
    elif kind == actions.NEXT_ITEM:
        return next_reducer(state, action)
    elif kind == actions.PREVIOUS_ITEM:
        return previous_reducer(state, action)
    return state


def set_reducer(state, action):
    state = dict(state)
    key, value = action["key"], action["value"]
    state[key] = value
    return state


def next_reducer(state, action):
    return select_reducer(state, action, next_value)


def next_value(items, item=None):
    if item is None:
        return max(items)
    items = list(sorted(items))
    i = items.index(item)
    return items[(i + 1) % len(items)]


def previous_reducer(state, action):
    return select_reducer(state, action, previous_value)


def previous_value(items, item):
    if item is None:
        return min(items)
    items = list(sorted(items))
    i = items.index(item)
    return items[i - 1]


def select_reducer(state, action, method):
    item_key = action["item_key"]
    items_key = action["items_key"]
    if items_key in state:
        item = state.get(item_key, None)
        items = state[items_key]
        state[item_key] = method(items, item)
        return state
    return state
