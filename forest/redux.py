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

__all__ = [
    "Store",
    "reducer"
]


class Store(Observable):
    def __init__(self, reducer, middlewares=None):
        self.reducer = reducer
        self.state = {}
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
    state = dict(state)
    kind, *rest = action
    if kind.upper() == "SET":
        key, value = rest
        state[key] = value
    elif kind.upper() == "MOVE":
        state = move_reducer(state, action)
    elif kind.upper() == "ADD":
        category, name, settings = rest
        presets = state.get("presets", [])
        tree = OrderedDict({p["name"]: p for p in presets})
        data = dict(settings)
        data["name"] = name
        tree[name] = data
        state["presets"] = list(tree.values())
    elif kind.upper() == "REMOVE":
        category, name = rest
        state["presets"] = [
            item for item in state["presets"]
            if item["name"] != name
        ]
    return state


def move_reducer(state, action):
    _, item_key, items_key, direction = action
    if items_key in state:
        item = state.get(item_key, None)
        items = state[items_key]
        if direction.lower() == "increment":
            state[item_key] = increment(items, item)
        else:
            state[item_key] = decrement(items, item)
    return state


def increment(items, item):
    if item is None:
        return max(items)
    items = list(sorted(items))
    i = items.index(item)
    return items[(i + 1) % len(items)]


def decrement(items, item):
    if item is None:
        return min(items)
    items = list(sorted(items))
    i = items.index(item)
    return items[i - 1]
