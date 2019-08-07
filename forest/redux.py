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
from forest import reducers

__all__ = [
    "Store",
    "reducer",
    "combine_reducers",
    "subtree"
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


def subtree(callback, key):
    """Pass state sub-tree to callback/view"""
    def wrapper(state):
        callback(state.get(key, {}))
    return wrapper


def combine_reducers(**reducers):
    """Helper method to decouple reducer responsibilities"""
    def reducer(state, action):
        for key, reducer in reducers.items():
            state[key] = reducer(state.get(key, {}), action)
        return state
    return reducer


reducer = combine_reducers(**{
    'navigate': reducers.navigate,
    'preset': reducers.preset
})
