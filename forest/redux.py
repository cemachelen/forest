"""Redux design pattern

Callbacks and MV* patterns are very common solutions to event-driven
design and indeed are a natural fit in most circumstances, but as
applications grow in complexity it becomes harder to keep track of state.
Centralising state into a singleton and only allowing updates to be
performed through reducers is one way to tackle callback hell and
complex model view controller relationships

"""

__all__ = [
    "Store",
    "reducer"
]


class Observable(object):
    def __init__(self):
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def notify(self, state):
        for callback in self.subscribers:
            callback(state)


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
        attr, direction = rest
        key = {
            "pressure": "pressures",
            "initial_time": "initial_times",
            "valid_time": "valid_times"
        }[attr]
        items = state[key]
        state[attr] = items[0]
    return state
