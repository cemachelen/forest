"""Controllers that generate actions and manage application state"""
import netCDF4
import bokeh.models
import bokeh.layouts
from forest import actions


class ActionLog(object):
    """Middleware to capture history of actions"""
    def __init__(self):
        self.actions = []

    def __call__(self, store):
        def inner(next_method):
            def inner_most(action):
                self.actions.append(action)
                next_method(action)
            return inner_most
        return inner


class NetCDF(object):
    """Middleware to extract meta-data from a file
    """
    def __call__(self, store):
        def inner(next_method):
            def inner_most(action):
                next_method(action)
                kind, *payload = action
                if kind == actions.SET_FILE:
                    file_name, = payload
                    with netCDF4.Dataset(file_name) as dataset:
                        variables = [v for v in dataset.variables.keys()]
                        next_method(actions.set_variables(variables))
                elif kind == actions.SET_VARIABLE:
                    file_name = store.state["file"]
                    variable = payload[0]
                    with netCDF4.Dataset(file_name) as dataset:
                        values = valid_times(dataset, variable)
                        next_method(actions.set_valid_times(values))
            return inner_most
        return inner


def valid_times(dataset, variable):
    var = dataset.variables["time_0"]
    return netCDF4.num2date(var[:], units=var.units)


class Database(object):
    """Middleware to query SQL database"""
    def __call__(self, store):
        def inner(next_method):
            def inner_most(action):
                next_method(action)
                kind, *payload = action
                if kind == actions.SET_FILE:
                    next_method(actions.set_variables(["mslp"]))
            return inner_most
        return inner


class Observable(object):
    def __init__(self):
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def notify(self, state):
        for callback in self.subscribers:
            callback(state)


class FileSystem(Observable):
    """Menu system based on file(s) meta-data"""
    def __init__(self):
        self.dropdown = bokeh.models.Dropdown()
        self.dropdown.on_change("value", self.on_file)
        self.layout = bokeh.layouts.Column(self.dropdown)
        super().__init__()

    def render(self, state):
        if "file" in state:
            self.dropdown.label = state["file"]
        if "file_names" in state:
            self.dropdown.menu = [(name, name) for name in state["file_names"]]

    def on_file(self, attr, old, new):
        self.notify(actions.set_file(new))


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
    kind, *payload = action
    if kind == actions.SET_FILE:
        state["file"] = payload[0]
    elif kind == actions.SET_FILE_NAMES:
        state["file_names"] = payload[0]
    elif kind == actions.SET_VARIABLES:
        state["variables"] = payload[0]
    elif kind == actions.SET_VARIABLE:
        state["variable"] = payload[0]
    elif kind == actions.SET_VALID_TIMES:
        state["valid_times"] = payload[0]
    return state
