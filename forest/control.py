"""Controllers that generate actions and manage application state"""
import netCDF4
import bokeh.models
import bokeh.layouts
from forest import actions
from forest.observe import Observable


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
                kind, *rest = action
                if kind.upper() == "SET":
                    attr, value = rest
                    if attr.upper() == "FILE_NAME":
                        file_name = value
                        with netCDF4.Dataset(file_name) as dataset:
                            variables = [v for v in dataset.variables.keys()]
                            next_method(actions.SET.variables.to(variables))
                    elif attr.upper() == "VARIABLE":
                        file_name = store.state["file_name"]
                        variable = value
                        with netCDF4.Dataset(file_name) as dataset:
                            values = valid_times(dataset, variable)
                            next_method(actions.SET.valid_times.to(values))
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
                kind, *rest = action
                if kind.upper() == "SET":
                    attr, value = rest
                    if attr.upper() == "FILE_NAME":
                        next_method(actions.SET.variables.to(["mslp"]))
            return inner_most
        return inner


class FileSystem(Observable):
    """Menu system based on file(s) meta-data"""
    def __init__(self):
        self.dropdown = bokeh.models.Dropdown()
        self.dropdown.on_change("value", self.on_file)
        self.layout = bokeh.layouts.Column(self.dropdown)
        super().__init__()

    def render(self, state):
        if "file_name" in state:
            self.dropdown.label = state["file_name"]
        if "file_names" in state:
            self.dropdown.menu = [(name, name) for name in state["file_names"]]

    def on_file(self, attr, old, new):
        self.notify(actions.SET.file_name.to(new))


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
