import netCDF4
import bokeh.models
from forest.observe import Observable
from forest.db.util import autowarn
from forest import actions
from forest.middleware import middleware


__all__ = [
    "Navigator",
    "Pattern",
    "FileName",
    "SQL",
    "FileSystem"
]


class Pattern(Observable):
    def __init__(self, drop_down=None):
        if drop_down is None:
            drop_down = bokeh.models.Dropdown(label="Model/observation")
        self.drop_down = drop_down
        self.drop_down.on_change("value", self.on_change)
        super().__init__()

    def on_change(self, attr, old, new):
        self.notify(("SET", "pattern", new))

    def render(self, state):
        if "pattern" in state:
            for label, pattern in state["patterns"]:
                if pattern == state["pattern"]:
                    self.drop_down.label = label
                    break
        if "patterns" in state:
            self.drop_down.menu = state["patterns"]


class FileName(Observable):
    def __init__(self, drop_down=None):
        if drop_down is None:
            drop_down = bokeh.models.Dropdown(label="File(s)")
        self.drop_down = drop_down
        self.drop_down.on_change("value", self.on_change)
        super().__init__()

    def on_change(self, attr, old, new):
        self.notify(actions.set_item("file_name", new))

    def render(self, state):
        if "file_name" in state:
            self.drop_down.label = state["file_name"]
        if "file_names" in state:
            self.drop_down.menu = [(s, s) for s in state["file_names"]]


class Navigator(Observable):
    """Navigation user interface"""
    def __init__(self):
        widths = {
            "dropdown": 180,
            "button": 75
        }
        self.dropdowns = {
            "variable": bokeh.models.Dropdown(
                label="Variable"),
            "initial_time": bokeh.models.Dropdown(
                label="Initial time",
                width=widths["dropdown"]),
            "valid_time": bokeh.models.Dropdown(
                label="Valid time",
                width=widths["dropdown"]),
            "pressure": bokeh.models.Dropdown(
                label="Pressure",
                width=widths["dropdown"])
        }
        for key, dropdown in self.dropdowns.items():
            autowarn(dropdown)
            dropdown.on_change("value", self.on_change(key))
        self.rows = {}
        self.buttons = {}
        for item_key, items_key in [
                ("pressure", "pressures"),
                ("valid_time", "valid_times"),
                ("initial_time", "initial_times")]:
            self.buttons[item_key] = {
                'next': bokeh.models.Button(
                    label="Next",
                    width=widths["button"]),
                'previous': bokeh.models.Button(
                    label="Previous",
                    width=widths["button"]),
            }
            self.buttons[item_key]['next'].on_click(
                self.on_click(item_key, items_key, 'next'))
            self.buttons[item_key]['previous'].on_click(
                self.on_click(item_key, items_key, 'previous'))
            self.rows[item_key] = bokeh.layouts.row(
                self.buttons[item_key]["previous"],
                self.dropdowns[item_key],
                self.buttons[item_key]["next"])
        self.layout = bokeh.layouts.column(
            self.dropdowns["variable"],
            self.rows["initial_time"],
            self.rows["valid_time"],
            self.rows["pressure"])
        super().__init__()

    def render(self, state):
        """Configure dropdown menus"""
        for key in [
                "variable",
                "initial_time",
                "valid_time",
                "pressure"]:
            if key in state:
                self.dropdowns[key].label = str(state[key])
        for item_key, items_key in [
                ("variable", "variables"),
                ("initial_time", "initial_times"),
                ("valid_time", "valid_times"),
                ("pressure", "pressures")]:
            if items_key not in state:
                disabled = True
            else:
                items = state[items_key]
                disabled = len(items) == 0
                if item_key == "pressure":
                    menu = [(self.hpa(p), str(p)) for p in items]
                else:
                    menu = [(str(i), str(i)) for i in items]
                self.dropdowns[item_key].menu = menu
            self.dropdowns[item_key].disabled = disabled
            if item_key in self.buttons:
                self.buttons[item_key]["next"].disabled = disabled
                self.buttons[item_key]["previous"].disabled = disabled

    def on_change(self, category):
        # Facade: between bokeh callback and FOREST actions
        def callback(attr, old, new):
            self.notify(actions.set_item(category, new))
        return callback

    def on_click(self, item_key, items_key, direction):
        # Facade: between bokeh callback and FOREST actions
        def callback():
            msg = "unknown direction: '{}'".format(direction)
            assert direction.lower() in ["next", "previous"], msg
            if direction.lower() == "next":
                self.notify(actions.next_item(item_key, items_key))
            else:
                self.notify(actions.previous_item(item_key, items_key))
        return callback


class SQL(object):
    """SQL queries to populate state"""
    @middleware
    def __call__(self, store, next_method, action):
        next_method(action)


class FileSystem(object):
    """Access file system to populate state"""
    @middleware
    def __call__(self, store, next_method, action):
        next_method(action)
        kind = action["kind"]
        if kind == actions.SET_ITEM:
            key = action["key"]
            if key.lower() == "file_name":
                self.set_file_name(store, next_method, action)
            elif key.lower() == "variable":
                self.set_variable(store, next_method, action)

    def set_file_name(self, store, next_method, action):
        file_name = action["value"]
        values = variables(file_name)
        next_method(actions.set_item("variables", values))
        if "variable" in store.state:
            values = valid_times(file_name)
            next_method(actions.set_item("valid_times", values))

    def set_variable(self, store, next_method, action):
        file_name = store.state["file_name"]
        values = valid_times(file_name)
        next_method(actions.set_item("valid_times", values))


def variables(file_name):
    with netCDF4.Dataset(file_name) as dataset:
        values = [v for v in dataset.variables.keys()]
    return values


def valid_times(file_name):
    with netCDF4.Dataset(file_name) as dataset:
        var = dataset.variables["time"]
        values = netCDF4.num2date(var[:], units=var.units)
    return values
