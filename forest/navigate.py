import bokeh.models
from forest.observe import Observable
from forest.db.util import autowarn


__all__ = [
    "Navigator",
    "Pattern",
    "FileName"
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
        self.notify(("SET", "file_name", new))

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
        for key in ["pressure", "valid_time", "initial_time"]:
            self.buttons[key] = {
                'next': bokeh.models.Button(
                    label="Next",
                    width=widths["button"]),
                'previous': bokeh.models.Button(
                    label="Previous",
                    width=widths["button"]),
            }
            self.buttons[key]['next'].on_click(
                self.on_click('next', key))
            self.buttons[key]['previous'].on_click(
                self.on_click('previous', key))
            self.rows[key] = bokeh.layouts.row(
                self.buttons[key]["previous"],
                self.dropdowns[key],
                self.buttons[key]["next"])
        self.layout = bokeh.layouts.column(
            self.dropdowns["variable"],
            self.rows["initial_time"],
            self.rows["valid_time"],
            self.rows["pressure"])
        super().__init__()

    def render(self, state):
        """Configure dropdown menus"""
        if "file_names" in state:
            print("render", state["file_names"])
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
                    menu = self.menu(items)
                self.dropdowns[item_key].menu = menu
            self.dropdowns[item_key].disabled = disabled
            if item_key in self.buttons:
                self.buttons[item_key]["next"].disabled = disabled
                self.buttons[item_key]["previous"].disabled = disabled

    def on_change(self, category):
        # Facade: between bokeh callback and FOREST actions
        def callback(attr, old, new):
            self.notify(("SET", category, new))
        return callback

    def on_click(self, direction, category):
        # Facade: between bokeh callback and FOREST actions
        def callback():
            self.notify(("MOVE", direction, category))
        return callback
