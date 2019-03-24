"""Minimalist implementation of FOREST"""
from collections import defaultdict
import datetime as dt
import re
import os
import glob
import yaml
import bokeh.plotting
import bokeh.models
import cartopy
import numpy as np
import netCDF4
import cf_units
import scipy.ndimage
from threading import Thread
from tornado import gen
from functools import partial
from bokeh.document import without_document_lock
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

import sys
sys.path.insert(0, os.path.dirname(__file__))

import rx
import ui
import bonsai
from util import timed


class Config(object):
    def __init__(self,
                 title="Bonsai - miniature Forest",
                 lon_range=None,
                 lat_range=None,
                 models=None,
                 observations=None):
        def assign(value, default):
            return default if value is None else value
        self.title = title
        self.lon_range = assign(lon_range, [-180, 180])
        self.lat_range = assign(lat_range, [-80, 80])
        self.models = assign(models, [])
        self.observations = assign(observations, [])

    @classmethod
    def load(cls, path):
        with open(path) as stream:
            data = yaml.load(stream)
        return cls(**data)


class Environment(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


def parse_env():
    config_file = os.environ.get("FOREST_CONFIG", None)
    directory = os.environ.get("FOREST_DIR", None)
    return Environment(
        config_file=config_file,
        directory=directory)


class LevelSelector(object):
    def __init__(self):
        self.slider = bokeh.models.Slider(
            start=0,
            end=4,
            step=1,
            value=0,
            height=100,
            show_value=False,
            direction="rtl",
            orientation="vertical")
        self.slider.on_change("value", self.on_change)

    def on_change(self, attr, old, new):
        print(attr, old, new)


def select(dropdown):
    def on_click(value):
        for label, v in dropdown.menu:
            if v == value:
                dropdown.label = label
                break
    return on_click


class Store(object):
    def __init__(self, reducer, state=None):
        self.reducer = reducer
        if state is None:
            state = {}
        self.state = state
        self._uid = 0
        self.listeners = OrderedDict()

    def uid(self):
        self._uid = self._uid + 1
        return self._uid

    def dispatch(self, action):
        self.state = self.reducer(self.state, action)
        for listener in self.listeners.values():
            listener()

    def subscribe(self, listener):
        uid = self.uid()
        self.listeners[uid] = listener
        return partial(self.unsubscribe, uid)

    def unsubscribe(self, uid):
        del self.listeners[uid]


def reducer(state, action):
    state = dict(state)
    if action["type"] == "SET_TITLE":
        state["title"] = action["text"]
    elif action["type"] == "SET_VALID_DATE":
        state["valid_date"] = action["value"]
    elif action["type"] == "SET_FORECAST":
        state["valid_date"] = action["valid_date"]
        state["length"] = action["length"]
        state["run_date"] = action["run_date"]
    elif action["type"] == "SET_NAME":
        category = action["category"]
        state[category] = state.get(category, {})
        state[category]["name"] = action["text"]
    elif action["type"] == "SET_MODEL_FIELD":
        state["model"] = state.get("model", {})
        state["model"]["field"] = action["text"]
    elif action["type"] in ["ACTIVATE", "DEACTIVATE"]:
        active = action["type"] == "ACTIVATE"
        category = action["category"]
        if category in state:
            state[category]["active"] = active
        else:
            state[category] = {"active": active}
    elif action["type"] == "REQUEST":
        flag = action["flag"]
        if action["status"] == "succeed":
            state[flag] = False
            response = action["response"]
            if flag == "listing":
                if "files" in state:
                    state["files"].update(response)
                else:
                    state["files"] = dict(response)
            else:
                state["loaded"] = response
        else:
            state[flag] = True
    elif action["type"] == "FILE_FOUND":
        state["found"] = action["status"]
    return state


class Action(object):
    @staticmethod
    def set_valid_date(date):
        return {
            "type": "SET_VALID_DATE",
            "value": date
        }

    @staticmethod
    def set_forecast(valid_date, length):
        return {
            "type": "SET_FORECAST",
            "valid_date": valid_date,
            "length": length,
            "run_date": valid_date - length
        }

    @staticmethod
    def set_title(text):
        return {
            "type": "SET_TITLE",
            "text": text
        }

    @staticmethod
    def set_model_name(text):
        return Action.set_name("model", text)

    @staticmethod
    def set_observation_name(text):
        return Action.set_name("observation", text)

    @staticmethod
    def set_name(category, text):
        return {
            "type": "SET_NAME",
            "category": category,
            "text": text
        }

    @staticmethod
    def set_model_field(text):
        return {
            "type": "SET_MODEL_FIELD",
            "text": text
        }

    @staticmethod
    def activate(category):
        return {
            "type": "ACTIVATE",
            "category": category
        }

    @staticmethod
    def deactivate(category):
        return {
            "type": "DEACTIVATE",
            "category": category
        }


class Request(object):
    def __init__(self, flag):
        self.flag = flag

    def started(self):
        return {
            "type": "REQUEST",
            "flag": self.flag,
            "status": "active"}

    def finished(self, response):
        return {
            "type": "REQUEST",
            "flag": self.flag,
            "status": "succeed",
            "response": response}

    def failed(self):
        return {
            "type": "REQUEST",
            "flag": self.flag,
            "status": "fail"}


class List(Request):
    def __init__(self):
        super().__init__("listing")


class Load(Request):
    def __init__(self):
        super().__init__("loading")


def file_found(status):
    return {
        "type": "FILE_FOUND",
        "status": status
    }

FILE_FOUND = file_found(True)
FILE_NOT_FOUND = file_found(False)


class Application(object):
    def __init__(self, config, directory=None):
        self.store = Store(reducer, {"model": {"active": True}})
        self.unsubscribe = self.store.subscribe(self.state_change)
        self.document = bokeh.plotting.curdoc()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.patterns = file_patterns(
            config.models + config.observations,
            directory)
        self.figures = {
            "map": full_screen_figure(
                lon_range=config.lon_range,
                lat_range=config.lat_range)
        }
        self.toolbar_box = bokeh.models.ToolbarBox(
            toolbar=self.figures["map"].toolbar,
            toolbar_location="below")

        self.title = Title(self.figures["map"])
        self.messenger = Messenger(self.figures["map"])
        self.image = Image(self.figures["map"])
        self.dropdowns = {}
        self.dropdowns["model"] = bokeh.models.Dropdown(
                label="Configuration",
                menu=as_menu(pluck(config.models, "name")))
        self.dropdowns["model"].on_click(select(self.dropdowns["model"]))
        self.dropdowns["model"].on_click(self.on_click(Action.set_model_name))
        self.dropdowns["observation"] = bokeh.models.Dropdown(
                label="Instrument/satellite",
                menu=as_menu(pluck(config.observations, "name")))
        self.dropdowns["observation"].on_click(select(self.dropdowns["observation"]))
        self.dropdowns["observation"].on_click(
            self.on_click(Action.set_observation_name))
        self.dropdowns["field"] = bokeh.models.Dropdown(
            label="Field",
            menu=[
                ("Precipitation", "precipitation"),
                ("Outgoing longwave radiation (OLR)", "olr"),
            ])
        self.dropdowns["field"].on_click(select(self.dropdowns["field"]))
        self.dropdowns["field"].on_click(self.on_click(Action.set_model_field))
        self.datetime_picker = bonsai.DatetimePicker()
        self.datetime_picker.date_picker.title = "Valid date"
        self.datetime_picker.on_change(
            "value", self.on_change(Action.set_valid_date))

        overlay_checkboxes = bokeh.models.CheckboxGroup(
            labels=["MSLP", "Wind vectors"],
            inline=True)
        minus, div, plus = (
            bokeh.models.Button(label="-", width=135),
            bokeh.models.Div(text="", width=10),
            bokeh.models.Button(label="+", width=135))
        self.tabs = bokeh.models.Tabs(tabs=[
            bokeh.models.Panel(child=bokeh.layouts.column(
                bokeh.layouts.row(
                    bokeh.layouts.column(minus),
                    bokeh.layouts.column(div),
                    bokeh.layouts.column(plus)),
                self.dropdowns["model"],
                self.dropdowns["field"],
                overlay_checkboxes,
            ), title="Model"),
            bokeh.models.Panel(child=bokeh.layouts.column(
                self.dropdowns["observation"],
            ), title="Observation")])
        self.tabs.on_change("active", self.on_tab_change)
        self.loaded_state = {}

    def on_tab_change(self, attr, old, new):
        if new == 0:
            self.store.dispatch(Action.deactivate("observation"))
            self.store.dispatch(Action.activate("model"))
        else:
            self.store.dispatch(Action.deactivate("model"))
            self.store.dispatch(Action.activate("observation"))

    def state_change(self):
        state = self.store.state
        # print(state)

        found = state.get("found", False)
        listing = state.get("listing", False)
        loading = state.get("loading", False)
        if listing:
            self.messenger.text = "Searching..."
        elif loading:
            self.messenger.text = "Loading..."
        elif not found:
            self.messenger.text = "File not found"
        else:
            self.messenger.text = ""

        if not listing:
            for category in ["model", "observation"]:
                if category not in state:
                    continue
                if not state[category].get("active", False):
                    continue
                if "name" not in state[category]:
                    continue
                name = state[category]["name"]
                files = state.get("files", {})
                pattern = self.patterns[name]
                if name not in files:
                    self.submit(List(), self.list_files(name, pattern))

        if not loading and not listing:
            if "valid_date" in state:
                if "files" in state:
                    self.load_file(state)

        self.render(state)

    def load_file(self, state):
        name = self.get_active(state)
        if name is None:
            return
        if name not in state["files"]:
            return
        paths = state["files"][name]
        valid_date = state["valid_date"]

        print("exhaustively searching all files")
        print(find_file(paths, valid_date))

        # times = np.array([parse_time(path) for path in paths], dtype=object)
        # try:
        #     i = np.argmax(times[times < valid_date])
        # except ValueError:
        #     return
        # path = paths[i]
        # index = 0
        # if self.load_needed(path, index, state):
        #     print(path, index)
        #     self.submit(Load(), self.load(path, index))

    @staticmethod
    def load_needed(path, index, state):
        if "loaded" not in state:
            return True
        if path != state["loaded"]["path"]:
            return True
        if index != state["loaded"]["index"]:
            return True
        return False

    def load(self, path, index):
        def task():
            with netCDF4.Dataset(path) as dataset:
                data = load_index(dataset, index)
            return {
                "path": path,
                "index": index,
                "data": data
            }
        return task

    @timed
    def list_files(self, key, pattern):
        def task():
            files = glob.glob(pattern)
            return {key: files}
        return task

    def submit(self, request, blocking_task):
        completed = partial(self.completed, request)
        self.document.add_next_tick_callback(
            partial(self.unlocked_task, blocking_task, completed))
        self.store.dispatch(request.started())

    @gen.coroutine
    @without_document_lock
    def unlocked_task(self, blocking_task, completed):
        response = yield self.executor.submit(blocking_task)
        self.document.add_next_tick_callback(partial(completed, response))

    @gen.coroutine
    def completed(self, request, response):
        self.store.dispatch(request.finished(response))

    def on_click(self, action_method):
        def wrapper(value):
            self.store.dispatch(action_method(value))
        return wrapper

    def on_change(self, action_method):
        def wrapper(attr, old, new):
            self.store.dispatch(action_method(new))
        return wrapper

    def render(self, state):
        self.title.text = self.title_text(state)
        if "loaded" in state:
            self.image.source.data = state["loaded"]["data"]

    def title_text(self, state):
        parts = []
        name = self.get_active(state)
        if name is not None:
            parts.append(name)
        if "valid_date" in state:
            parts.append(state["valid_date"].strftime("%Y-%m-%d %H:%M"))
        return " ".join(parts)

    @staticmethod
    def get_active(state):
        for category in ["model", "observation"]:
            if category not in state:
                continue
            if "name" not in state[category]:
                continue
            if state[category].get("active", False):
                return state[category]["name"]


def main():
    env = parse_env()
    if env.config_file is None:
        config = Config()
    else:
        config = Config.load(env.config_file)

    application = Application(config, env.directory)
    figure = application.figures["map"]

    level_selector = LevelSelector()

    datetime_picker = application.datetime_picker

    def on_click(datetime_picker, incrementer):
        def callback():
            datetime_picker.value = incrementer(datetime_picker.value)
        return callback

    minus, div, plus = (
        bokeh.models.Button(label="-", width=135),
        bokeh.models.Div(text="", width=10),
        bokeh.models.Button(label="+", width=135))
    plus.on_click(on_click(datetime_picker, lambda d: d + dt.timedelta(days=1)))
    minus.on_click(on_click(datetime_picker, lambda d: d - dt.timedelta(days=1)))

    button_row = bokeh.layouts.row(
        bokeh.layouts.column(minus),
        bokeh.layouts.column(div),
        bokeh.layouts.column(plus))

    controls = bokeh.layouts.column(
        datetime_picker.date_picker,
        button_row,
        datetime_picker.hour_slider,
        application.tabs,
        name="controls")

    height_controls = bokeh.layouts.column(
        level_selector.slider,
        name="height")

    document = bokeh.plotting.curdoc()
    document.add_root(figure)
    document.add_root(application.toolbar_box)
    document.add_root(controls)
    document.add_root(height_controls)
    document.title = config.title


def pluck(items, key):
    return [item[key] for item in items]


def as_menu(items):
    return [(item, item) for item in items]


def load_times(dataset):
    for name in ["time_2", "time"]:
        if name not in dataset.variables:
            continue
        units = dataset.variables[name].units
        values = dataset.variables[name][:]
        return netCDF4.num2date(values, units=units)


def hours_before(date):
    def wrapped(path):
        d = parse_time(path)
        return (date - d).total_seconds() / (60 * 60)
    return wrapped


@timed
def find_file(paths, date):
    none_files = [
            path for path in paths
            if parse_time(path) is None]
    stamp_files = [
            path for path in paths
            if parse_time(path) is not None]
    before_files = [
            path for path in stamp_files
            if parse_time(path) <= date]
    before_files = sorted(before_files,
            key=hours_before(date))

    for path in before_files:
        with netCDF4.Dataset(path) as dataset:
            values = dataset.variables["time_2_bnds"][:]
            units = dataset.variables["time_2"].units
        bounds = netCDF4.num2date(values, units=units)
        start, end = np.min(bounds), np.max(bounds)
        if date > end:
            return
        if start <= date <= end:
            index = np.where(
                    (bounds[:, 0] <= date) &
                    (date < bounds[:, 1]))
            return path, index[0]

    for path in none_files:
        with netCDF4.Dataset(path) as dataset:
            values = dataset.variables["time_2_bnds"][:]
            units = dataset.variables["time_2"].units
        bounds = netCDF4.num2date(values, units=units)
        start, end = np.min(bounds), np.max(bounds)
        if start <= date <= end:
            index = np.where(
                    (bounds[:, 0] <= date) &
                    (date < bounds[:, 1]))
            return path, index[0]


def find_by_date(paths, date):
    for path in sorted(paths):
        if parse_time(path) == date:
            return path


def most_recent(times, time):
    """Helper to find files likely to contain data"""
    if isinstance(times, list):
        times = np.array(times, dtype=object)
    return np.max(times[times < time])


def file_patterns(models, directory=None):
    table = {}
    for entry in models:
        name, pattern = entry["name"], entry["pattern"]
        if directory is not None:
            pattern = os.path.join(directory, pattern)
        table[name] = pattern
    return table


def parse_time(path):
    file_name = os.path.basename(path)
    patterns = [
        ("[0-9]{8}T[0-9]{4}Z", "%Y%m%dT%H%MZ"),
        ("[0-9]{8}", "%Y%m%d"),
    ]
    for regex, fmt in patterns:
        matches = re.search(regex, file_name)
        if matches is None:
            continue
        timestamp = matches.group()
        return dt.datetime.strptime(timestamp, fmt)


class Image(object):
    def __init__(self, figure):
        self.figure = figure
        self.empty_data = {
            "x": [],
            "y": [],
            "dw": [],
            "dh": [],
            "image": []
        }
        self.source = bokeh.models.ColumnDataSource(self.empty_data)
        color_mapper = bokeh.models.LinearColorMapper(
            palette="RdYlBu11",
            nan_color=bokeh.colors.RGB(0, 0, 0, a=0),
            low=0,
            high=32
        )
        figure.image(
            x="x",
            y="y",
            dw="dw",
            dh="dh",
            image="image",
            source=self.source,
            color_mapper=color_mapper)
        colorbar = bokeh.models.ColorBar(
            title="Preciptation rate (mm/h)",
            color_mapper=color_mapper,
            orientation="horizontal",
            background_fill_alpha=0.,
            location="bottom_center",
            major_tick_line_color="black",
            bar_line_color="black")
        figure.add_layout(colorbar, 'center')


def load(path, valid_time):
    with netCDF4.Dataset(path) as dataset:
        bounds, units = ui.time_bounds(dataset)
        bounds = netCDF4.num2date(bounds, units=units)
        index = time_index(bounds, valid_time)
        data = load_index(dataset, index)
    return data


def load_index(dataset, index):
    try:
        lons = dataset.variables["longitude_0"][:]
    except KeyError:
        lons = dataset.variables["longitude"][:]
    try:
        lats = dataset.variables["latitude_0"][:]
    except KeyError:
        lats = dataset.variables["latitude"][:]
    try:
        var = dataset.variables["stratiform_rainfall_rate"]
    except KeyError:
        var = dataset.variables["precipitation_flux"]
    values = var[index]
    if var.units == "mm h-1":
        values = values
    else:
        values = convert_units(values, var.units, "kg m-2 hour-1")
    gx, _ = transform(
        lons,
        np.zeros(len(lons), dtype="d"),
        cartopy.crs.PlateCarree(),
        cartopy.crs.Mercator.GOOGLE)
    _, gy = transform(
        np.zeros(len(lats), dtype="d"),
        lats,
        cartopy.crs.PlateCarree(),
        cartopy.crs.Mercator.GOOGLE)
    values = stretch_y(gy)(values)
    image = np.ma.masked_array(values, values < 0.1)
    x = gx.min()
    y = gy.min()
    dw = gx.max() - gx.min()
    dh = gy.max() - gy.min()
    data = {
        "x": [x],
        "y": [y],
        "dw": [dw],
        "dh": [dh],
        "image": [image]
    }
    return data


def time_index(bounds, time):
    if isinstance(bounds, list):
        bounds = np.asarray(bounds, dtype=object)
    lower, upper = bounds[:, 0], bounds[:, 1]
    pts = (time >= lower) & (time < upper)
    idx = np.arange(len(lower))[pts]
    if len(idx) > 0:
        return idx[0]


def convert_units(values, old_unit, new_unit):
    if isinstance(values, list):
        values = np.asarray(values)
    return cf_units.Unit(old_unit).convert(values, new_unit)


class Messenger(object):
    def __init__(self, figure):
        self.figure = figure
        self.label = bokeh.models.Label(
            x=0,
            y=0,
            x_units="screen",
            y_units="screen",
            text_align="center",
            text_baseline="top",
            text="",
            text_font_style="bold")
        figure.add_layout(self.label)
        custom_js = bokeh.models.CustomJS(
            args=dict(label=self.label), code="""
                label.x = cb_obj.layout_width / 2
                label.y = cb_obj.layout_height / 2
            """)
        figure.js_on_change("layout_width", custom_js)
        figure.js_on_change("layout_height", custom_js)

    @property
    def text(self):
        return self.label.text

    @text.setter
    def text(self, value):
        self.label.text = value


class Title(object):
    def __init__(self, figure):
        self.caption = bokeh.models.Label(
            x=0,
            y=0,
            x_units="screen",
            y_units="screen",
            y_offset=-10,
            text_font_size="12pt",
            text_font_style="bold",
            text_align="center",
            text_baseline="top",
            text="")
        figure.add_layout(self.caption)
        custom_js = bokeh.models.CustomJS(
            args=dict(caption=self.caption), code="""
                caption.x = cb_obj.layout_width / 2
                caption.y = cb_obj.layout_height
            """)
        figure.js_on_change("layout_width", custom_js)
        figure.js_on_change("layout_height", custom_js)

    @property
    def text(self):
        return self.caption.text

    @text.setter
    def text(self, value):
        self.caption.text = value


def full_screen_figure(
        lon_range=(-180, 180),
        lat_range=(-80, 80)):
    x_range, y_range = transform(
        lon_range,
        lat_range,
        cartopy.crs.PlateCarree(),
        cartopy.crs.Mercator.GOOGLE)
    figure = bokeh.plotting.figure(
        sizing_mode="stretch_both",
        x_range=x_range,
        y_range=y_range,
        x_axis_type="mercator",
        y_axis_type="mercator",
        active_scroll="wheel_zoom")
    figure.toolbar_location = None
    figure.axis.visible = False
    figure.min_border = 0
    tile = bokeh.models.WMTSTileSource(
        url='https://c.tile.openstreetmap.org/{Z}/{X}/{Y}.png',
        attribution="&copy; <a href='http://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors"
    )
    figure.add_tile(tile)
    return figure


def transform(x, y, src_crs, dst_crs):
    x, y = np.asarray(x), np.asarray(y)
    xt, yt, _ = dst_crs.transform_points(src_crs, x.flatten(), y.flatten()).T
    return xt, yt


def stretch_y(uneven_y):
    """Mercator projection stretches longitude spacing

    To remedy this effect an even-spaced resampling is performed
    in the projected space to make the pixels and grid line up

    .. note:: This approach assumes the grid is evenly spaced
              in longitude/latitude space prior to projection
    """
    if isinstance(uneven_y, list):
        uneven_y = np.asarray(uneven_y, dtype=np.float)
    even_y = np.linspace(
        uneven_y.min(), uneven_y.max(), len(uneven_y),
        dtype=np.float)
    index = np.arange(len(uneven_y), dtype=np.float)
    index_function = scipy.interpolate.interp1d(uneven_y, index)
    index_fractions = index_function(even_y)

    def wrapped(values, axis=0):
        if isinstance(values, list):
            values = np.asarray(values, dtype=np.float)
        assert values.ndim == 2, "Can only stretch 2D arrays"
        msg = "{} != {} do not match".format(values.shape[axis], len(uneven_y))
        assert values.shape[axis] == len(uneven_y), msg
        if axis == 0:
            i = index_fractions
            j = np.arange(values.shape[1], dtype=np.float)
        elif axis == 1:
            i = np.arange(values.shape[0], dtype=np.float)
            j = index_fractions
        else:
            raise Exception("Can only handle axis 0 or 1")
        return scipy.ndimage.map_coordinates(
            values,
            np.meshgrid(i, j, indexing="ij"),
            order=1)
    return wrapped


if __name__.startswith("bk"):
    main()
