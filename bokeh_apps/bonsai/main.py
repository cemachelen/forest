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
        value = action["type"] == "ACTIVATE"
        if action["category"] in state:
            state[action["category"]]["active"] = value
        else:
            state[action["category"]] = {"active": value}
    elif action["type"] == "REQUEST":
        if action["status"] == "succeed":
            state["listing"] = False
            response = action["response"]
            if "files" in state:
                state["files"][response["key"]] = response["files"]
            else:
                state["files"] = {response["key"]: response["files"]}
        else:
            state["listing"] = True
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
    def set_observation(text):
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
    @staticmethod
    def started():
        return {
            "type": "REQUEST",
            "status": "active"}

    @staticmethod
    def finished(response):
        return {
            "type": "REQUEST",
            "status": "succeed",
            "response": response}

    @staticmethod
    def failed():
        return {
            "type": "REQUEST",
            "status": "fail"}


class Application(object):
    def __init__(self, config):
        self.store = Store(reducer, {"model": {"active": True}})
        self.unsubscribe = self.store.subscribe(self.state_change)
        self.document = bokeh.plotting.curdoc()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.figures = {
            "map": full_screen_figure(
                lon_range=config.lon_range,
                lat_range=config.lat_range)
        }
        self.toolbar_box = bokeh.models.ToolbarBox(
            toolbar=self.figures["map"].toolbar,
            toolbar_location="below")

        self.title = Title(self.figures["map"])
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
        self.dropdowns["observation"].on_click(self.on_click(Action.set_observation))
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
        self.pseudo_request_submitted = False

    def on_tab_change(self, attr, old, new):
        if new == 0:
            self.store.dispatch(Action.deactivate("observation"))
            self.store.dispatch(Action.activate("model"))
        else:
            self.store.dispatch(Action.deactivate("model"))
            self.store.dispatch(Action.activate("observation"))

    def state_change(self):
        state = self.store.state
        print(state)

        if "model" in state:
            listing = state.get("listing", False)
            active = state["model"].get("active", False)
            name = state["model"].get("name", "")
            files = state.get("files", {})
            if active:
                pattern = self.patterns[name]
                if (name not in files) and not listing:
                    self.submit(self.list_files(name, pattern))

        self.render(state)

    @timed
    def list_files(self, key, pattern):
        def task():
            files = glob.glob(pattern)
            return {key: files}
        return task

    def submit(self, blocking_task):
        self.document.add_next_tick_callback(
            partial(self.unlocked_task, blocking_task))
        self.store.dispatch(Request.started())

    @gen.coroutine
    @without_document_lock
    def unlocked_task(self, blocking_task):
        response = yield self.executor.submit(blocking_task)
        self.document.add_next_tick_callback(partial(self.completed, response))

    @gen.coroutine
    def completed(self, response):
        self.store.dispatch(Request.finished(response))

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

    def title_text(self, state):
        parts = []
        for category in ["model", "observation"]:
            if category in state:
                if state[category].get("active", False):
                    parts.append(state[category].get("name", ""))
        if "valid_date" in state:
            parts.append(state["valid_date"].strftime("%Y-%m-%d %H:%M"))
        return " ".join(parts)


def main():
    env = parse_env()
    if env.config_file is None:
        config = Config()
    else:
        config = Config.load(env.config_file)

    application = Application(config)
    figure = application.figures["map"]

    document = bokeh.plotting.curdoc()
    messenger = Messenger(figure)

    # async_image = AsyncImage(
    #     document,
    #     figure,
    #     messenger,
    #     executor)
    # plot_stream.subscribe(lambda args: async_image.update(*args))

    # GPM
    # gpm = GPM(async_image)

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

    document.add_root(figure)
    document.add_root(application.toolbar_box)
    document.add_root(controls)
    document.add_root(height_controls)
    document.title = config.title


def pluck(items, key):
    return [item[key] for item in items]


def as_menu(items):
    return [(item, item) for item in items]


class GPM(object):
    def __init__(self, async_image):
        self.async_image = async_image
        self.figure = time_figure()
        self.figure.title.text = "Observation times"
        self.source = time_source(self.figure)
        self.source.selected.on_change("indices", self.on_indices)
        self._paths = {}
        self._format = "%Y%m%dT%H%M%S"

    def load_times(self, path):
        if path is None:
            return
        if "gpm" not in path:
            return
        with netCDF4.Dataset(path) as dataset:
            times = load_times(dataset)
            data = {
                "x": times,
                "y": np.ones(len(times))
            }
        self.source.stream(data)

        # Update catalogue
        for i, t in enumerate(times):
            k = t.strftime(self._format)
            self._paths[k] = (path, i)

    def on_indices(self, attr, old, new):
        for i in new:
            time = self.source.data["x"][i]
            self.load_image(time)

    def load_image(self, time):
        key = time.strftime(self._format)
        path, index = self._paths[key]
        self.async_image.update(path, time)


def time_figure():
    figure = bokeh.plotting.figure(
        x_axis_type="datetime",
        plot_width=300,
        plot_height=100,
        toolbar_location="below",
        background_fill_alpha=0,
        border_fill_alpha=0,
        tools="xwheel_zoom,ywheel_zoom,xpan,ypan,reset,tap",
        active_scroll="xwheel_zoom",
        active_drag="xpan",
        active_tap="tap",
    )
    figure.outline_line_alpha = 0
    figure.grid.visible = False
    figure.yaxis.visible = False
    figure.toolbar.logo = None
    return figure


def time_source(figure):
    source = bokeh.models.ColumnDataSource({
        "x": [],
        "y": []
    })
    renderer = figure.square(
        x="x", y="y", size=10, source=source)
    hover_tool = bokeh.models.HoverTool(
        toggleable=False,
        tooltips=[
            ('date', '@x{%Y-%m-%d %H:%M}')
        ],
        formatters={
            'x': 'datetime'
        }
    )
    glyph = bokeh.models.Square(
        fill_color="red",
        line_color="black")
    renderer.hover_glyph = glyph
    renderer.selection_glyph = glyph
    renderer.nonselection_glyph = bokeh.models.Square(
        fill_color="white",
        line_color="black")
    figure.add_tools(hover_tool)
    return source


class Observable(object):
    def __init__(self):
        self.callbacks = defaultdict(list)

    def on_change(self, attr, callback):
        self.callbacks[attr].append(callback)

    def trigger(self, attr, value):
        for cb in self.callbacks[attr]:
            cb(attr, None, value)

    def notify(self, new):
        attr, old = None, None
        for cbs in self.callbacks.values():
            for cb in cbs:
                cb(attr, old, new)


def load_times(dataset):
    for name in ["time_2", "time"]:
        if name not in dataset.variables:
            continue
        units = dataset.variables[name].units
        values = dataset.variables[name][:]
        return netCDF4.num2date(values, units=units)


def find_forecast(paths, run_date):
    """Find file, index and model date associated with valid date"""
    run_dates = [parse_time(path) for path in paths]
    return find_by_date(paths, most_recent(run_dates, run_date))


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


class AsyncImage(object):
    def __init__(self, document, figure, messenger, executor):
        self.document = document
        self.figure = figure
        self.messenger = messenger
        self.executor = executor
        self.previous_tick = None
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

    def update(self, path, valid_time):
        if path is None:
            self.document.add_next_tick_callback(
                partial(self.render, self.empty_data))
            self.document.add_next_tick_callback(
                self.messenger.on_file_not_found)
            return
        print("Image: {} {}".format(path, valid_time))
        if self.previous_tick is not None:
            try:
                self.document.remove_next_tick_callback(self.previous_tick)
            except ValueError:
                pass
        blocking_task = partial(self.load, path, valid_time)
        self.previous_tick = self.document.add_next_tick_callback(
            partial(self.unlocked_task, blocking_task))

    @gen.coroutine
    @without_document_lock
    def unlocked_task(self, blocking_task):
        self.document.add_next_tick_callback(self.messenger.on_load)
        data = yield self.executor.submit(blocking_task)
        self.document.add_next_tick_callback(partial(self.render, data))
        self.document.add_next_tick_callback(self.messenger.on_complete)

    def load(self, path, valid_time):
        with netCDF4.Dataset(path) as dataset:
            bounds, units = ui.time_bounds(dataset)
            bounds = netCDF4.num2date(bounds, units=units)
            index = time_index(bounds, valid_time)
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

    @gen.coroutine
    def render(self, data):
        self.source.data = data


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

    def echo(self, message):
        @gen.coroutine
        def method():
            self.render(message)
        return method

    @gen.coroutine
    def erase(self):
        self.label.text = ""

    @gen.coroutine
    def on_load(self):
        self.render("Loading...")

    @gen.coroutine
    def on_complete(self):
        self.render("")

    @gen.coroutine
    def on_file_not_found(self):
        self.render("File not available")

    def render(self, text):
        self.label.text = text


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
