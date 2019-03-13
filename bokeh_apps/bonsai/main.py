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

import sys
sys.path.insert(0, os.path.dirname(__file__))
import rx
from util import timed


class Config(object):
    def __init__(self,
                 title="Bonsai - miniature Forest",
                 lon_range=None,
                 lat_range=None,
                 models=None,
                 model_dir=None):
        def assign(value, default):
            return default if value is None else value
        self.title = title
        self.lon_range = assign(lon_range, [-180, 180])
        self.lat_range = assign(lat_range, [-80, 80])
        self.models = assign(models, [])
        self.model_dir = model_dir

    @classmethod
    def merge(cls, *dicts):
        settings = {}
        for d in dicts:
            settings.update(d)
        return Config(**settings)

    @property
    def model_names(self):
        return [model["name"] for model in self.models]

    @classmethod
    def load(cls, path):
        return cls(**cls.load_dict(path))

    @staticmethod
    def load_dict(path):
        with open(path) as stream:
            kwargs = yaml.load(stream)
        return kwargs


class Environment(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


def parse_env():
    config_file = os.environ.get("FOREST_CONFIG", None)
    model_dir = os.environ.get("FOREST_MODEL_DIR", None)
    return Environment(
        config_file=config_file,
        model_dir=model_dir)


def main():
    env = parse_env()
    dicts = [dict(model_dir=env.model_dir)]
    if env.config_file is not None:
        dicts.append(Config.load_dict(env.config_file))
    config = Config.merge(*dicts)

    state = State()
    state.register(Echo())

    figure = full_screen_figure(
        lon_range=config.lon_range,
        lat_range=config.lat_range)
    toolbar_box = bokeh.models.ToolbarBox(
        toolbar=figure.toolbar,
        toolbar_location="below")

    document = bokeh.plotting.curdoc()
    messenger = Messenger(figure)
    executor = ThreadPoolExecutor(max_workers=2)

    async_glob = AsyncGlob(
        document,
        messenger,
        executor)

    file_system = FileSystem(
        models=config.models,
        model_dir=config.model_dir,
        async_glob=async_glob)
    file_system.on_change("path", state.on_change("path"))
    state.register(file_system, "model")
    state.register(file_system, "run_date")

    @timed
    def process_files(pattern):
        paths = glob.glob(pattern)
        dates = [model_run_time(path) for path in paths]
        return sorted(dates)

    def select(dropdown):
        def on_click(value):
            dropdown.label = value
        return on_click

    menu = [(name, name) for name in config.model_names]
    dropdown = bokeh.models.Dropdown(
        label="Select model",
        menu=menu)
    dropdown.on_click(select(dropdown))
    dropdown.on_click(state.on("model"))

    model_names = rx.Stream()
    dropdown.on_click(model_names.emit)

    table = file_patterns(
        config.models,
        config.model_dir)
    patterns = rx.map(model_names, lambda v: table[v])
    run_dates = rx.map(patterns, process_files)

    time_controls = TimeControls()
    time_controls.on_change("datetime", state.on_change("run_date"))

    forecast_tool = ForecastTool()
    model_run = ModelRun()
    run_dates.subscribe(model_run.on_run_times)

    model_run.on_change("run_date", state.on_change("run_date"))
    forecast_tool.on_change("run_date", state.on_change("run_date"))
    forecast_tool.on_change("run_date", model_run.on_run_date)
    forecast_tool.on_change("valid_date", state.on_change("valid_date"))
    forecast_tool.on_change("index", state.on_change("index"))

    state.register(forecast_tool, "path")

    image = AsyncImage(
        document,
        figure,
        messenger,
        executor)
    state.register(image, "path")
    state.register(image, "index")

    title = Title(figure)
    state.register(title, "model")
    state.register(title, "run_date")
    state.register(title, "valid_date")

    document.add_root(figure)
    document.add_root(toolbar_box)
    document.add_root(bokeh.layouts.column(
        dropdown,
        time_controls.date_picker,
        time_controls.radio_group,
        model_run.figure,
        model_run.button_row,
        forecast_tool.figure,
        forecast_tool.button_row,
        name="controls"))
    document.title = config.title


def navigation_figure(plot_height=240, toolbar_location="below"):
    figure = bokeh.plotting.figure(
        x_axis_type="datetime",
        plot_height=plot_height,
        plot_width=290,
        tools="xwheel_zoom,ywheel_zoom,xpan,ypan,reset",
        active_scroll="xwheel_zoom",
        active_drag="xpan",
        toolbar_location=toolbar_location)
    figure.background_fill_alpha = 0.8
    figure.border_fill_alpha = 0
    figure.yaxis.visible = False
    figure.toolbar.logo = None
    return figure


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


class ModelRun(Observable):
    def __init__(self):
        self.figure = navigation_figure(
            plot_height=90,
            toolbar_location=None)
        self.figure.title.text = "Run date"
        self.source = bokeh.models.ColumnDataSource({
            "x": [],
            "y": []
        })
        renderer = self.figure.square(
            x="x",
            y="y",
            source=self.source)
        renderer.selection_glyph = bokeh.models.Square(
            fill_color="red",
            line_color="red")
        hover_tool = bokeh.models.HoverTool(
            toggleable=False,
            tooltips=[
                ('run start', '@x{%Y-%m-%d %H:%M}')
            ],
            formatters={
                'x': 'datetime'
            },
            renderers=[renderer]
        )
        self.figure.add_tools(hover_tool)
        tap_tool = bokeh.models.TapTool()
        self.figure.add_tools(tap_tool)
        self.source.selected.on_change("indices", self.on_selection)

        # Button row
        width = 50
        plus = bokeh.models.Button(label="+", width=width)
        plus.on_click(self.on_plus)
        minus = bokeh.models.Button(label="-", width=width)
        minus.on_click(self.on_minus)
        self.button_row = bokeh.layouts.row(plus, minus)
        super().__init__()

    def on_selection(self, attr, old, new):
        if len(new) == 0:
            return
        i = new[0]
        run_date = self.source.data["x"][i]
        self.trigger("run_date", run_date)

    def on_plus(self):
        if len(self.source.selected.indices) == 0:
            return
        i = self.source.selected.indices[0] + 1
        self.source.selected.indices = [i]

    def on_minus(self):
        if len(self.source.selected.indices) == 0:
            return
        i = self.source.selected.indices[0] - 1
        self.source.selected.indices = [i]

    def on_run_date(self, attr, old, new):
        run_times = list(self.source.data["x"])
        if new in run_times:
            i = run_times.index(new)
            self.source.selected.indices = [i]

    def on_run_times(self, run_times):
        self.source.data = {
            "x": run_times,
            "y": np.zeros(len(run_times))
        }


class ForecastTool(Observable):
    def __init__(self):
        self.figure = navigation_figure()
        self.figure.title.text = "Forecast date"
        self.source = bokeh.models.ColumnDataSource({
            "top": [],
            "bottom": [],
            "left": [],
            "right": [],
            "start": [],
            "index": []
        })
        renderer = self.figure.quad(
            top="top",
            bottom="bottom",
            left="left",
            right="right",
            source=self.source)
        renderer.hover_glyph = bokeh.models.Quad(
            fill_color="red",
            line_color="red")
        renderer.selection_glyph = bokeh.models.Quad(
            fill_color="red",
            line_color="red")
        hover_tool = bokeh.models.HoverTool(
            toggleable=False,
            tooltips=[
                ('time', '@left{%Y-%m-%d %H:%M}'),
                ('length', 'T@bottom{%+i} to T@top{%+i}'),
                ('run start', '@start{%Y-%m-%d %H:%M}')
            ],
            formatters={
                'left': 'datetime',
                'bottom': 'printf',
                'top': 'printf',
                'start': 'datetime'},
            renderers=[renderer]
        )
        self.figure.add_tools(hover_tool)
        tap_tool = bokeh.models.TapTool()
        self.figure.add_tools(tap_tool)
        self.starts = set()
        self.source.selected.on_change("indices", self.on_selection)

        # Button row
        width = 50
        plus = bokeh.models.Button(label="+", width=width)
        plus.on_click(self.on_plus)
        minus = bokeh.models.Button(label="-", width=width)
        minus.on_click(self.on_minus)
        self.button_row = bokeh.layouts.row(plus, minus)
        super().__init__()

    def on_selection(self, attr, old, new):
        if len(new) == 0:
            return
        i = new[0]
        self.trigger("run_date", self.source.data["start"][i])
        self.trigger("valid_date", self.source.data["left"][i])
        self.trigger("index", self.source.data["index"][i])

    def on_plus(self):
        if len(self.source.selected.indices) == 0:
            return
        i = self.source.selected.indices[0] + 1
        self.source.selected.indices = [i]

    def on_minus(self):
        if len(self.source.selected.indices) == 0:
            return
        i = self.source.selected.indices[0] - 1
        self.source.selected.indices = [i]

    def on_run_date(self, attr, old, new):
        pass

    def notify(self, state):
        path = state["path"]
        if path is None:
            return
        with netCDF4.Dataset(path) as dataset:
            units = dataset.variables["time_2"].units
            bounds = dataset.variables["time_2_bnds"][:]
        data = self.data(bounds, units)
        start = data["start"][0]
        if start not in self.starts:
            self.source.stream(data)
            self.starts.add(start)

    @staticmethod
    def data(bounds, units):
        """Helper to convert from netCDF4 values to bokeh source"""
        top = bounds[:, 1] - bounds[0, 0]
        bottom = bounds[:, 0] - bounds[0, 0]
        left = netCDF4.num2date(bounds[:, 0], units=units)
        right = netCDF4.num2date(bounds[:, 1], units=units)
        start = np.full(len(left), left[0], dtype=object)
        index = np.arange(bounds.shape[0])
        return {
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "start": start,
            "index": index
        }


class TimeControls(Observable):
    def __init__(self):
        self._date = None
        self._time = None
        self.date_picker = bokeh.models.DatePicker()
        self.date_picker.on_change("value", self.on_date)
        self.radio_group = bokeh.models.RadioGroup(
            labels=["00:00", "12:00"],
            inline=True,
            active=0,
            css_classes=[
                "bonsai-mg-lf-10",
                "bonsai-lh-24"])
        self.radio_group.on_change("active", self.on_time)
        super().__init__()

    def on_time(self, attr, old, new):
        if new == 0:
            hour = 0
        else:
            hour = 12
        self._time = dt.time(hour)
        self.announce()

    def on_date(self, attr, old, new):
        self._date = new
        self.announce()

    def announce(self):
        for value in [self._date, self._time]:
            if value is None:
                return
        self.notify(dt.datetime(
            self._date.year,
            self._date.month,
            self._date.day,
            self._time.hour))


class AsyncGlob(object):
    def __init__(self, document, messenger, executor):
        self.document = document
        self.messenger = messenger
        self.executor = executor

    def glob(self, pattern, cb):
        self.document.add_next_tick_callback(
            partial(self.task, pattern, cb))

    @gen.coroutine
    @without_document_lock
    def task(self, pattern, cb):
        self.document.add_next_tick_callback(self.messenger.echo("Searching..."))
        files = yield self.executor.submit(partial(glob.glob, pattern))
        self.document.add_next_tick_callback(partial(cb, files))


class FileSystem(object):
    def __init__(self,
                 models=None,
                 model_dir=None,
                 async_glob=None):
        self.async_glob = async_glob
        if models is None:
            models = []
        self.models = models
        self.model_dir = model_dir
        self.callbacks = []

    def on_change(self, attr, callback):
        self.callbacks.append(callback)

    def notify(self, state):
        for attr in ["model", "run_date"]:
            if attr not in state:
                return
        model, date = state["model"], state["run_date"]
        pattern = self.full_pattern(model)

        def glob_callback(paths):
            path = self.find_file(paths, date)
            for cb in self.callbacks:
                cb("path", None, path)
        self.async_glob.glob(pattern, glob_callback)

    def full_pattern(self, name):
        for model in self.models:
            if name == model["name"]:
                pattern = model["pattern"]
                if self.model_dir is None:
                    return pattern
                else:
                    return os.path.join(self.model_dir, pattern)

    @staticmethod
    def find_file(paths, date):
        """Search for file matching date"""
        for path in paths:
            if model_run_time(path) == date:
                return path


def file_patterns(models, directory=None):
    table = {}
    for entry in models:
        name, pattern = entry["name"], entry["pattern"]
        if directory is not None:
            pattern = os.path.join(directory, pattern)
        table[name] = pattern
    return table


def model_run_time(path):
    file_name = os.path.basename(path)
    timestamp = re.search("[0-9]{8}T[0-9]{4}Z", file_name).group()
    return dt.datetime.strptime(timestamp, "%Y%m%dT%H%MZ")


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

    def notify(self, state):
        if "path" not in state:
            return
        if "index" not in state:
            return
        path = state["path"]
        index = state["index"]
        if path is None:
            self.document.add_next_tick_callback(
                partial(self.render, self.empty_data))
            self.document.add_next_tick_callback(
                self.messenger.on_file_not_found)
            return
        print("Image: {} {}".format(path, index))
        if self.previous_tick is not None:
            try:
                self.document.remove_next_tick_callback(self.previous_tick)
            except ValueError:
                print("Previous callback either already started or not added")
        blocking_task = partial(self.load, path, index)
        self.previous_tick = self.document.add_next_tick_callback(
            partial(self.unlocked_task, blocking_task))

    @gen.coroutine
    @without_document_lock
    def unlocked_task(self, blocking_task):
        self.document.add_next_tick_callback(self.messenger.on_load)
        data = yield self.executor.submit(blocking_task)
        self.document.add_next_tick_callback(partial(self.render, data))
        self.document.add_next_tick_callback(self.messenger.on_complete)

    def load(self, path, index):
        with netCDF4.Dataset(path) as dataset:
            lons = dataset.variables["longitude_0"][:]
            lats = dataset.variables["latitude_0"][:]
            try:
                var = dataset.variables["stratiform_rainfall_rate"]
            except KeyError:
                var = dataset.variables["precipitation_flux"]
            values = var[index]
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

    def notify(self, state):
        words = []
        if "model" in state:
            words.append(state["model"])
        for date_attr in ["run_date", "valid_date"]:
            if date_attr in state:
                words.append("{:%Y-%m-%d %H:%M}".format(state[date_attr]))
        text = " ".join(words)
        self.render(text)

    def render(self, text):
        self.caption.text = text


class Echo(object):
    def notify(self, state):
        print(state)


class State(object):
    def __init__(self):
        self.state = {}
        self.subscribers = []
        self.special_subscribers = defaultdict(list)
        self.callbacks = defaultdict(list)

    def on(self, attr):
        def callback(value):
            self.trigger(attr, value)
        return callback

    def on_change(self, state_attr):
        def callback(attr, old, new):
            self.trigger(state_attr, new)
        return callback

    def register(self, subscriber, state_attr=None):
        if state_attr is None:
            self.subscribers.append(subscriber)
        else:
            self.special_subscribers[state_attr].append(subscriber)

    def add_callback(self, attr, callback):
        self.callbacks[attr].append(callback)

    def trigger(self, attr, value):
        self.state = dict(self.state)
        self.state[attr] = value
        for s in self.subscribers:
            s.notify(self.state)
        for s in self.special_subscribers[attr]:
            s.notify(self.state)
        for cb in self.callbacks[attr]:
            cb(self.state[attr])


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
