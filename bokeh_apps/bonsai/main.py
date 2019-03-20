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


def main():
    env = parse_env()
    if env.config_file is None:
        config = Config()
    else:
        config = Config.load(env.config_file)

    figure = full_screen_figure(
        lon_range=config.lon_range,
        lat_range=config.lat_range)
    toolbar_box = bokeh.models.ToolbarBox(
        toolbar=figure.toolbar,
        toolbar_location="below")

    document = bokeh.plotting.curdoc()
    messenger = Messenger(figure)
    executor = ThreadPoolExecutor(max_workers=2)

    model_dropdown = bokeh.models.Dropdown(
        label="Configuration",
        menu=as_menu(pluck(config.models, "name")))
    model_dropdown.on_click(select(model_dropdown))

    obs_dropdown = bokeh.models.Dropdown(
        label="Instrument/satellite",
        menu=as_menu(pluck(config.observations, "name")))
    obs_dropdown.on_click(select(obs_dropdown))

    models = rx.Stream()
    model_dropdown.on_click(models.emit)

    table = file_patterns(
        config.models,
        env.directory)
    patterns = rx.map(models, lambda v: table[v])
    patterns.subscribe(print)

    async_image = AsyncImage(
        document,
        figure,
        messenger,
        executor)
    # plot_stream.subscribe(lambda args: async_image.update(*args))

    title = Title(figure)
    models.subscribe(lambda x: title.update({"model": x}))

    # Field drop down
    field_dropdown = bokeh.models.Dropdown(
        label="Field",
        menu=[
            ("Precipitation", "precipitation"),
            ("Outgoing longwave radiation (OLR)", "olr"),
        ])
    field_dropdown.on_click(select(field_dropdown))

    # Overlay choices
    overlay_checkboxes = bokeh.models.CheckboxGroup(
        labels=["MSLP", "Wind vectors"],
        inline=True)

    # GPM
    gpm = GPM(async_image)

    level_selector = LevelSelector()

    datetime_picker = bonsai.DatetimePicker()

    def callback(attr, old, new):
        print(attr, old, new)

    valid_dates = rx.Stream()
    datetime_picker.on_change("value", rx.on_change(valid_dates))
    datetime_picker.date_picker.title = "Valid date"

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

    minus, div, plus = (
        bokeh.models.Button(label="-", width=135),
        bokeh.models.Div(text="", width=10),
        bokeh.models.Button(label="+", width=135))
    tabs = bokeh.models.Tabs(tabs=[
        bokeh.models.Panel(child=bokeh.layouts.column(
            bokeh.layouts.row(
                bokeh.layouts.column(minus),
                bokeh.layouts.column(div),
                bokeh.layouts.column(plus)),
            model_dropdown,
            field_dropdown,
            overlay_checkboxes,
        ), title="Model"),
        bokeh.models.Panel(child=bokeh.layouts.column(
            obs_dropdown,
        ), title="Observation")])
    def on_change(attr, old, new):
        if new == 0:
            print("model")
        else:
            print("observation")
    tabs.on_change("active", on_change)
    controls = bokeh.layouts.column(
        datetime_picker.date_picker,
        button_row,
        datetime_picker.hour_slider,
        tabs,
        name="controls")

    height_controls = bokeh.layouts.column(
        level_selector.slider,
        name="height")

    document.add_root(figure)
    document.add_root(toolbar_box)
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

    def update(self, state):
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
