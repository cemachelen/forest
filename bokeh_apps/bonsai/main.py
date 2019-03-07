"""Minimalist implementation of FOREST"""
from collections import defaultdict
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

    def model_pattern(self, name):
        for model in self.models:
            if name == model["name"]:
                return model["pattern"]

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

    file_system = FileSystem(
        models=config.models,
        model_dir=config.model_dir)
    file_system.on_change("path", state.on_change("path"))
    state.register(file_system, "model")

    figure = full_screen_figure(
        lon_range=config.lon_range,
        lat_range=config.lat_range)
    toolbar_box = bokeh.models.ToolbarBox(
        toolbar=figure.toolbar,
        toolbar_location="below")

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

    date_picker = bokeh.models.DatePicker()
    date_picker.on_change("value", state.on_change("date"))

    document = bokeh.plotting.curdoc()
    messenger = Messenger(figure)
    image = AsyncImage(
        document,
        figure,
        messenger)
    state.register(image, "path")

    title = Title(figure)
    state.register(title, "model")
    state.register(title, "date")

    document.add_root(figure)
    document.add_root(toolbar_box)
    document.add_root(bokeh.layouts.column(
        date_picker,
        dropdown,
        name="controls"))
    document.title = config.title


class FileSystem(object):
    def __init__(self,
                 models,
                 model_dir=None):
        self.models = models
        self.model_dir = model_dir
        self.callbacks = []

    def on_change(self, attr, callback):
        self.callbacks.append(callback)

    def notify(self, state):
        if "model" not in state:
            return
        pattern = self.full_pattern(state["model"])
        path = sorted(glob.glob(pattern))[-1]
        for cb in self.callbacks:
            cb("path", None, path)

    def full_pattern(self, name):
        for model in self.models:
            if name == model["name"]:
                pattern = model["pattern"]
                if self.model_dir is None:
                    return pattern
                else:
                    return os.path.join(self.model_dir, pattern)


class AsyncImage(object):
    def __init__(self, document, figure, messenger):
        self.document = document
        self.figure = figure
        self.messenger = messenger
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.source = bokeh.models.ColumnDataSource({
            "x": [],
            "y": [],
            "dw": [],
            "dh": [],
            "image": []
        })
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
        print("Image: {}".format(state["path"]))
        blocking_task = partial(self.load, state["path"])
        self.document.add_next_tick_callback(
            partial(self.unlocked_task, blocking_task))

    @gen.coroutine
    @without_document_lock
    def unlocked_task(self, blocking_task):
        self.document.add_next_tick_callback(self.messenger.on_load)
        data = yield self.executor.submit(blocking_task)
        self.document.add_next_tick_callback(partial(self.render, data))
        self.document.add_next_tick_callback(self.messenger.on_complete)

    def load(self, path):
        i = 0
        with netCDF4.Dataset(path) as dataset:
            lons = dataset.variables["longitude_0"][:]
            lats = dataset.variables["latitude_0"][:]
            try:
                var = dataset.variables["stratiform_rainfall_rate"]
            except KeyError:
                var = dataset.variables["precipitation_flux"]
            values = var[i]
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
            text="",
            text_font_style="bold")
        self.figure.add_layout(self.label)

    @gen.coroutine
    def on_load(self):
        self.render("Loading...")

    @gen.coroutine
    def on_complete(self):
        self.render("")

    def render(self, text):
        self.label.x = (
            self.figure.x_range.start +
            self.figure.x_range.end) / 2
        dy = 0.5
        self.label.y = (
            (1 - dy) * self.figure.y_range.start +
            dy * self.figure.y_range.end)
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
        if "date" in state:
            words.append("{:%Y-%m-%d %H:%M}".format(state["date"]))
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

    def on(self, attr):
        def callback(value):
            self.announce(attr, value)
        return callback

    def on_change(self, state_attr):
        def callback(attr, old, new):
            self.announce(state_attr, new)
        return callback

    def register(self, subscriber, state_attr=None):
        if state_attr is None:
            self.subscribers.append(subscriber)
        else:
            self.special_subscribers[state_attr].append(subscriber)

    def announce(self, attr, value):
        self.state = dict(self.state)
        self.state[attr] = value
        for s in self.subscribers:
            s.notify(self.state)
        for s in self.special_subscribers[attr]:
            s.notify(self.state)


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
