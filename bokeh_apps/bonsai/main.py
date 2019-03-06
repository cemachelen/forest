"""Minimalist implementation of FOREST"""
import os
import yaml
import bokeh.plotting
import bokeh.models
import cartopy
import numpy as np


class Config(object):
    def __init__(self,
                 title="Bonsai - miniature Forest",
                 lon_range=None,
                 lat_range=None,
                 models=None):
        def assign(value, default):
            return default if value is None else value
        self.title = title
        self.lon_range = assign(lon_range, [-180, 180])
        self.lat_range = assign(lat_range, [-80, 80])
        self.models = assign(models, [])

    @property
    def model_names(self):
        return [model["name"] for model in self.models]

    def model_pattern(self, name):
        for model in self.models:
            if name == model["name"]:
                return model["pattern"]

    @classmethod
    def load(cls, path):
        with open(path) as stream:
            kwargs = yaml.load(stream)
        return cls(**kwargs)


class Environment(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


def parse_env():
    config_file = os.environ.get("FOREST_CONFIG", None)
    return Environment(config_file=config_file)


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

    def select(dropdown):
        def on_click(value):
            dropdown.label = value
        return on_click

    def on_model(config):
        def on_click(value):
            print(value, config.model_pattern(value))
        return on_click

    menu = [(name, name) for name in config.model_names]
    dropdown = bokeh.models.Dropdown(
        label="Select model",
        menu=menu)
    dropdown.on_click(select(dropdown))
    dropdown.on_click(on_model(config))

    document = bokeh.plotting.curdoc()
    document.add_root(figure)
    document.add_root(toolbar_box)
    document.add_root(bokeh.layouts.column(
        dropdown,
        name="controls"))
    document.title = config.title


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


if __name__.startswith("bk"):
    main()
