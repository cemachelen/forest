import unittest
import json
import bokeh.plotting
import bokeh.models
import bokeh.palettes


class TestRDT(unittest.TestCase):
    def setUp(self):
        self.path = "samples/RDT_features_eastafrica_201903281445.json"

    def test_load(self):
        with open(self.path) as stream:
            data = json.load(stream)
        print(data.keys())
        self.assertTrue(False)

    def test_plotting_rdt_geo_json(self):
        figure = bokeh.plotting.figure()
        with open(self.path) as stream:
            json_string = stream.read()
        source = bokeh.models.GeoJSONDataSource(geojson=json_string)
        color_mapper = bokeh.models.LinearColorMapper(
                palette=bokeh.palettes.Viridis6)
        renderer = figure.multi_line(
                xs="xs",
                ys="ys",
                line_width=2,
                line_color={'field': 'PhaseLife', 'transform': color_mapper},
                source=source)
