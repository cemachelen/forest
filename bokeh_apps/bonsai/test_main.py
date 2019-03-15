# pylint: disable=missing-docstring, invalid-name
import unittest
import unittest.mock
import datetime as dt
import os
import yaml
import bokeh.plotting
import main
import numpy as np
import netCDF4


class TestEnvironment(unittest.TestCase):
    def tearDown(self):
        for variable in ["FOREST_MODEL_DIR"]:
            if variable in os.environ:
                del os.environ[variable]

    def test_parse_env_given_forest_model_dir(self):
        os.environ["FOREST_MODEL_DIR"] = "/some/dir"
        result = main.parse_env().model_dir
        expect = "/some/dir"
        self.assertEqual(expect, result)

    def test_parse_env_default_forest_model_dir(self):
        result = main.parse_env().model_dir
        expect = None
        self.assertEqual(expect, result)


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.path = "test-config.yaml"
        self.config = main.Config()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_merge_configs(self):
        dict_0 = dict(
            lat_range=[-5, 5],
            lon_range=[-180, 180])
        with open(self.path, "w") as stream:
            yaml.dump({"lon_range": [-10, 10]}, stream)
            dict_1 = main.Config.load_dict(self.path)
        merged = main.Config.merge(dict_0, dict_1)
        self.assertEqual(merged.lon_range, [-10, 10])
        self.assertEqual(merged.lat_range, [-5, 5])

    def test_load(self):
        with open(self.path, "w") as stream:
            yaml.dump({"lon_range": [-10, 10]}, stream)
        result = self.config.load(self.path).lon_range
        expect = [-10, 10]
        self.assertEqual(expect, result)

    def test_model_names(self):
        settings = {
            "models": [
                {"name": "A"},
                {"name": "B"}
            ]
        }
        with open(self.path, "w") as stream:
            yaml.dump(settings, stream)
        result = main.Config.load(self.path).model_names
        expect = ["A", "B"]
        self.assertEqual(expect, result)

    def test_model_dir(self):
        self.check_kwarg("model_dir", "/some/dir")

    def test_default_model_dir_returns_none(self):
        self.check_default("model_dir", None)

    def test_default_lon_range(self):
        self.check_default("lon_range", [-180, 180])

    def test_default_lat_range(self):
        self.check_default("lat_range", [-80, 80])

    def test_default_title(self):
        self.check_default("title", "Bonsai - miniature Forest")

    def test_default_models(self):
        self.check_default("models", [])

    def check_default(self, attr, expect):
        result = getattr(main.Config(), attr)
        self.assertEqual(expect, result)

    def check_kwarg(self, attr, expect):
        result = getattr(main.Config(**{attr: expect}), attr)
        self.assertEqual(expect, result)


class TestImage(unittest.TestCase):
    def test_constructor(self):
        executor = None
        document = bokeh.plotting.curdoc()
        figure = bokeh.plotting.figure()
        messenger = main.Messenger(figure)
        image = main.AsyncImage(
            document,
            figure,
            messenger,
            executor)


class TestConvertUnits(unittest.TestCase):
    def test_convert_units(self):
        result = main.convert_units([1], "kg m-2 s-1", "kg m-2 hour-1")
        expect = np.array([3600.])
        np.testing.assert_array_almost_equal(expect, result)


class TestStretchY(unittest.TestCase):
    """Web Mercator projection introduces a y-axis stretching"""
    def test_stretch_y(self):
        values = [[0, 1, 2]]
        uneven_y = [0, 2, 3]
        transform = main.stretch_y(uneven_y)
        result = transform(values, axis=1)
        expect = [[0, 0.75, 2]]
        np.testing.assert_array_almost_equal(expect, result)


class TestParseTime(unittest.TestCase):
    def test_parse_time_given_file_name(self):
        result = main.parse_time("/some/file/takm4p4_20190305T1200Z.nc")
        expect = dt.datetime(2019, 3, 5, 12)
        self.assertEqual(expect, result)

    def test_parse_time_given_different_time(self):
        result = main.parse_time("/some/file/ga6_20180105T0000Z.nc")
        expect = dt.datetime(2018, 1, 5)
        self.assertEqual(expect, result)

    def test_run_time_given_gpm_imerg(self):
        example = "/data/local/frrn/buckets/stephen-sea-public-london/gpm_imerg/gpm_imerg_NRTlate_V05B_20190312_highway_only.nc"
        result = main.parse_time(example)
        expect = dt.datetime(2019, 3, 12)
        self.assertEqual(expect, result)

    def test_find_file_by_date(self):
        paths = [
            "/some/file_20180101T0000Z.nc",
            "/some/file_20180101T1200Z.nc",
            "/some/file_20180102T1200Z.nc"]
        date = dt.datetime(2018, 1, 2, 12)
        result = main.find_by_date(paths, date)
        expect = "/some/file_20180102T1200Z.nc"
        self.assertEqual(expect, result)


@unittest.skip("refactoring")
class TestFileDates(unittest.TestCase):
    def setUp(self):
        self.file_dates = main.FileDates()

    def test_source_data(self):
        result = self.file_dates.source.data
        expect = {
            "x": [],
            "y": []
        }
        self.assertEqual(expect, result)

    def test_selected_indices(self):
        result = self.file_dates.source.selected.indices
        expect = []
        self.assertEqual(expect, result)

    def test_on_dates(self):
        date = dt.datetime(2019, 1, 1)
        self.file_dates.on_dates([date])
        result = self.file_dates.source
        expect = bokeh.models.ColumnDataSource({"x": [date], "y": [0]})
        expect.selected.indices = []
        self.assert_sources_equal(expect, result)
        np.testing.assert_array_equal(
            expect.selected.indices, result.selected.indices)

    def assert_sources_equal(self, expect, result):
        self.assertEqual(set(expect.data.keys()), set(result.data.keys()))
        for k in expect.data.keys():
            np.testing.assert_array_equal(expect.data[k], result.data[k])


class TestFilePatterns(unittest.TestCase):
    def test_file_patterns(self):
        result = main.file_patterns([{
            "name": "A",
            "pattern": "*.nc"
        }], directory="/some/dir")
        expect = {
            "A": "/some/dir/*.nc"
        }
        self.assertEqual(expect, result)


class TestForecastTool(unittest.TestCase):
    def test_data_from_bounds(self):
        units = "hours since 1970-01-01 00:00:00"
        bounds = np.array([[24, 27],
                           [27, 30],
                           [30, 33]], dtype=np.float64)
        result = main.ForecastTool.data(bounds, units)
        expect = {
            "top": [3, 6, 9],
            "bottom": [0, 3, 6],
            "left": [
                dt.datetime(1970, 1, 2, 0),
                dt.datetime(1970, 1, 2, 3),
                dt.datetime(1970, 1, 2, 6),
            ],
            "right": [
                dt.datetime(1970, 1, 2, 3),
                dt.datetime(1970, 1, 2, 6),
                dt.datetime(1970, 1, 2, 9),
            ],
            "start": [
                dt.datetime(1970, 1, 2, 0),
                dt.datetime(1970, 1, 2, 0),
                dt.datetime(1970, 1, 2, 0),
            ],
            "index": [0, 1, 2]
        }
        self.assert_data_equal(expect, result)

    def assert_data_equal(self, expect, result):
        for k, v in expect.items():
            np.testing.assert_array_equal(v, result[k])


class TestTimeTimePlot(unittest.TestCase):
    def test_time_time_plot(self):
        y, m, d = 2019, 1, 1
        start = dt.datetime(y, m, d, 0)
        bounds = np.array(
            [[dt.datetime(y, m, d, 0), dt.datetime(y, m, d, 3)],
             [dt.datetime(y, m, d, 3), dt.datetime(y, m, d, 6)],
             [dt.datetime(y, m, d, 6), dt.datetime(y, m, d, 9)]], dtype=object)
        interval = dt.timedelta(hours=12)
        result = main.time_time_graph(bounds, interval)
        expect = {
            "top": 3 * [start + interval],
            "bottom": 3 * [start],
            "left": bounds[:, 0],
            "right": bounds[:, 1]
        }
        self.assert_data_equal(expect, result)

    def assert_data_equal(self, expect, result):
        for k, v in expect.items():
            np.testing.assert_array_equal(v, result[k])


class TestObservable(unittest.TestCase):
    def test_trigger(self):
        cb = unittest.mock.Mock()
        observable = main.Observable()
        observable.on_change("attr", cb)
        observable.trigger("attr", "value")
        cb.assert_called_once_with("attr", None, "value")

    def test_on_change_selectively_calls_callbacks(self):
        cb = unittest.mock.Mock()
        observable = main.Observable()
        observable.on_change("B", cb)
        observable.trigger("A", "value")
        observable.trigger("B", "value")
        cb.assert_called_once_with("B", None, "value")


class TestTimeIndex(unittest.TestCase):
    def setUp(self):
        self.bounds = [
            [dt.datetime(2019, 1, 1), dt.datetime(2019, 1, 2)],
            [dt.datetime(2019, 1, 2), dt.datetime(2019, 1, 3)],
            [dt.datetime(2019, 1, 3), dt.datetime(2019, 1, 4)]
        ]

    def test_time_index_outside_bounds_returns_none(self):
        self.check(self.bounds, dt.datetime(2019, 1, 10), None)

    def test_time_index_given_value_on_lower_bound(self):
        self.check(self.bounds, dt.datetime(2019, 1, 2), 1)

    def test_time_index_given_value_inside_bounds(self):
        self.check(self.bounds, dt.datetime(2019, 1, 1, 12), 0)

    def test_time_index_given_value_on_upper_bound(self):
        self.check(self.bounds, dt.datetime(2019, 1, 3), 2)

    def check(self, bounds, time, expect):
        result = main.time_index(bounds, time)
        self.assertEqual(expect, result)


class TestMostRecent(unittest.TestCase):
    def test_most_recent(self):
        times = [
            dt.datetime(2019, 1, 1),
            dt.datetime(2019, 1, 2),
            dt.datetime(2019, 1, 5)
        ]
        time = dt.datetime(2019, 1, 4)
        self.check(times, time, times[1])

    def test_most_recent_given_time_to_left_of_series(self):
        times = [
            dt.datetime(2019, 1, 1),
            dt.datetime(2019, 1, 2),
            dt.datetime(2019, 1, 5)
        ]
        time = dt.datetime(2019, 1, 9)
        self.check(times, time, times[2])

    def check(self, times, time, expect):
        result = main.most_recent(times, time)
        self.assertEqual(expect, result)
