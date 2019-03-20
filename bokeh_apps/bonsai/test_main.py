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
import ui


class TestEnvironment(unittest.TestCase):
    def tearDown(self):
        for variable in ["FOREST_DIR"]:
            if variable in os.environ:
                del os.environ[variable]

    def test_parse_env_given_forest_dir(self):
        os.environ["FOREST_DIR"] = "/some/dir"
        result = main.parse_env().directory
        expect = "/some/dir"
        self.assertEqual(expect, result)

    def test_parse_env_default_forest_dir(self):
        result = main.parse_env().directory
        expect = None
        self.assertEqual(expect, result)

class TestDropdownHelpers(unittest.TestCase):
    def test_pluck(self):
        result = main.pluck([{"name": "A"}, {"name": "B"}], "name")
        expect = ["A", "B"]
        self.assertEqual(expect, result)

    def test_as_menu(self):
        result = main.as_menu(["Name"])
        expect = [("Name", "Name")]
        self.assertEqual(expect, result)

class TestConfig(unittest.TestCase):
    def setUp(self):
        self.path = "test-config.yaml"
        self.config = main.Config()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_load(self):
        with open(self.path, "w") as stream:
            yaml.dump({"lon_range": [-10, 10]}, stream)
        result = self.config.load(self.path).lon_range
        expect = [-10, 10]
        self.assertEqual(expect, result)

    def test_default_lon_range(self):
        self.check_default("lon_range", [-180, 180])

    def test_default_lat_range(self):
        self.check_default("lat_range", [-80, 80])

    def test_default_title(self):
        self.check_default("title", "Bonsai - miniature Forest")

    def test_default_models(self):
        self.check_default("models", [])

    def test_default_observations(self):
        self.check_default("observations", [])

    def test_load_given_observations(self):
        data = {
            "observations": []
        }
        self.check_load(data, "observations", [])

    def test_load_given_observation_entry(self):
        data = {
            "observations": [
                {"name": "GPM", "pattern": "*.nc"}
            ]
        }
        expect = [{"name": "GPM", "pattern": "*.nc"}]
        self.check_load(data, "observations", expect)

    def check_load(self, data, attr, expect):
        with open(self.path, "w") as stream:
            yaml.dump(data, stream)
        result = getattr(main.Config.load(self.path), attr)
        self.assertEqual(expect, result)

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
    def test_find_forecast(self):
        paths = [
            "file_20190101T0000Z.nc",
            "file_20190101T1200Z.nc",
            "file_20190102T0000Z.nc"
        ]
        run_date = dt.datetime(2019, 1, 1, 2)
        result = main.find_forecast(paths, run_date)
        expect = "file_20190101T0000Z.nc"
        self.assertEqual(expect, result)

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
