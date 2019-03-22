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


class TestApplication(unittest.TestCase):
    def setUp(self):
        self.app = main.Application(main.Config())

    def test_application_given_state(self):
        state = {}
        state = main.reducer(state, main.Action.activate("model"))
        state = main.reducer(state, main.Action.set_model_name("Model"))
        self.app.render(state)
        self.assertEqual(self.app.title.text, "Model")

    def test_title_text(self):
        self.check({}, "")

    def test_title_text_given_valid_date(self):
        date = dt.datetime(2019, 1, 1)
        state = main.reducer({}, main.Action.set_valid_date(date))
        self.check(state, "2019-01-01 00:00")

    def test_title_text_given_observation_name(self):
        state = {}
        for action in [
                main.Action.activate("observation"),
                main.Action.set_name("observation", "GPM")]:
            state = main.reducer(state, action)
        self.check(state, "GPM")

    def check(self, state, expect):
        result = self.app.title_text(state)
        self.assertEqual(expect, result)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.store = main.Store(main.reducer)

    def test_store(self):
        action = {
            "type": "SET_TITLE",
            "text": "Hello, world!"
        }
        self.store.dispatch(action)
        result = self.store.state
        expect = {"title": "Hello, world!"}
        self.assertEqual(expect, result)

    def test_set_valid_date(self):
        action = main.Action.set_valid_date(dt.datetime(2019, 1, 1))
        self.store.dispatch(action)
        result = self.store.state
        expect = {"valid_date": dt.datetime(2019, 1, 1)}
        self.assertEqual(expect, result)

    def test_set_forecast_action(self):
        valid_date = dt.datetime(2019, 1, 1)
        length = dt.timedelta(hours=12)
        result = main.Action.set_forecast(valid_date, length)
        expect = {
            "type": "SET_FORECAST",
            "valid_date": valid_date,
            "length": length,
            "run_date": dt.datetime(2018, 12, 31, 12)
        }
        self.assertEqual(expect, result)

    def test_reducer_given_set_forecast(self):
        valid_date = dt.datetime(2019, 1, 1)
        length = dt.timedelta(hours=12)
        action = main.Action.set_forecast(valid_date, length)
        result = main.reducer({}, action)
        expect = {
            "valid_date": valid_date,
            "length": length,
            "run_date": dt.datetime(2018, 12, 31, 12)
        }
        self.assertEqual(expect, result)

    def test_action_set_model_name(self):
        result = main.Action.set_model_name("East Africa 4.4km")
        expect = {
            "type": "SET_NAME",
            "category": "model",
            "text": "East Africa 4.4km"
        }
        self.assertEqual(expect, result)

    def test_reducer_set_model_name(self):
        action = main.Action.set_model_name("Tropical Africa 4.4km")
        result = main.reducer({}, action)
        expect = {
            "model": {
                "name": "Tropical Africa 4.4km"
            }
        }
        self.assertEqual(expect, result)

    def test_reducer_set_observation(self):
        action = main.Action.set_observation_name("GPM IMERG")
        expect = {
            "observation": {
                "name": "GPM IMERG"
            }
        }
        self.check(action, expect)

    def test_reducer_set_field(self):
        action = main.Action.set_model_field("Precipitation")
        self.check(action, {"model": {"field": "Precipitation"}})

    def check(self, action, expect):
        result = main.reducer({}, action)
        self.assertEqual(expect, result)

    def test_store_subscribe(self):
        action = main.Action.set_model_name("Model")
        listener = unittest.mock.Mock()
        unsubscribe = self.store.subscribe(listener)
        self.store.dispatch(action)
        listener.assert_called_once_with()

    def test_store_unsubscribe(self):
        action = main.Action.set_model_name("Model")
        listener = unittest.mock.Mock()
        unsubscribe = self.store.subscribe(listener)
        self.store.dispatch(action)
        unsubscribe()
        self.store.dispatch(action)
        listener.assert_called_once_with()


class TestReducer(unittest.TestCase):
    def test_reducer_on_pending_glob_request(self):
        result = main.reducer({}, main.List().started())
        expect = {
            "listing": True
        }
        self.assertEqual(expect, result)

    def test_reducer_on_successful_glob_request(self):
        response = {"Tropical Africa": ["file.nc"]}
        result = main.reducer({}, main.List().finished(response))
        expect = {
            "listing": False,
            "files": {
                "Tropical Africa": ["file.nc"]
            }
        }
        self.assertEqual(expect, result)

    def test_reducer_on_pending_load_request(self):
        result = main.reducer({}, main.Load().started())
        expect = {
            "loading": True
        }
        self.assertEqual(expect, result)

    def test_reducer_on_finished_load_request(self):
        response = {
            "name": "GPM IMERG",
            "valid_date": dt.datetime(2019, 1, 1),
            "data":{
                "x": []
            }
        }
        result = main.reducer({}, main.Load().finished(response))
        expect = {
            "loading": False,
            "loaded": {
                "name": "GPM IMERG",
                "valid_date": dt.datetime(2019, 1, 1),
                "data": {"x": []}
            }
        }
        self.assertEqual(expect, result)

    def test_reducer_sets_active_name(self):
        state = {}
        for action in [
                main.Action.set_observation_name("GPM IMERG"),
                main.Action.activate("observation")]:
            state = main.reducer(state, action)
        result = state
        expect = {
            "observation": {
                "name": "GPM IMERG",
                "active": True
            }
        }
        self.assertEqual(expect, result)


class TestLoadNeeded(unittest.TestCase):
    def test_load_needed(self):
        result = main.Application.load_needed(
            {})
        expect = False
        self.assertEqual(expect, result)

    def test_load_needed_given_valid_date(self):
        result = main.Application.load_needed(
            {"valid_date": dt.datetime(2019, 1, 1)})
        expect = False
        self.assertEqual(expect, result)

    def test_load_needed_given_different_valid_dates(self):
        state = {
            "model": {"name": "Tropical", "active": True},
            "valid_date": dt.datetime(2019, 1, 1),
            "loaded": {
                "valid_date": dt.datetime(2019, 1, 2)
            }
        }
        result = main.Application.load_needed(state)
        expect = True
        self.assertEqual(expect, result)

    def test_load_needed_given_same_valid_date(self):
        state = {
            "model": {"name": "Tropical", "active": True},
            "valid_date": dt.datetime(2019, 1, 1),
            "loaded": {
                "valid_date": dt.datetime(2019, 1, 1)
            }
        }
        result = main.Application.load_needed(state)
        expect = False
        self.assertEqual(expect, result)

    def test_load_needed_given_same_date_but_different_model(self):
        state = {
            "observation": {"name": "GPM IMERG early", "active": True},
            "valid_date": dt.datetime(2019, 1, 1),
            "loaded": {
                "name": "GPM IMERG late",
                "valid_date": dt.datetime(2019, 1, 1)
            }
        }
        result = main.Application.load_needed(state)
        expect = True
        self.assertEqual(expect, result)

    def test_load_needed_without_name_returns_false(self):
        result = main.Application.load_needed({
            "observation": {"name": "GPM", "active": False},
            "valid_date": dt.datetime(2019, 1, 1)
        })
        expect = False
        self.assertEqual(expect, result)


class TestAction(unittest.TestCase):
    def test_activate_observation(self):
        result = main.Action.activate("observation")
        expect = {
            "type": "ACTIVATE",
            "category": "observation"
        }
        self.assertEqual(expect, result)

    def test_reducer_given_two_activate_actions(self):
        state = {}
        for action in [
                main.Action.activate("observation"),
                main.Action.deactivate("observation"),
                main.Action.activate("model")]:
            state = main.reducer(state, action)
        result = state
        expect = {
            "model": {"active": True},
            "observation": {"active": False}
        }
        self.assertEqual(expect, result)

    def test_request_started(self):
        result = main.Request("flag").started()
        expect = {
            "type": "REQUEST",
            "flag": "flag",
            "status": "active"}
        self.assertEqual(expect, result)

    def test_request_finished(self):
        response = []
        result = main.Request("flag").finished(response)
        expect = {
            "type": "REQUEST",
            "flag": "flag",
            "status": "succeed",
            "response": response}
        self.assertEqual(expect, result)

    def test_request_failed(self):
        result = main.Request("flag").failed()
        expect = {
            "type": "REQUEST",
            "flag": "flag",
            "status": "fail"}
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
