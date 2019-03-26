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
        state = main.State()
        state = main.reducer(state, main.Action.activate("model"))
        state = main.reducer(state, main.SetName("model", "Model"))
        self.app.render(state)
        self.assertEqual(self.app.title.text, "Model")

    def test_title_text(self):
        self.check(main.State(), "")

    def test_title_text_given_valid_date(self):
        date = dt.datetime(2019, 1, 1)
        state = main.reducer(main.State(), main.Action.set_valid_date(date))
        self.check(state, "2019-01-01 00:00")

    def test_title_text_given_observation_name(self):
        state = main.State()
        for action in [
                main.Action.activate("observation"),
                main.Action.set_name("observation", "GPM")]:
            state = main.reducer(state, action)
        self.check(state, "GPM")

    def test_file_not_found_sets_messenger_text(self):
        self.app.store.dispatch(main.FileNotFound("key"))
        result = self.app.messenger.text
        expect = "File not found"
        self.assertEqual(expect, result)

    def test_file_found_leaves_messenger_unaltered(self):
        self.app.store.dispatch(main.FileFound("key", "value"))
        result = self.app.messenger.text
        expect = ""
        self.assertEqual(expect, result)

    def check(self, state, expect):
        result = self.app.title_text(state)
        self.assertEqual(expect, result)


class TestState(unittest.TestCase):
    def test_valid_date(self):
        self.check("valid_date", None)

    def test_listing(self):
        self.check("listing", False)

    def test_found(self):
        self.check("found", False)

    def test_loading(self):
        self.check("loading", False)

    def test_files(self):
        self.check("files", {})

    def check(self, attr, expect):
        result = getattr(main.State(), attr)
        self.assertEqual(expect, result)

    def test_reducer(self):
        valid_date = dt.datetime(2019, 1, 1)
        state = main.State()
        action = main.Action.set_valid_date(valid_date)
        result = main.reducer(state, action).valid_date
        expect = valid_date
        self.assertEqual(expect, result)

    def test_reducer_given_file_not_found(self):
        state = main.State()
        action = main.FileNotFound(("Model", dt.datetime(2019, 1, 1)))
        result = main.reducer(state, action).missing_files
        expect = set([
            ("Model", dt.datetime(2019, 1, 1))
        ])
        self.assertEqual(expect, result)

    def test_reducer_given_multiple_actions(self):
        date = dt.datetime(2019, 1, 1)
        state = main.State()
        actions = [main.Action.set_valid_date(date),
                   main.FileFound("k", "v")]
        for action in actions:
            state = main.reducer(state, action)
        self.assertEqual(state.valid_date, dt.datetime(2019, 1, 1))
        self.assertEqual(state.found, True)
        self.assertEqual(state.found_files, {"k": "v"})

    def test_reducer_given_activate_sets_selected_name(self):
        state = main.State()
        actions = [
            main.FileFound("k", "v"),
            main.Action.activate("model")]
        for action in actions:
            state = main.reducer(state, action)
        self.assertEqual(state.found, True)
        self.assertEqual(state.active.category, "model")

    def test_reducer_given_model_and_observations(self):
        state = main.State()
        actions = [
            main.Action.set_name("model", "A"),
            main.Action.set_name("model", "B"),
            main.Action.set_name("observation", "C"),
            main.Action.activate("model"),
            main.Action.activate("observation"),
            main.Action.activate("model"),
        ]
        for action in actions:
            state = main.reducer(state, action)
        self.assertEqual(state.active.category, "model")
        self.assertEqual(state.active.name, "B")


class TestFindFileByValidDate(unittest.TestCase):
    def setUp(self):
        self.paths = []

    def tearDown(self):
        for path in self.paths:
            if os.path.exists(path):
                os.remove(path)

    def test_find_file_given_date(self):
        self.paths = ["file.nc"]
        start = dt.datetime(2019, 1, 1)
        bounds = self.make_bounds(start, [[0, 3], [3, 6]])
        for path in self.paths:
            with netCDF4.Dataset(path, "w") as dataset:
                self.set_bounds(dataset, bounds)
        date = dt.datetime(2019, 1, 1)
        result = main.find_file(self.paths, date)
        expect = "file.nc", 0
        self.assertEqual(expect, result)

    def test_find_file_given_date_in_second_file(self):
        self.paths = [
                "file_0.nc",
                "file_1.nc",
                "file_2.nc"]
        starts = [
                dt.datetime(2019, 1, 1),
                dt.datetime(2019, 1, 2),
                dt.datetime(2019, 1, 3)]
        hours = [[0, 3], [3, 6], [6, 9]]
        for path, start in zip(self.paths, starts):
            with netCDF4.Dataset(path, "w") as dataset:
                bounds = self.make_bounds(start, hours)
                self.set_bounds(dataset, bounds)
        date = dt.datetime(2019, 1, 2)
        result = main.find_file(self.paths, date)
        expect = "file_1.nc", 0
        self.assertEqual(expect, result)

    def test_find_file_returns_index(self):
        self.paths = [
                "file_0.nc",
                "file_1.nc",
                "file_2.nc"]
        starts = [
                dt.datetime(2019, 1, 1),
                dt.datetime(2019, 1, 2),
                dt.datetime(2019, 1, 3)]
        hours = [[0, 3], [3, 6], [6, 9]]
        for path, start in zip(self.paths, starts):
            with netCDF4.Dataset(path, "w") as dataset:
                bounds = self.make_bounds(start, hours)
                self.set_bounds(dataset, bounds)
        date = dt.datetime(2019, 1, 2, 7)
        result = main.find_file(self.paths, date)
        expect = "file_1.nc", 2
        self.assertEqual(expect, result)

    def test_find_file_only_searches_likely_files(self):
        paths = [
                "file_20190101T0000Z.nc",
                "file_20190102T0000Z.nc",
                "file_20190103T0000Z.nc"]
        self.paths = paths
        start = dt.datetime(2019, 1, 2)
        hours = [[0, 3], [3, 6], [6, 9]]
        with netCDF4.Dataset(paths[1], "w") as dataset:
            bounds = self.make_bounds(start, hours)
            self.set_bounds(dataset, bounds)
        date = dt.datetime(2019, 1, 2, 7)
        result = main.find_file(self.paths, date)
        expect = "file_20190102T0000Z.nc", 2
        self.assertEqual(expect, result)

    def test_find_file_stops_if_after_recent_file(self):
        paths = [
                "file_20190101T0000Z.nc",
                "file_20190102T0000Z.nc",
                "file_20190103T0000Z.nc"]
        self.paths = paths
        start = dt.datetime(2019, 1, 3)
        hours = [[0, 3], [3, 6], [6, 9]]
        with netCDF4.Dataset(paths[2], "w") as dataset:
            bounds = self.make_bounds(start, hours)
            self.set_bounds(dataset, bounds)
        date = dt.datetime(2019, 1, 10)
        result = main.find_file(self.paths, date)
        expect = None
        self.assertEqual(expect, result)

    def make_bounds(self, start, hours):
        if isinstance(hours, list):
            hours = np.array(hours, dtype=int)
        cast = np.vectorize(lambda x: dt.timedelta(hours=int(x)))
        return start + cast(hours)

    def set_bounds(self, dataset, bounds):
        if isinstance(bounds, list):
            bounds = np.array(bounds, dtype=object)
        units = "hours since 1970-01-01 00:00:00"
        dataset.createDimension("time_2", bounds.shape[0])
        dataset.createDimension("bnds", 2)
        var = dataset.createVariable(
                "time_2", "d", ("time_2",))
        var.units = units
        var = dataset.createVariable(
                "time_2_bnds", "d", ("time_2", "bnds"))
        var[:] = netCDF4.date2num(bounds, units)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.store = main.Store(main.reducer)

    def test_set_valid_date(self):
        action = main.Action.set_valid_date(dt.datetime(2019, 1, 1))
        self.store.dispatch(action)
        result = self.store.state.valid_date
        expect = dt.datetime(2019, 1, 1)
        self.assertEqual(expect, result)

    def test_action_set_model_name(self):
        action = main.SetName("model", "East Africa 4.4km")
        self.assertEqual(action.kind, "SET_NAME")
        self.assertEqual(action.category, "model")
        self.assertEqual(action.text, "East Africa 4.4km")

    def test_reducer_set_model_name(self):
        action = main.SetName("model", "Tropical Africa 4.4km")
        result = main.reducer(main.State(), action)
        self.assertEqual(result.active.name, "Tropical Africa 4.4km")

    def test_reducer_set_observation_name(self):
        action = main.Action.set_name("observation", "GPM IMERG")
        result = main.reducer(main.State(), action)
        self.assertEqual(result.active.category, "observation")
        self.assertEqual(result.active.name, "GPM IMERG")

    def test_store_subscribe(self):
        action = main.SetName("model", "Model")
        listener = unittest.mock.Mock()
        unsubscribe = self.store.subscribe(listener)
        self.store.dispatch(action)
        listener.assert_called_once_with()

    def test_store_unsubscribe(self):
        action = main.SetName("model", "Model")
        listener = unittest.mock.Mock()
        unsubscribe = self.store.subscribe(listener)
        self.store.dispatch(action)
        unsubscribe()
        self.store.dispatch(action)
        listener.assert_called_once_with()


class TestReducer(unittest.TestCase):
    def setUp(self):
        self.state = main.State()

    def test_reducer_on_pending_glob_request(self):
        result = main.reducer(self.state, main.List().started())
        self.assertEqual(result.listing, True)

    def test_reducer_on_successful_glob_request(self):
        state = main.State()
        response = {"Tropical Africa": ["file.nc"]}
        result = main.reducer(state, main.List().finished(response))
        self.assertEqual(result.listing, False)
        self.assertEqual(result.files, {
            "Tropical Africa": ["file.nc"]
        })

    def test_reducer_on_pending_load_request(self):
        result = main.reducer(self.state, main.Load().started())
        self.assertEqual(result.loading, True)

    def test_reducer_on_finished_load_request(self):
        response = {
            "name": "GPM IMERG",
            "valid_date": dt.datetime(2019, 1, 1),
            "data":{
                "x": []
            }
        }
        result = main.reducer(main.State(), main.Load().finished(response))
        self.assertEqual(result.loading, False)
        self.assertEqual(result.loaded, {
            "name": "GPM IMERG",
            "valid_date": dt.datetime(2019, 1, 1),
            "data": {"x": []}
        })

    def test_reducer_sets_active_name(self):
        state = main.State()
        for action in [
                main.Action.set_observation_name("GPM IMERG"),
                main.Action.activate("observation")]:
            state = main.reducer(state, action)
        result = state
        self.assertEqual(result.active.category, "observation")
        self.assertEqual(result.active.name, "GPM IMERG")

    def test_file_not_found(self):
        result = main.reducer(self.state, main.FileNotFound("k"))
        self.assertEqual(result.found, False)

    def test_file_found(self):
        result = main.reducer(self.state, main.FileFound("k", "v"))
        self.assertEqual(result.found, True)


class TestAction(unittest.TestCase):
    def test_activate_observation(self):
        result = main.Action.activate("observation")
        expect = {
            "type": "ACTIVATE",
            "category": "observation"
        }
        self.assertEqual(expect, result)

    def test_reducer_given_two_activate_actions(self):
        state = main.State()
        for action in [
                main.Action.activate("observation"),
                main.Action.deactivate("observation"),
                main.Action.activate("model")]:
            state = main.reducer(state, action)
        result = state
        self.assertEqual(result.active.category, "model")

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
