import unittest
import unittest.mock
import bokeh.layouts
import forest
import os
import datetime as dt
import netCDF4


class TestFileName(unittest.TestCase):
    def setUp(self):
        self.controller = forest.navigate.FileName()

    def test_render_given_state(self):
        state = {
            'file_names': ["hello.nc", "goodbye.nc"]
        }
        self.controller.render(state)
        result = self.controller.drop_down.menu
        expect = [("hello.nc", "hello.nc"), ("goodbye.nc", "goodbye.nc")]
        self.assertEqual(expect, result)

    def test_render_given_state_without_file_names(self):
        state = {}
        self.controller.render(state)
        result = self.controller.drop_down.menu
        expect = []
        self.assertEqual(expect, result)

    def test_render_given_file_name(self):
        state = {
            'file_name': "hello.nc"
        }
        self.controller.render(state)
        result = self.controller.drop_down.label
        expect = "hello.nc"
        self.assertEqual(expect, result)

    def test_on_change_emits_action(self):
        attr, old, new = None, None, "file.nc"
        listener = unittest.mock.Mock()
        self.controller.subscribe(listener)
        self.controller.on_change(attr, old, new)
        action = forest.actions.set_item(
                "file_name", "file.nc")
        listener.assert_called_once_with(action)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.store = forest.Store(forest.reducer)

    def test_default_state(self):
        result = self.store.state
        expect = {}
        self.assertEqual(expect, result)

    def test_dispatch_given_action_updates_state(self):
        action = forest.actions.set_item(
                "file_name", "file.nc")
        self.store.dispatch(action)
        result = self.store.state
        expect = {
            "file_name": "file.nc"
        }
        self.assertEqual(expect, result)

    def test_store_state_is_observable(self):
        def reducer(state, action):
            return "STATE"
        store = forest.Store(reducer)
        listener = unittest.mock.Mock()
        store.subscribe(listener)
        store.dispatch("ACTION")
        expect = "STATE"
        listener.assert_called_once_with(expect)


class TestReducer(unittest.TestCase):
    def test_reducer_given_file_name(self):
        self.set_item("file_name", "file.nc")

    def test_reducer_given_file_names_action(self):
        self.set_item("file_names", ["a.nc", "b.nc"])

    def test_reducer_given_variable(self):
        self.set_item("variable", "air_temperature")

    def test_reducer_given_pressures(self):
        self.set_item("pressures", [1000., 950.])

    def test_reducer_given_pressure(self):
        self.set_item("pressure", 850.)

    def test_reducer_given_initial_time(self):
        self.set_item("initial_time", "2019-01-01 00:00:00")

    def test_reducer_given_initial_times(self):
        self.set_item('initial_times', [
            "2019-01-01 00:00:00", "2019-01-01 12:00:00"])

    def test_reducer_given_set_valid_time(self):
        self.set_item("valid_time", "2019-01-01 00:00:00")

    def set_item(self, attr, value):
        action = forest.actions.set_item(attr, value)
        result = forest.reducer({}, action)
        expect = {attr: value}
        self.assertEqual(expect, result)


class TestMiddlewares(unittest.TestCase):
    def test_middleware_log_actions(self):
        action = forest.actions.set_item('k', 'v')
        log = forest.actions.Log()
        store = forest.Store(forest.reducer, middlewares=[log])
        store.dispatch(action)
        result = log.actions
        expect = [action]
        self.assertEqual(expect, result)
