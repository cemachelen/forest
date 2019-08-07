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
        action = forest.actions.SET.file_name.to("file.nc")
        listener.assert_called_once_with(action)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.store = forest.Store(forest.reducer)

    def test_default_state(self):
        result = self.store.state
        expect = {}
        self.assertEqual(expect, result)

    def test_dispatch_given_action_updates_state(self):
        action = ('navigate', 'set', 'file_name', 'file.nc')
        self.store.dispatch(action)
        result = self.store.state['navigate']
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
    def test_reducer(self):
        action = ('navigate', 'set', 'file_name', 'file.nc')
        result = forest.reducer({}, action)['navigate']
        expect = {
            "file_name": "file.nc"
        }
        self.assertEqual(expect, result)

    def test_reducer_given_set_file_names_action(self):
        files = ["a.nc", "b.nc"]
        action = ('navigate', 'set', 'file_names', files)
        result = forest.reducer({}, action)['navigate']
        expect = {
            "file_names": files
        }
        self.assertEqual(expect, result)

    def test_reducer_given_set_variable(self):
        action = ('navigate', 'set', 'variable', 'air_temperature')
        result = forest.reducer({}, action)['navigate']
        expect = {
            "variable": "air_temperature"
        }
        self.assertEqual(expect, result)

    def test_reducer_given_set_pressures(self):
        action = ('navigate', 'set', 'pressures', [1000., 950.])
        self.check(action, "pressures", [1000., 950.])

    def test_reducer_given_set_pressure(self):
        action = ('navigate', 'set', 'pressure', 850.)
        self.check(action, "pressure", 850.)

    def test_reducer_given_set_initial_time(self):
        value = "2019-01-01 00:00:00"
        action = ('navigate', 'set', 'initial_time', value)
        self.check(action, "initial_time", value)

    def test_reducer_given_set_initial_times(self):
        value = ["2019-01-01 00:00:00", "2019-01-01 12:00:00"]
        action = ('navigate', 'set', 'initial_times', value)
        self.check(action, "initial_times", value)

    def test_reducer_given_set_valid_time(self):
        value = "2019-01-01 00:00:00"
        action = ('navigate', 'set', 'valid_time', value)
        self.check(action, "valid_time", value)

    def check(self, action, attr, value):
        result = forest.reducer({}, action)['navigate']
        expect = {
            attr: value
        }
        self.assertEqual(expect, result)


class TestMiddlewares(unittest.TestCase):
    def test_middleware_log_actions(self):
        action = ("Hello", "World!")
        log = forest.ActionLog()
        store = forest.Store(forest.reducer, middlewares=[log])
        store.dispatch(action)
        result = log.actions
        expect = [action]
        self.assertEqual(expect, result)
