import unittest
import unittest.mock
import bokeh.layouts
import forest.control


class TestFileSystem(unittest.TestCase):
    def setUp(self):
        self.controller = forest.control.FileSystem()

    def test_render(self):
        self.controller.render({})
        self.assertIsInstance(self.controller.layout, bokeh.layouts.Column)

    def test_render_given_state(self):
        state = {
            'file_names': ["hello.nc", "goodbye.nc"]
        }
        self.controller.render(state)
        result = self.controller.dropdown.menu
        expect = [("hello.nc", "hello.nc"), ("goodbye.nc", "goodbye.nc")]
        self.assertEqual(expect, result)

    def test_render_given_state_without_file_names(self):
        state = {}
        self.controller.render(state)
        result = self.controller.dropdown.menu
        expect = []
        self.assertEqual(expect, result)

    def test_render_given_file_name(self):
        state = {
            'file': "hello.nc"
        }
        self.controller.render(state)
        result = self.controller.dropdown.label
        expect = "hello.nc"
        self.assertEqual(expect, result)

    def test_on_file_emits_action(self):
        attr, old, new = None, None, "file.nc"
        listener = unittest.mock.Mock()
        self.controller.subscribe(listener)
        self.controller.on_file(attr, old, new)
        action = ("set file", "file.nc")
        listener.assert_called_once_with(action)


class TestStore(unittest.TestCase):
    def setUp(self):
        self.store = forest.control.Store(forest.control.reducer)

    def test_default_state(self):
        result = self.store.state
        expect = {}
        self.assertEqual(expect, result)

    def test_dispatch_given_action_updates_state(self):
        action = forest.control.set_file("file.nc")
        self.store.dispatch(action)
        result = self.store.state
        expect = {
            "file": "file.nc"
        }
        self.assertEqual(expect, result)

    def test_store_state_is_observable(self):
        action = forest.control.set_file("file.nc")
        listener = unittest.mock.Mock()
        self.store.subscribe(listener)
        self.store.dispatch(action)
        expect = {"file": "file.nc"}
        listener.assert_called_once_with(expect)


class TestReducer(unittest.TestCase):
    def test_reducer(self):
        result = forest.control.reducer({}, ("set file", "file.nc"))
        expect = {
            "file": "file.nc"
        }
        self.assertEqual(expect, result)

    def test_reducer_given_set_file_names_action(self):
        files = ["a.nc", "b.nc"]
        action = forest.control.set_file_names(files)
        result = forest.control.reducer({}, action)
        expect = {
            "file_names": files
        }
        self.assertEqual(expect, result)


class TestMiddlewares(unittest.TestCase):
    def test_middleware_log_actions(self):
        action = ("Hello", "World!")
        log = forest.control.ActionLog()
        store = forest.control.Store(forest.control.reducer, middlewares=[log])
        store.dispatch(action)
        result = log.actions
        expect = [action]
        self.assertEqual(expect, result)
