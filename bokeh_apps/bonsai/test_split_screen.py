import unittest
import bokeh.models
import main


class TestSplitScreen(unittest.TestCase):
    def test_toggle_action(self):
        state = main.State()
        action = main.Toggle("split_screen")
        result = main.reducer(state, action)
        self.assertEqual(action.kind, "TOGGLE")
        self.assertEqual(result.split_screen, True)

    def test_app_creates_button(self):
        app = main.Application(main.Config())
        result = app.buttons["split_screen"]
        self.assertIsInstance(result, bokeh.models.Button)
