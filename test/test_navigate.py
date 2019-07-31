import unittest
import unittest.mock
import forest


class TestNavigator(unittest.TestCase):
    def setUp(self):
        self.navigator = forest.Navigator()

    def test_navigator_variables_dropdown_width(self):
        result = self.navigator.dropdowns["variable"].width
        expect = None
        self.assertEqual(expect, result)

    def test_on_click_forward(self):
        listener = unittest.mock.Mock()
        self.navigator.subscribe(listener)
        self.navigator.on_click("pressure", "pressures", "increment")()
        listener.assert_called_once_with(
            ("MOVE", "pressure", "pressures", "increment"))
