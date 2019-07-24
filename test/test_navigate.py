import unittest
import forest


class TestNavigator(unittest.TestCase):
    def test_navigator_variables_dropdown_width(self):
        navigator = forest.Navigator()
        result = navigator.dropdowns["variable"].width
        expect = None
        self.assertEqual(expect, result)
