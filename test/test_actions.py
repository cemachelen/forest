import unittest
import forest
from forest.actions import SET, ADD, REMOVE


class TestForwardBackward(unittest.TestCase):
    def test_reducer_given_empty_state(self):
        action = ("navigate", "move", "item", "items", "increment")
        result = forest.reducer({}, action)
        expect = {
            "navigate": {},
            "preset": {}
        }
        self.assertEqual(expect, result)

    def test_reducer_next_default_value_returns_max(self):
        pressures = [1, 2, 3]
        action = ("navigate", "move", "pressure", "pressures", "increment")
        state = forest.reducer({
                "navigate": {"pressures": pressures}
            }, action)
        result = state["navigate"]
        expect = {
            "pressures": pressures,
            "pressure": 3
        }
        self.assertEqual(expect, result)

    def test_reducer_next_item_given_item_in_items(self):
        item = 2
        items = [1, 2, 3, 4, 5]
        action = ("navigate", "move", "item", "items", "increment")
        state = {
            "navigate": {
                "item": item,
                "items": items
            }
        }
        state = forest.reducer(state, action)
        result = state["navigate"]
        expect = {
            "item": 3,
            "items": items
        }
        self.assertEqual(expect, result)

    def test_reducer_decrease_default_value_returns_min(self):
        pressures = [1, 2, 3]
        action = ("navigate", "move", "pressure", "pressures", "decrement")
        state = {
            "navigate": {
                "pressures": pressures
            }
        }
        state = forest.reducer(state, action)
        result = state["navigate"]
        expect = {
            "pressures": pressures,
            "pressure": 1
        }
        self.assertEqual(expect, result)


class TestPreset(unittest.TestCase):
    """Hypothetical implementation of presets as data"""
    def test_add_preset_reducer(self):
        action = ("preset", "add", {"name": "Custom", "palette": "Viridis"})
        result = forest.reducer({}, action)["preset"]
        expect = {
            "presets": [
                {
                    "name": "Custom",
                    "palette": "Viridis"
                }
            ]
        }
        self.assertEqual(expect, result)

    def test_remove_preset(self):
        action = ("preset", "remove", "Default")
        result = forest.reducer({
            "preset": {
                "presets": [
                    {
                        "name": "Default",
                        "min": 0
                    }
                ]
            }
        }, action)
        expect = {
            "preset": {
                "presets": []
            }
        }
        self.assertEqual(expect, result)

    def test_add_preset_twice(self):
        state = {}
        state = forest.reducer(state, ("preset", "add", "Hello", {
            "palette": "Viridis"
        }))
        state = forest.reducer(state, ("preset", "add", "Hello", {
            "palette": "Blues"
        }))
        result = state
        expect = {
            "presets": [{
                "name": "Hello",
                "palette": "Blues"
            }]
        }
        self.assertEqual(expect, result)

    def test_add_different_presets(self):
        state = {}
        state = forest.reducer(state, ("preset", "add", "A", {
            "palette": "Viridis"
        }))
        state = forest.reducer(state, ("preset", "add", "B", {
            "palette": "Blues"
        }))
        result = state["preset"]
        expect = {
            "presets": [
                {"name": "A", "palette": "Viridis"},
                {"name": "B", "palette": "Blues"}]
        }
        self.assertEqual(expect, result)
