import unittest
import forest
from forest.actions import SET, ADD, REMOVE


class TestForwardBackward(unittest.TestCase):
    def test_actions_given_next_pressure(self):
        result = forest.move("pressure", "pressures", "forward")
        expect = ("MOVE", "pressure", "GIVEN", "pressures", "forward")
        self.assertEqual(expect, result)

    def test_actions_given_previous_valid_time(self):
        result = forest.move("valid_time", "valid_times", "backward")
        expect = ("MOVE", "valid_time", "GIVEN", "valid_times", "backward")
        self.assertEqual(expect, result)

    def test_reducer_given_empty_state(self):
        action = forest.move("item", "items", "forward")
        result = forest.reducer({}, action)
        expect = {}
        self.assertEqual(expect, result)

    def test_reducer_next_default_value_returns_max(self):
        pressures = [1, 2, 3]
        action = forest.move("pressure", "pressures", "forward")
        result = forest.reducer({"pressures": pressures}, action)
        expect = {
            "pressures": pressures,
            "pressure": 3
        }
        self.assertEqual(expect, result)

    def test_reducer_next_item_given_item_in_items(self):
        item = 2
        items = [1, 2, 3, 4, 5]
        action = forest.move("item", "items", "forward")
        result = forest.reducer({"item": item, "items": items}, action)
        expect = {
            "item": 3,
            "items": items
        }
        self.assertEqual(expect, result)

    def test_reducer_backward_default_value_returns_min(self):
        pressures = [1, 2, 3]
        action = forest.move("pressure", "pressures", "backward")
        result = forest.reducer({"pressures": pressures}, action)
        expect = {
            "pressures": pressures,
            "pressure": 1
        }
        self.assertEqual(expect, result)


class TestPreset(unittest.TestCase):
    """Hypothetical implementation of presets as data"""
    def test_set_current_preset_action(self):
        result = SET.preset.to({
            "name": "Custom",
            "palette": "Viridis"
        })
        expect = ("SET", "preset", {"name": "Custom", "palette": "Viridis"})
        self.assertEqual(expect, result)

    def test_set_current_preset_reducer(self):
        action = SET.preset.to({
            "name": "Custom",
            "palette": "Viridis"
        })
        result = forest.reducer({}, action)
        expect = {
            "preset": {
                "name": "Custom",
                "palette": "Viridis"
            }
        }
        self.assertEqual(expect, result)

    def test_add_preset_action(self):
        result = ADD.preset.by_name("Default", {
            "palette": "Viridis"
        })
        expect = ("ADD", "preset", "Default", {"palette": "Viridis"})
        self.assertEqual(expect, result)

    def test_add_preset_reducer(self):
        result = forest.reducer({}, ADD.preset.by_name("Hello", {
            "palette": "Viridis"
        }))
        expect = {
            "presets": [{
                "name": "Hello",
                "palette": "Viridis"
            }]
        }
        self.assertEqual(expect, result)

    def test_remove_preset(self):
        result = forest.reducer({
            "presets": [
                {
                    "name": "Default",
                    "min": 0
                }
            ]
        }, REMOVE.preset.by_name("Default"))
        expect = {
            "presets": []
        }
        self.assertEqual(expect, result)

    def test_add_preset_twice(self):
        state = {}
        state = forest.reducer(state, ADD.preset.by_name("Hello", {
            "palette": "Viridis"
        }))
        state = forest.reducer(state, ADD.preset.by_name("Hello", {
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
        state = forest.reducer(state, ADD.preset.by_name("A", {
            "palette": "Viridis"
        }))
        state = forest.reducer(state, ADD.preset.by_name("B", {
            "palette": "Blues"
        }))
        result = state
        expect = {
            "presets": [
                {"name": "A", "palette": "Viridis"},
                {"name": "B", "palette": "Blues"}]
        }
        self.assertEqual(expect, result)


class TestActions(unittest.TestCase):
    def test_set_pressure(self):
        result = SET.pressure.to(1000.)
        expect = ("SET", "pressure", 1000.)
        self.assertEqual(expect, result)

    def test_move_item_given_items_forward(self):
        result = forest.move("pressure", "pressures", "forward")
        expect = ("MOVE", "pressure", "GIVEN", "pressures", "forward")
        self.assertEqual(expect, result)
