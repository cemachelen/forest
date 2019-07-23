import unittest
import forest
from forest.actions import SET, MOVE, ADD, REMOVE


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

    def test_move(self):
        result = MOVE.pressure.forward
        expect = ("MOVE", "pressure", "forward")
        self.assertEqual(expect, result)

    def test_move_backward(self):
        result = MOVE.level.backward
        expect = ("MOVE", "level", "backward")
        self.assertEqual(expect, result)

    def test_whitespace(self):
        result = MOVE["some attr"].backward
        expect = ("MOVE", "some attr", "backward")
        self.assertEqual(expect, result)

    def test_move_action(self):
        kind, *rest = MOVE.pressure.forward
        self.assertEqual(kind, "MOVE")
