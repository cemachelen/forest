import unittest
import forest


class TestActionCreators(unittest.TestCase):
    def test_next_item(self):
        result = forest.actions.next_item("item", "items")
        expect = {
            "kind": forest.actions.NEXT_ITEM,
            "item_key": "item",
            "items_key": "items"
        }
        self.assertEqual(expect, result)

    def test_previous_item(self):
        result = forest.actions.previous_item("item", "items")
        expect = {
            "kind": forest.actions.PREVIOUS_ITEM,
            "item_key": "item",
            "items_key": "items"
        }
        self.assertEqual(expect, result)

    def test_set_item(self):
        result = forest.actions.set_item("k", "v")
        expect = {
            "kind": forest.actions.SET_ITEM,
            "key": "k",
            "value": "v"
        }
        self.assertEqual(expect, result)


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
