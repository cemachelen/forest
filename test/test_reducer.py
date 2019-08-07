import unittest
import forest
import datetime as dt
import numpy as np


class Convert(object):
    """Middleware to convert values"""
    def __init__(self, keys, convert):
        self.keys = keys
        self.convert = convert

    def __call__(self, store):
        def inner(next_method):
            def inner_most(action):
                print(action)
                kind = action["kind"]
                if kind == forest.actions.SET_ITEM:
                    key, value = action["key"], action["value"]
                    if key in self.keys:
                        action = forest.actions.set_item(
                                key, self.convert(value))
                        next_method(action)
                    else:
                        next_method(action)
                else:
                    next_method(action)
            return inner_most
        return inner


def to_string(items):
    if isinstance(items, str):
        return items
    return [str(item) for item in items]


class TestForestReducer(unittest.TestCase):
    def test_next_valid_time_given_datetimes_and_str(self):
        valid_times = [
            dt.datetime(2019, 1, 1),
            dt.datetime(2019, 1, 2),
            dt.datetime(2019, 1, 3),
            dt.datetime(2019, 1, 4)
        ]
        store = forest.Store(forest.reducer,
                middlewares=[
                    Convert(["valid_time", "valid_times"],
                            to_string)])
        actions = [
            forest.actions.set_item(
                "valid_time", "2019-01-02 00:00:00"),
            forest.actions.set_item(
                "valid_times", valid_times),
            forest.actions.next_item(
                "valid_time", "valid_times"),
        ]
        for action in actions:
            store.dispatch(action)
        result = store.state["valid_time"]
        expect = "2019-01-03 00:00:00"
        self.assertEqual(expect, result)

    def test_set_valid_times(self):
        valid_times = [
            dt.datetime(2019, 1, 1),
            dt.datetime(2019, 1, 2)
        ]
        middleware = Convert(["valid_times"], to_string)
        store = forest.Store(forest.reducer, middlewares=[middleware])
        action = forest.actions.set_item(
                "valid_times", valid_times)
        store.dispatch(action)
        result = store.state["valid_times"]
        expect = ["2019-01-01 00:00:00", "2019-01-02 00:00:00"]
        self.assertEqual(expect, result)
