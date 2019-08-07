import unittest
import forest
import datetime as dt
import numpy as np


class Convert(object):
    """Middleware to convert values"""
    def __init__(self, attrs, convert):
        self.attrs = attrs
        self.convert = convert

    def __call__(self, store):
        def inner(next_method):
            def inner_most(action):
                section, kind, *rest = action
                if kind.lower() == "set":
                    attr, values = rest
                    if attr in self.attrs:
                        values = self.convert(values)
                        next_method((section, 'set', attr, values))
                    else:
                        next_method(action)
                else:
                    next_method(action)
            return inner_most
        return inner


def to_string(items):
    return [str(item) for item in items]


class TestForestReducer(unittest.TestCase):
    def test_next_valid_time_given_datetimes_and_str(self):
        valid_times = [
            dt.datetime(2019, 1, 1),
            dt.datetime(2019, 1, 2),
            dt.datetime(2019, 1, 3),
            dt.datetime(2019, 1, 4)
        ]
        store = forest.Store(forest.reducer, state={
            "navigate": {
                "valid_time": "2019-01-02 00:00:00",
                "valid_times": valid_times
            }
        })
        action = ("navigate", "MOVE", "valid_time", "valid_times", "INCREMENT")
        store.dispatch(action)
        result = store.state["navigate"]["valid_time"]
        expect = "2019-01-03 00:00:00"
        self.assertEqual(expect, result)

    def test_set_valid_times(self):
        valid_times = [
            dt.datetime(2019, 1, 1),
            dt.datetime(2019, 1, 2)
        ]
        middleware = Convert(["valid_times"], to_string)
        store = forest.Store(forest.reducer, middlewares=[middleware])
        action = ("navigate", "set", "valid_times", valid_times)
        store.dispatch(action)
        result = store.state["navigate"]["valid_times"]
        expect = ["2019-01-01 00:00:00", "2019-01-02 00:00:00"]
        self.assertEqual(expect, result)
