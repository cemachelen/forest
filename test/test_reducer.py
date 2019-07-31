import unittest
import forest
import datetime as dt
import numpy as np


class TestForestReducer(unittest.TestCase):
    def test_next_valid_time_given_datetimes_and_str(self):
        valid_times = [
            dt.datetime(2019, 1, 1),
            dt.datetime(2019, 1, 2),
            dt.datetime(2019, 1, 3),
            dt.datetime(2019, 1, 4)
        ]
        store = forest.Store(forest.reducer, state={
            "valid_time": "2019-01-02 00:00:00",
            "valid_times": valid_times
        })
        action = forest.actions.Move("valid_time", "valid_times").increment
        store.dispatch(action)
        result = store.state["valid_time"]
        expect = "2019-01-03 00:00:00"
        self.assertEqual(expect, result)

    def test_next_valid_time_given_current_time_not_in_list(self):
        valid_times = [
            dt.datetime(2019, 1, 1),
            dt.datetime(2019, 1, 2),
            dt.datetime(2019, 1, 3),
            dt.datetime(2019, 1, 4)
        ]
        store = forest.Store(forest.reducer, state={
            "valid_time": "2019-01-02 00:00:00",
            "valid_times": valid_times
        })
        action = forest.actions.Move("valid_time", "valid_times").increment
        store.dispatch(action)
        result = store.state["valid_time"]
        expect = "2019-01-03 00:00:00"
        self.assertEqual(expect, result)
