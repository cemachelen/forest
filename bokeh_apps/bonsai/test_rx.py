import unittest
import unittest.mock
from functools import partial


class Stream(object):
    def __init__(self, history=None):
        if history is None:
            history = []
        self.history = history
        self.callbacks = []

    @classmethod
    def from_list(cls, values):
        return cls(history=values)

    def subscribe(self, callback):
        self.callbacks.append(callback)
        for event in self.history:
            callback(event)

    def emit(self, value):
        self.history.append(value)
        for callback in self.callbacks:
            callback(value)


def combine(*streams):
    n = len(streams)
    combined = Stream()

    def callback(i, value):
        state = n * [None]
        state[i] = value
        combined.emit(tuple(state))

    for i, stream in enumerate(streams):
        stream.subscribe(partial(callback, i))
    return combined


def combine_latest(streams, combinator):
    n = len(streams)
    combined = Stream()
    latest = {}

    def callback(i, value):
        nonlocal latest
        latest[i] = value
        for i in range(n):
            if i not in latest:
                return
        args = [latest[i] for i in range(n)]
        state = combinator(*args)
        combined.emit(state)

    for i, stream in enumerate(streams):
        stream.subscribe(partial(callback, i))
    return combined


def scan(stream, seed, accumulator):
    state = seed
    scanned = Stream()

    def callback(value):
        nonlocal state
        state = accumulator(state, value)
        scanned.emit(state)

    stream.subscribe(callback)
    return scanned


def map(stream, transform):
    mapped = Stream()

    def callback(value):
        mapped.emit(transform(value))

    stream.subscribe(callback)
    return mapped


def echo(stream):
    stream.subscribe(print)


class TestStream(unittest.TestCase):
    def setUp(self):
        self.view = unittest.mock.Mock()

    def test_main(self):
        stream = Stream.from_list([1, 2, 3])
        stream.subscribe(self.view)
        self.assert_has_calls(self.view, [1, 2, 3])

    def test_emit_tracks_history(self):
        stream = Stream()
        stream.emit("a")
        stream.emit("b")
        stream.emit("c")
        stream.subscribe(self.view)
        self.assert_has_calls(self.view, ["a", "b", "c"])

    def test_combine_streams(self):
        models = Stream()
        dates = Stream()
        stream = combine(models, dates)
        stream.subscribe(self.view)
        dates.emit("20190101")
        models.emit("A")
        self.assert_has_calls(self.view, [(None, "20190101"), ("A", None)])

    def test_scan(self):
        stream = Stream()
        stream.emit(1)
        scanned = scan(stream, 0, lambda s, i: s + i)
        scanned.subscribe(self.view)
        stream.emit(2)
        stream.emit(3)
        self.assert_has_calls(self.view, [1, 3, 6])

    def test_map(self):
        stream = Stream()
        stream.emit(1)
        mapped = map(stream, lambda v: 2 * v)
        mapped.subscribe(self.view)
        stream.emit(2)
        stream.emit(3)
        self.assert_has_calls(self.view, [2, 4, 6])

    def test_combine_latest(self):
        def combinator(value_1, value_2):
            return (value_1, value_2)

        streams = [Stream(), Stream()]
        combined = combine_latest(streams, combinator)
        combined.subscribe(self.view)
        streams[0].emit(1)
        streams[1].emit(2)
        streams[0].emit(3)
        self.assert_has_calls(self.view, [(1, 2), (3, 2)])

    def assert_has_calls(self, view, values):
        calls = [unittest.mock.call(v) for v in values]
        view.assert_has_calls(calls)
