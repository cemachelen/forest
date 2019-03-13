"""Minimal reactive helper methods and classes"""
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


def merge(*streams):
    merged = Stream()
    for stream in streams:
        stream.subscribe(merged.emit)
    return merged


def unique(stream):
    """Emit unique values from a stream

    .. note:: The returned stream.emit() does not filter duplicates
              the duplicate check is done on values emitted from the
              input stream
    """
    uniqued = Stream()

    def callback(value):
        if len(uniqued.history) == 0:
            uniqued.emit(value)
            return
        else:
            previous = uniqued.history[-1]
            if value == previous:
                return
            uniqued.emit(value)

    stream.subscribe(callback)
    return uniqued


def on_change(stream):
    """Convert callback(attr, old, new) to stream.emit(new)"""
    def callback(attr, old, new):
        stream.emit(new)
    return callback
