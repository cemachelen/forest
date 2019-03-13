import unittest
import unittest.mock
import rx


class TestStream(unittest.TestCase):
    def setUp(self):
        self.view = unittest.mock.Mock()

    def test_main(self):
        stream = rx.Stream.from_list([1, 2, 3])
        stream.subscribe(self.view)
        self.assert_has_calls(self.view, [1, 2, 3])

    def test_emit_tracks_history(self):
        stream = rx.Stream()
        stream.emit("a")
        stream.emit("b")
        stream.emit("c")
        stream.subscribe(self.view)
        self.assert_has_calls(self.view, ["a", "b", "c"])

    def test_combine_streams(self):
        models = rx.Stream()
        dates = rx.Stream()
        stream = rx.combine(models, dates)
        stream.subscribe(self.view)
        dates.emit("20190101")
        models.emit("A")
        self.assert_has_calls(self.view, [(None, "20190101"), ("A", None)])

    def test_scan(self):
        stream = rx.Stream()
        stream.emit(1)
        scanned = rx.scan(stream, 0, lambda s, i: s + i)
        scanned.subscribe(self.view)
        stream.emit(2)
        stream.emit(3)
        self.assert_has_calls(self.view, [1, 3, 6])

    def test_map(self):
        stream = rx.Stream()
        stream.emit(1)
        mapped = rx.map(stream, lambda v: 2 * v)
        mapped.subscribe(self.view)
        stream.emit(2)
        stream.emit(3)
        self.assert_has_calls(self.view, [2, 4, 6])

    def test_combine_latest(self):
        def combinator(value_1, value_2):
            return (value_1, value_2)

        streams = [rx.Stream(), rx.Stream()]
        combined = rx.combine_latest(streams, combinator)
        combined.subscribe(self.view)
        streams[0].emit(1)
        streams[1].emit(2)
        streams[0].emit(3)
        self.assert_has_calls(self.view, [(1, 2), (3, 2)])

    def test_unique(self):
        stream = rx.Stream()
        uniqued = rx.unique(stream)
        uniqued.subscribe(self.view)
        stream.emit(1)
        stream.emit(2)
        stream.emit(2)
        stream.emit(3)
        self.assert_has_calls(self.view, [1, 2, 3])

    def assert_has_calls(self, view, values):
        calls = [unittest.mock.call(v) for v in values]
        view.assert_has_calls(calls)
