import unittest
from forest.actions import SET, MOVE


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
