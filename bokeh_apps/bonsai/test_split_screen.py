import unittest
import bokeh.models
import main


class SetSide(object):
    def __init__(self, key, value):
        self.kind = "SET_SIDE"
        self.key = key
        self.value = value


class State(object):
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

    @property
    def kwargs(self):
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_")}

    def copy(self):
        return State(**self.kwargs)

    def __eq__(self, other):
        if isinstance(other, dict):
            other = State(**other)
        return self.kwargs == other.kwargs


def reducer(state, action):
    if isinstance(state, dict):
        state = State(**state)
    state = state.copy()
    if action.kind == "SET_SIDE":
        if not hasattr(state, "sides"):
            state.sides = {}
        state.sides[action.key] = action.value
    return state


class TestSplitScreen(unittest.TestCase):
    def test_toggle_action(self):
        state = main.State()
        action = main.Toggle("split_screen")
        result = main.reducer(state, action)
        self.assertEqual(action.kind, "TOGGLE")
        self.assertEqual(result.split_screen, True)

    def test_app_creates_button(self):
        app = main.Application(main.Config())
        result = app.buttons["split_screen"]
        self.assertIsInstance(result, bokeh.models.Button)

    def test_set_side_assigns_glyphs(self):
        action = SetSide("item", "left")
        self.assertEqual(action.kind, "SET_SIDE")
        self.assertEqual(action.key, "item")
        self.assertEqual(action.value, "left")

    def test_reducer_given_sides(self):
        action = SetSide(0, "left")
        result = reducer({}, action)
        expect = {
            "sides": {
                0: "left"
            }
        }
        self.assertEqual(expect, result)

    def test_split_screen_hides_glyphs_on_first_figure(self):
        state = State(
                split_screen=True,
                sides={
                    "GPM early": "left"
                },
                name=None,
                valid_date=None,
                file_not_found=False,
                listing=False,
                loading=False,
                loaded=None,
                hours=[],
                sources={})
        app = main.Application(main.Config())
        app.render(state)
        self.assertEqual(len(app.glyphs), 1)
        for glyph in app.glyphs:
            self.assertEqual(glyphs.visible, False)
