import unittest
import unittest.mock
import forest


class TestNavigator(unittest.TestCase):
    def setUp(self):
        self.navigator = forest.Navigator()

    def test_navigator_variables_dropdown_width(self):
        result = self.navigator.dropdowns["variable"].width
        expect = None
        self.assertEqual(expect, result)

    def test_on_click_forward(self):
        listener = unittest.mock.Mock()
        self.navigator.subscribe(listener)
        self.navigator.on_click("pressure", "pressures", "next")()
        expect = forest.actions.Move("pressure", "pressures").increment
        listener.assert_called_once_with(expect)


class TestActionCreator(unittest.TestCase):
    def test_decouple_actions_from_navigator(self):
        listener = unittest.mock.Mock()
        navigator = forest.Navigator()
        navigator.notify = forest.actions.prepend(navigator.notify, 'navigate')
        navigator.subscribe(listener)
        navigator.on_click("pressure", "pressures", "next")()
        expect = ("navigate", "MOVE", "pressure", "pressures", "INCREMENT")
        listener.assert_called_once_with(expect)

    def test_combine_reducers(self):
        reducer = forest.combine_reducers(navigate=forest.reducers.navigate)
        result = reducer({}, ('navigate', 'set', 'hello', 'world'))
        expect = {
            'navigate': {
                'hello': 'world'
            }
        }
        self.assertEqual(expect, result)

    def test_navigate_reducer_given_navigate_set_action(self):
        result = forest.reducers.navigate({}, ('navigate', 'set', 'hello', 'world'))
        expect = {
            'hello': 'world'
        }
        self.assertEqual(expect, result)

    def test_navigate_reducer_given_arbitrary_set_action(self):
        result = forest.reducers.navigate({}, ('other', 'set', 'hello', 'world'))
        expect = {
        }
        self.assertEqual(expect, result)

    def test_combine_reducers_given_unhandled_action(self):
        reducer = forest.combine_reducers(navigate=forest.reducers.navigate)
        result = reducer({}, ('unknown', 'set', 'hello', 'world'))
        expect = {'navigate': {}}
        self.assertEqual(expect, result)

    def test_combine_reducers_given_two_actions_related_to_different_sections(self):
        reducer = forest.combine_reducers(
                navigate=forest.reducers.navigate,
                preset=forest.reducers.preset)
        state = {}
        for action in [
                ('navigate', 'set', 'hello', 'world'),
                ('preset', 'set', 'hello', 'world')]:
            state = reducer(state, action)
        result = state
        expect = {
            'navigate': {
                'hello': 'world'
            },
            'preset': {
                'hello': 'world'
            }
        }
        self.assertEqual(expect, result)

    def test_render_only_sees_part_of_state(self):
        def reducer(state, action):
            return state

        state = {
            'navigate': {
                'hello': 'world'
            }
        }
        render = unittest.mock.Mock()
        store = forest.Store(reducer, state=state)
        store.subscribe(forest.subtree(render, 'navigate'))
        store.dispatch(None)
        expect = {"hello": "world"}
        render.assert_called_once_with(expect)
