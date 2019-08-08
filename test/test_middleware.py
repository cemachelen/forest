import unittest
import forest


@forest.middleware
def echo(store, next_method, action):
    next_method(action)
    next_method(action)


def reducer(state, action):
    state = dict(state)
    if "action" in state:
        state["action"].append(action)
    else:
        state["action"] = [action]
    return state


class TestMiddleware(unittest.TestCase):
    def test_echo_middleware(self):
        store = forest.Store(reducer, middlewares=[echo])
        store.dispatch("ACTION")
        result = store.state
        expect = {"action": ["ACTION", "ACTION"]}
        self.assertEqual(expect, result)

    def test_log_middleware(self):
        log = forest.actions.Log()
        store = forest.Store(reducer, middlewares=[log])
        store.dispatch("ACTION")
        result = log.actions
        expect = ["ACTION"]
        self.assertEqual(expect, result)
