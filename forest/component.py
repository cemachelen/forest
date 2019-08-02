"""Template for how to build components"""
import bokeh.models


def restrict(state, props):
    return {k: state[k] for k in props if k in state}


class Component(object):
    def __call__(self, state):
        self.render(state)

    def render(self, state):
        state = restrict(state, [
            "file_name",
            "variable",
            "valid_time"])
        print(state)


class Message(object):
    def __init__(self):
        self.div = bokeh.models.Div()

    def __call__(self, state):
        state = restrict(state, ["file_name"])
        self.div.text = state.get("file_name", "")


def image(render, load):
    def callback(state):
        render(load(state))
    return callback
