"""Template for how to build components"""


class Component(object):
    def __call__(self, state):
        self.render(state)

    def render(self, state):
        state = self.select(state, [
            "file_name",
            "variable",
            "valid_time"])
        print(state)

    @staticmethod
    def select(state, props):
        return {k: state[k] for k in props if k in state}
