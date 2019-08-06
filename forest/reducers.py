"""Collection of sub-reducers"""


def action_key(key):
    def inner(reducer):
        def inner_most(state, action):
            _key, *action = action
            if key.lower() != _key.lower():
                return state
            return reducer(state, action)
        return inner_most
    return inner


@action_key('navigate')
def navigate(state, action):
    return set_reducer(state, action)


@action_key('preset')
def preset(state, action):
    return set_reducer(state, action)


def set_reducer(state, action):
    state = dict(state)
    method, attr, value = action
    if method.lower() == "set":
        state[attr] = value
    return state
