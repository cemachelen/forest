"""Collection of sub-reducers"""
from collections import OrderedDict


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
    kind, *rest = action
    if kind.lower() == 'set':
        return set_reducer(state, action)
    elif kind.lower() == 'move':
        return move_reducer(state, action)
    return state


@action_key('preset')
def preset(state, action):
    kind, *rest = action
    state = dict(state)
    if kind.lower() == 'add':
        name, settings = rest
        tree = OrderedDict([(p["name"], p) for p in state])
        data = dict(settings)
        data["name"] = name
        tree[name] = data
        state['presets'] = list(tree.values())
        return state
    elif kind.lower() == 'remove':
        _, name = rest
        state['presets'] = [
                p for p in state if p["name"] != name]
        return state
    return state


def set_reducer(state, action):
    state = dict(state)
    method, attr, value = action
    if method.lower() == "set":
        state[attr] = value
    return state


def move_reducer(state, action):
    _, item_key, items_key, direction = action
    if items_key in state:
        item = state.get(item_key, None)
        items = state[items_key]
        if isinstance(item, str):
            items = [str(v) for v in items]
        if direction.lower() == "increment":
            state[item_key] = increment(items, item)
        else:
            state[item_key] = decrement(items, item)
    return state


def increment(items, item):
    if item is None:
        return max(items)
    items = list(sorted(items))
    i = items.index(item)
    return items[(i + 1) % len(items)]


def decrement(items, item):
    if item is None:
        return min(items)
    items = list(sorted(items))
    i = items.index(item)
    return items[i - 1]
