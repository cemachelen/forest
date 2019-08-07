"""Tokens to communicate between components

Module constants prevent string literals from polluting the code base
and factory methods are more readable, ``action = SET.file_name.to('file.nc')``
is easier to read than ``action = ("SET", "file_name", "file.nc")``
"""
import datetime as dt

__all__ = [
    "ActionLog",
]


SET_ITEM = "SET_ITEM"
NEXT_ITEM = "NEXT_ITEM"
PREVIOUS_ITEM = "PREVIOUS_ITEM"


def set_item(key, value):
    return {**locals(), **dict(kind=SET_ITEM)}


def next_item(item_key, items_key):
    return {**locals(), **dict(kind=NEXT_ITEM)}


def previous_item(item_key, items_key):
    return {**locals(), **dict(kind=PREVIOUS_ITEM)}


class ActionLog(object):
    """Middleware to capture history of actions"""
    def __init__(self):
        self.actions = []

    def __call__(self, store):
        def inner(next_method):
            def inner_most(action):
                self.actions.append(action)
                next_method(action)
            return inner_most
        return inner

    def summary(self, state):
        """Print a summary of recorded actions"""
        print("{} {} action(s) logged".format(
            dt.datetime.now(), len(self.actions)))
        n = 3
        if len(self.actions) > n:
            print("...")
            for action in self.actions[-n:]:
                print(action)
        else:
            for action in self.actions:
                print(action)
