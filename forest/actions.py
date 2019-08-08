"""Data structures to decouple communication between components
"""
import datetime as dt
from forest.middleware import middleware


__all__ = [
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


class Log(object):
    """Middleware to capture history of actions"""
    def __init__(self):
        self.actions = []

    @middleware
    def __call__(self, store, next_method, action):
        self.actions.append(action)
        next_method(action)

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
