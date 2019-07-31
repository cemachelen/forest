"""Tokens to communicate between components

Module constants prevent string literals from polluting the code base
and factory methods are more readable, ``action = SET.file_name.to('file.nc')``
is easier to read than ``action = ("SET", "file_name", "file.nc")``
"""
import datetime as dt

__all__ = [
    "ActionLog",
    "ADD",
    "REMOVE",
    "SET",
    "Move"
]


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


class Assign(object):
    def __init__(self, props):
        self._props = props

    def to(self, value):
        return self._props + (value,)


class Append(object):
    def __init__(self, props):
        self._props = props

    def by_name(self, name, value):
        return self._props + (name, value)


class Remove(object):
    def __init__(self, props):
        self._props = props

    def by_name(self, name):
        return self._props + (name,)


class Action(object):
    def __init__(self, verb):
        self._verb = verb

    def __getattr__(self, key):
        if self._verb == "SET":
            return Assign((self._verb, key))
        elif self._verb == "MOVE":
            return Motion((self._verb, key))
        elif self._verb == "ADD":
            return Append((self._verb, key))
        elif self._verb == "REMOVE":
            return Remove((self._verb, key))
        else:
            raise Exception("Unknown verb: {}".format(self._verb))

    __getitem__ = __getattr__


class Move(object):
    """Helper to represent MOVE action as tuple

    :param item_key: key in state to be moved
    :param items_key: key in state containing multiple values
    """
    def __init__(self, item_key, items_key):
        self.item_key = item_key
        self.items_key = items_key

    @property
    def increment(self):
        return ("MOVE", self.item_key, self.items_key, "INCREMENT")

    @property
    def decrement(self):
        return ("MOVE", self.item_key, self.items_key, "DECREMENT")


ADD = Action("ADD")
REMOVE = Action("REMOVE")
SET = Action("SET")
