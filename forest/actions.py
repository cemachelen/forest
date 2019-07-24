"""Tokens to communicate between components

Module constants prevent string literals from polluting the code base
and factory methods are more readable, ``action = SET.file_name.to('file.nc')``
is easier to read than ``action = ("SET", "file_name", "file.nc")``
"""

__all__ = [
    "ADD",
    "REMOVE",
    "SET",
    "move"
]


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


def move(item_key, items_key, direction):
    """Helper to represent MOVE action as tuple

    :param item_key: key in state to be moved
    :param items_key: key in state containing multiple values
    :param direction: either increase/decrease
    """
    return ("MOVE", item_key, "GIVEN", items_key, direction)


ADD = Action("ADD")
REMOVE = Action("REMOVE")
SET = Action("SET")
