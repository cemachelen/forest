"""Tokens to communicate between components

Module constants prevent string literals from polluting the code base
and factory methods are more readable, ``action = SET.file_name.to('file.nc')``
is easier to read than ``action = ("SET", "file_name", "file.nc")``

Or indeed, ``MOVE.pressure.forward``, is probably easier to read
than ``("MOVE", "pressure", "forward")`` although both generate the same tuple
"""

__all__ = [
    "ADD",
    "REMOVE",
    "SET",
    "MOVE"
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


class Motion(object):
    def __init__(self, props):
        self._props = props

    @property
    def forward(self):
        return self._props + ("forward",)

    @property
    def backward(self):
        return self._props + ("backward",)


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


ADD = Action("ADD")
REMOVE = Action("REMOVE")
SET = Action("SET")
MOVE = Action("MOVE")
