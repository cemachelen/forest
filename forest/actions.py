"""Tokens to communicate between components

Module constants prevent string literals from polluting the code base
and factory methods are more readable, ``action = actions.set_file('file.nc')``
is easier to read than ``action = (actions.SET_FILE, 'file.nc')``

.. note:: There has been some usage of Python's import system
          to reduce boiler-plate code
"""

def closure(token):
    def factory(value):
        return (token, value)
    return factory


class ActionsModule(object):
    def __init__(self, tokens):
        for token in tokens:
            self.__setattr__(token.upper(), token)
            self.__setattr__(token.lower(), closure(token.upper()))

import sys
sys.modules[__name__] = ActionsModule([
    "SET_FILE",
    "SET_FILE_NAMES",
    "SET_VARIABLE",
    "SET_VARIABLES",
    "SET_VALID_TIME",
    "SET_VALID_TIMES",
    "SET_INITIAL_TIME",
    "SET_INITIAL_TIMES",
    "SET_PRESSURE",
    "SET_PRESSURES",
])
