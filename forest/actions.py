"""Tuples used to communicate between components

Constants prevent string literals from polluting the code base
and factory methods are more readable, ``action = actions.set_file('file.nc')``
is easier to read than ``action = (actions.SET_FILE, 'file.nc')``
"""


SET_FILE = "set file"
SET_FILE_NAMES = "set file names"
SET_VARIABLE = "set variable"
SET_VARIABLES = "set variables"
SET_VALID_TIMES = "set valid times"


def set_file(value):
    return (SET_FILE, value)


def set_file_names(value):
    return (SET_FILE_NAMES, value)


def set_variable(value):
    return (SET_VARIABLE, value)


def set_variables(value):
    return (SET_VARIABLES, value)


def set_valid_times(value):
    return (SET_VALID_TIMES, value)
