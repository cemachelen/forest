"""User interface helper methods/classes"""
import netCDF4


def time_bounds(dataset):
    for name in ["time_2", "time"]:
        if name not in dataset.variables:
            continue
        units = dataset.variables[name].units
        bounds = dataset.variables[name + "_bnds"][:]
        return bounds, units
