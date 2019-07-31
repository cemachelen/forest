import unittest
import forest
import netCDF4
import os
import datetime as dt


def eida50(dataset, time=1, longitude=1, latitude=1):
    """EIDA50 dimensions, variables and attributes"""
    dataset.createDimension("time", time)
    dataset.createDimension("longitude", longitude)
    dataset.createDimension("latitude", latitude)
    var = dataset.createVariable("time", "d", ("time",))
    var.axis = "T"
    var.units = "hours since 1970-01-01 00:00:00"
    var.standard_name = "time"
    var.calendar = "gregorian"
    var = dataset.createVariable(
        "longitude", "f", ("longitude",))
    var.axis = "X"
    var.units = "degrees_east"
    var.standard_name = "longitude"
    var = dataset.createVariable(
        "latitude", "f", ("latitude",))
    var.axis = "Y"
    var.units = "degrees_north"
    var.standard_name = "latitude"
    var = dataset.createVariable(
        "data", "f", ("time", "latitude", "longitude"), fill_value=-99999.)
    var.standard_name = "toa_brightness_temperature"
    var.long_name = "toa_brightness_temperature"
    var.units = "K"
    var.comment = "Infra-red channel top of atmosphere brightness temperature, central wavelength of 10.80 microns, Geostationary projection"
    var.keywords = "Infra-red, brightness temperature, MSG, SEVIRI"
    dataset.title = "TOA brightness temperature"
    dataset.history = "Created: 2019-04-16T08:18:00Z"
    dataset.Conventions = "CF-1.7"
    dataset.institution = "Met Office UK"
    dataset.acknowledgement = "EUMETSAT"
    dataset.standard_name_vocabulary = "CF Standard Name Table v27"
    dataset.platform = "MSG"
    dataset.instrument = "SEVIRI"



class TestMiddleware(unittest.TestCase):
    def setUp(self):
        self.path = "test-file.nc"
        self.store = forest.Store(forest.reducer, middlewares=[
            forest.FileSystem()
        ])

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_file_system_given_file_sets_variables(self):
        with netCDF4.Dataset(self.path, "w") as dataset:
            eida50(dataset)
        self.store.dispatch(("SET", "file_name", self.path))
        result = self.store.state["variables"]
        expect = ['time', 'longitude', 'latitude', 'data']
        self.assertEqual(expect, result)

    def test_file_system_given_variable_sets_valid_times(self):
        times = [dt.datetime(2019, 1, 1)]
        with netCDF4.Dataset(self.path, "w") as dataset:
            eida50(dataset, time=len(times))
            var = dataset.variables["time"]
            var[:] = netCDF4.date2num(times, units=var.units)
        self.store.dispatch(("SET", "file_name", self.path))
        self.store.dispatch(("SET", "variable", "data"))
        result = self.store.state["valid_times"]
        expect = times
        self.assertEqual(expect, result)
