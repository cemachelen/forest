import unittest
import os
import datetime as dt
import dateutil.parser
import pandas as pd


def read_earth_networks(csv_files):
    if isinstance(csv_files, str):
        csv_files = [csv_files]
    frames = []
    for csv_file in csv_files:
        frame = pd.read_csv(
            csv_file,
            parse_dates=[1],
            converters={0: flash_type},
            usecols=[0, 1, 2, 3],
            names=["flash_type", "date", "longitude", "latitude"],
            header=None)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def flash_type(value):
    return {
        "0": "CG",
        "1": "IC",
        "9": "Keep alive"
    }.get(value, value)


class TestEarthNetworks(unittest.TestCase):
    def setUp(self):
        self.paths = [
            "sample-earth-networks-0.txt",
            "sample-earth-networks-1.txt"]
        self.date = dt.datetime(2019, 3, 28, 0, 56, 0, 52000)

    def tearDown(self):
        for path in self.paths:
            if os.path.exists(path):
                os.remove(path)

    def test_read_earth_networks(self):
        path = self.paths[0]
        entry = "0,20190328T005600.052,+29.3603000,+007.6370000,-000025503,000,00000,006,001"
        with open(path, "w") as stream:
            stream.write(entry + "\n")
        result = read_earth_networks(path)
        print(result)
        expect = pd.DataFrame({
            "flash_type": ["CG"],
            "date": [self.date],
            "longitude": [29.3603],
            "latitude": [7.637]
        })
        pd.testing.assert_frame_equal(expect, result)

    def test_read_multiple_files(self):
        entries = [
            "0,20190328T005600.052,+29.3603000,+007.6370000,-000025503,000,00000,006,001",
            "1,20190328T005600.052,+29.3603000,+007.6370000,-000025503,000,00000,006,001",
            "9,20190328T005600.052,+29.3603000,+007.6370000,-000025503,000,00000,006,001",
        ]
        for path in self.paths:
            with open(path, "w") as stream:
                stream.write("\n".join(entries) + "\n")
        result = read_earth_networks(self.paths)
        expect = pd.DataFrame({
            "flash_type": 2 * ["CG", "IC", "Keep alive"],
            "date": 6 * [self.date],
            "longitude": 6 * [29.3603],
            "latitude": 6 * [7.637]
        })
        pd.testing.assert_frame_equal(expect, result)
