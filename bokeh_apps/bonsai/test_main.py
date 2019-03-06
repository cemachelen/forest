# pylint: disable=missing-docstring, invalid-name
import unittest
import os
import yaml
import main


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.path = "test-config.yaml"

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_load(self):
        with open(self.path, "w") as stream:
            yaml.dump({"lon_range": [-10, 10]}, stream)
        result = main.Config.load(self.path).lon_range
        expect = [-10, 10]
        self.assertEqual(expect, result)

    def test_default_lon_range(self):
        self.check_default("lon_range", [-180, 180])

    def test_default_lat_range(self):
        self.check_default("lat_range", [-80, 80])

    def test_default_title(self):
        self.check_default("title", "Bonsai - miniature Forest")

    def check_default(self, attr, expect):
        result = getattr(main.Config(), attr)
        self.assertEqual(expect, result)
