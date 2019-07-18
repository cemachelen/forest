"""sys.path adjusted to replicate bokeh serve conditions"""
import unittest
import yaml
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../forest"))
import main


class TestMain(unittest.TestCase):
    def setUp(self):
        self.config_file = "test-config.yml"

    def tearDown(self):
        if os.path.exists(self.config_file):
            os.remove(self.config_file)

    # @unittest.skip("integration test")
    def test_main_program_given_minimal_config_file(self):
        data = {
            "models": [
            ]
        }
        with open(self.config_file, "w") as stream:
            stream.write(yaml.safe_dump(data))

        main.main([
            "--config", self.config_file
        ])

    @unittest.skip("integration test")
    def test_main_file_system_navigation(self):
        main.main([
            "file.nc"
        ])
