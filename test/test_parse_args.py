"""sys.path adjusted to replicate bokeh serve environment"""
import unittest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../forest"))
import parse_args


class TestParseArgs(unittest.TestCase):
    def test_directory_returns_none_by_default(self):
        args = parse_args.parse_args([
            "--database", "file.db",
            "--config", "file.yml"
        ])
        result = args.directory
        expect = None
        self.assertEqual(expect, result)

    def test_directory_returns_value(self):
        args = parse_args.parse_args([
            "--directory", "/some",
            "--database", "file.db",
            "--config", "file.yml"
        ])
        result = args.directory
        expect = "/some"
        self.assertEqual(expect, result)

    def test_parse_args_given_a_single_file(self):
        args = parse_args.parse_args(["file.nc"])
        result = args.files
        expect = ["file.nc"]
        self.assertEqual(expect, result)

    def test_parse_args_given_config(self):
        args = parse_args.parse_args(["--config", "file.yml"])
        result = args.config_file
        expect = "file.yml"
        self.assertEqual(expect, result)
