"""sys.path adjusted to replicate bokeh serve conditions"""
import unittest
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../forest"))
import main


class TestMain(unittest.TestCase):
    @unittest.skip("integration test")
    def test_main_program(self):
        main.main()
