# pylint: disable=missing-docstring, invalid-name
import unittest
import unittest.mock
import os
import yaml
import main


class TestEnvironment(unittest.TestCase):
    def tearDown(self):
        for variable in ["FOREST_MODEL_DIR"]:
            if variable in os.environ:
                del os.environ[variable]

    def test_parse_env_given_forest_model_dir(self):
        os.environ["FOREST_MODEL_DIR"] = "/some/dir"
        result = main.parse_env().model_dir
        expect = "/some/dir"
        self.assertEqual(expect, result)

    def test_parse_env_default_forest_model_dir(self):
        result = main.parse_env().model_dir
        expect = None
        self.assertEqual(expect, result)


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.path = "test-config.yaml"
        self.config = main.Config()

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_merge_configs(self):
        dict_0 = dict(
            lat_range=[-5, 5],
            lon_range=[-180, 180])
        with open(self.path, "w") as stream:
            yaml.dump({"lon_range": [-10, 10]}, stream)
            dict_1 = main.Config.load_dict(self.path)
        merged = main.Config.merge(dict_0, dict_1)
        self.assertEqual(merged.lon_range, [-10, 10])
        self.assertEqual(merged.lat_range, [-5, 5])

    def test_load(self):
        with open(self.path, "w") as stream:
            yaml.dump({"lon_range": [-10, 10]}, stream)
        result = self.config.load(self.path).lon_range
        expect = [-10, 10]
        self.assertEqual(expect, result)

    def test_model_names(self):
        settings = {
            "models": [
                {"name": "A"},
                {"name": "B"}
            ]
        }
        with open(self.path, "w") as stream:
            yaml.dump(settings, stream)
        result = main.Config.load(self.path).model_names
        expect = ["A", "B"]
        self.assertEqual(expect, result)

    def test_model_pattern(self):
        config = main.Config(models=[
            {"name": "A", "pattern": "a.nc"},
            {"name": "B", "pattern": "b.nc"}])
        result = config.model_pattern("A")
        expect = "a.nc"
        self.assertEqual(expect, result)

    def test_model_dir(self):
        self.check_kwarg("model_dir", "/some/dir")

    def test_default_model_dir_returns_none(self):
        self.check_default("model_dir", None)

    def test_default_lon_range(self):
        self.check_default("lon_range", [-180, 180])

    def test_default_lat_range(self):
        self.check_default("lat_range", [-80, 80])

    def test_default_title(self):
        self.check_default("title", "Bonsai - miniature Forest")

    def test_default_models(self):
        self.check_default("models", [])

    def check_default(self, attr, expect):
        result = getattr(main.Config(), attr)
        self.assertEqual(expect, result)

    def check_kwarg(self, attr, expect):
        result = getattr(main.Config(**{attr: expect}), attr)
        self.assertEqual(expect, result)


class TestOnClick(unittest.TestCase):
    def test_on_click(self):
        state = main.State()
        view = unittest.mock.Mock()
        state.register(view)
        state.on("model")("A")
        view.notify.assert_called_once_with({"model": "A"})

    def test_state_merges_streams(self):
        state = main.State()
        view = unittest.mock.Mock()
        state.register(view)
        state.on("model")("A")
        state.on("date")("B")
        state.on("model")("B")
        calls = [
            unittest.mock.call({"model": "A"}),
            unittest.mock.call({"model": "A", "date": "B"}),
            unittest.mock.call({"model": "B", "date": "B"}),
        ]
        view.notify.assert_has_calls(calls)
