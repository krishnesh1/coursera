import os, unittest
import numpy as np
from unittests.helpers import IsolatedHome, sample_data_dir
import model, config
from data_ingestion import ingest_data


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = sample_data_dir()

    def test_ingest_returns_series_per_country_and_all(self):
        df = ingest_data(self.data_dir)
        self.assertIn("all", set(df.country))
        self.assertIn("united_kingdom", set(df.country))
        self.assertTrue({"date", "country", "revenue"}.issubset(df.columns))

    def test_ingest_missing_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            ingest_data("/no/such/folder")

    def test_train_saves_models_and_compares_several(self):
        with IsolatedHome() as home:
            res = model.model_train(self.data_dir, test=True, countries=["all"])
            self.assertEqual(res[0]["country"], "all")
            self.assertIn(res[0]["best_model"], {"ridge", "random_forest", "gradient_boosting"})
            self.assertGreater(len(res[0]["all_scores"]), 1)
            self.assertTrue(os.path.exists(os.path.join(home, "test_models", "model-all.joblib")))

    def test_predict_all_and_single_country(self):
        with IsolatedHome():
            model.model_train(self.data_dir, test=True, countries=["all", "france"])
            a = model.model_predict("all", "2018-10-01", test=True, data_dir=self.data_dir)
            f = model.model_predict("France", "2018-10-01", test=True, data_dir=self.data_dir)
            self.assertGreaterEqual(a["predicted_revenue_next_30_days"], 0)
            self.assertEqual(f["country"], "france")
            self.assertNotEqual(a["predicted_revenue_next_30_days"], f["predicted_revenue_next_30_days"])

    def test_predict_unknown_country_raises(self):
        with IsolatedHome():
            with self.assertRaises(FileNotFoundError):
                model.model_predict("atlantis", "2018-10-01", test=True, data_dir=self.data_dir)

    def test_predict_too_little_history_raises(self):
        with IsolatedHome():
            model.model_train(self.data_dir, test=True, countries=["all"])
            with self.assertRaises(ValueError):
                model.model_predict("all", "2018-01-10", test=True, data_dir=self.data_dir)

    def test_features_use_only_past_data(self):
        rev = np.arange(1, 201, dtype=float)
        x = model.features_for(rev, 99)
        rev2 = rev.copy(); rev2[100:] = 0
        self.assertEqual(x, model.features_for(rev2, 99))

    def test_test_mode_paths_differ_from_production(self):
        self.assertNotEqual(config.get_paths(True)["model_dir"], config.get_paths(False)["model_dir"])
        self.assertNotEqual(config.get_paths(True)["log_dir"], config.get_paths(False)["log_dir"])


if __name__ == "__main__":
    unittest.main()
