import unittest
from unittests.helpers import IsolatedHome, sample_data_dir
import app as api


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = sample_data_dir()

    def setUp(self):
        self.ctx = IsolatedHome(); self.ctx.__enter__()
        self.client = api.app.test_client()

    def tearDown(self):
        self.ctx.__exit__(None, None, None)

    def _train(self):
        return self.client.post("/train", json={"test": True, "data_dir": self.data, "top_n": 3})

    def test_health(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_train_endpoint(self):
        r = self._train()
        self.assertEqual(r.status_code, 200)
        self.assertGreaterEqual(r.get_json()["trained"], 2)

    def test_train_bad_data_dir_returns_400(self):
        r = self.client.post("/train", json={"test": True, "data_dir": "/nope"})
        self.assertEqual(r.status_code, 400)

    def test_predict_all_countries_and_single_country(self):
        self._train()
        for country in ("all", "united_kingdom"):
            r = self.client.post("/predict", json={"country": country, "date": "2018-10-01",
                                                   "test": True, "data_dir": self.data})
            self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
            body = r.get_json()
            self.assertEqual(body["country"], country)
            self.assertIn("predicted_revenue_next_30_days", body)

    def test_predict_get_query_string(self):
        self._train()
        r = self.client.get(f"/predict?country=all&date=2018-10-01&test=true&data_dir={self.data}")
        self.assertEqual(r.status_code, 200)

    def test_predict_missing_fields_400_and_unknown_country_404(self):
        self._train()
        self.assertEqual(self.client.post("/predict", json={"country": "all"}).status_code, 400)
        r = self.client.post("/predict", json={"country": "atlantis", "date": "2018-10-01", "test": True,
                                               "data_dir": self.data})
        self.assertEqual(r.status_code, 404)

    def test_monitor_reports_models_and_predictions(self):
        self._train()
        self.client.post("/predict", json={"country": "all", "date": "2018-10-01", "test": True,
                                           "data_dir": self.data})
        body = self.client.get("/monitor?test=true").get_json()
        self.assertIn("all", body["models"])
        self.assertEqual(body["predictions"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
