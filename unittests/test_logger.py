import os, unittest
import pandas as pd
from unittests.helpers import IsolatedHome, sample_data_dir
import logger, model


class LoggerTests(unittest.TestCase):
    def test_train_log_written_with_header_and_row(self):
        with IsolatedHome() as home:
            d = os.path.join(home, "logs")
            logger.update_train_log(d, "all", "ridge", 10.5, 12.0, 300, 1.2, "1.0")
            df = pd.read_csv(os.path.join(d, "train.log"))
            self.assertEqual(list(df.columns), logger.TRAIN_HEADER)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.country[0], "all")

    def test_predict_log_appends(self):
        with IsolatedHome() as home:
            d = os.path.join(home, "logs")
            logger.update_predict_log(d, "france", "2018-10-01", 1234.5, "1.0", 0.02)
            logger.update_predict_log(d, "all", "2018-10-01", 9999.0, "1.0", 0.03)
            df = pd.read_csv(os.path.join(d, "predict.log"))
            self.assertEqual(len(df), 2)
            self.assertEqual(df.unique_id.nunique(), 2)

    def test_training_and_prediction_write_logs_in_test_folder_only(self):
        with IsolatedHome() as home:
            data = sample_data_dir()
            model.model_train(data, test=True, countries=["all"])
            model.model_predict("all", "2018-10-01", test=True, data_dir=data)
            self.assertTrue(os.path.exists(os.path.join(home, "test_logs", "train.log")))
            self.assertTrue(os.path.exists(os.path.join(home, "test_logs", "predict.log")))
            self.assertFalse(os.path.exists(os.path.join(home, "logs")))


if __name__ == "__main__":
    unittest.main()
