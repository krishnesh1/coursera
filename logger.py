"""Logging for training and prediction.  Every function takes a log_dir so tests
can write somewhere else than production."""
import csv, os, time, uuid


def _write(log_dir, filename, header, row):
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, filename)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)
    return path


TRAIN_HEADER = ["unique_id", "timestamp", "country", "model", "rmse", "baseline_rmse",
                "n_rows", "runtime_seconds", "version"]
PREDICT_HEADER = ["unique_id", "timestamp", "country", "date", "predicted_revenue",
                  "model_version", "runtime_seconds"]


def update_train_log(log_dir, country, model, rmse, baseline_rmse, n_rows, runtime, version):
    return _write(log_dir, "train.log", TRAIN_HEADER,
                  [uuid.uuid4().hex, time.time(), country, model, round(rmse, 4),
                   round(baseline_rmse, 4), n_rows, round(runtime, 4), version])


def update_predict_log(log_dir, country, date, predicted, version, runtime):
    return _write(log_dir, "predict.log", PREDICT_HEADER,
                  [uuid.uuid4().hex, time.time(), country, date, round(float(predicted), 4),
                   version, round(runtime, 4)])
